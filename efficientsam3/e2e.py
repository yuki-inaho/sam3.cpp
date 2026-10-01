"""Real EV-M native acceptance: image and six ordered artificial frame motions."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops

from contract import CONTEXT, MODEL_SHA256, SOURCE_REVISION
from reference import Grounding, Text, Vision, build_reference


def run_e2e(source, checkpoint, gguf, binary, output, threads=16):
    torch.set_num_threads(threads)
    model = build_reference(source, checkpoint)
    vision, text, grounding = Vision(model), Text(model), Grounding(model)
    binary, gguf, output = Path(binary).resolve(), Path(gguf).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(Path(source) / "sam3/assets/dog_person.jpeg").convert("RGB")
    image = image.resize((1008, 1008), Image.Resampling.BILINEAR)
    tokens = model.backbone.language_backbone.tokenizer(["dog"], context_length=CONTEXT)
    reports = []
    filenames = []
    for frame in range(6):
        filename = output / f"frame_{frame:06d}.png"
        ImageChops.offset(image, frame * 4, frame * 2).save(filename)
        filenames.append(filename)
    destination = output / "native_sequence"
    destination.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment["PATH"] = "/NO_EXECUTABLES"
    environment.pop("SAM3_EFFICIENT_TRACE_DIR", None)
    command = [
        str(binary),
        "--model",
        str(gguf),
        "--text",
        "dog",
        "--output",
        str(destination),
        "--threads",
        str(threads),
    ]
    for filename in filenames:
        command.extend(["--image", str(filename)])
    begin = time.perf_counter()
    process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=1200)
    if process.returncode:
        raise ValueError(f"Native sequence failed: {process.stderr}")
    sequence_ms = (time.perf_counter() - begin) * 1000
    native_frames = json.loads((destination / "annotations.json").read_text())["frames"]
    if len(native_frames) != 6 or [f["frame_index"] for f in native_frames] != list(range(6)):
        raise ValueError("Native frame order/count mismatch")
    with torch.inference_mode():
        text_output = text(tokens)
        for frame in range(6):
            current = ImageChops.offset(image, frame * 4, frame * 2)
            filename = output / f"frame_{frame:06d}.png"
            current.save(filename)
            data = (
                np.asarray(current, dtype=np.float32).transpose(2, 0, 1)[None] / 255 - 0.5
            ) / 0.5
            start = time.perf_counter()
            features = vision(torch.from_numpy(data))
            expected = grounding(*features, *text_output)
            reference_ms = (time.perf_counter() - start) * 1000
            boxes, logits, presence, masks = [v.numpy() for v in expected]
            if not all(np.isfinite(v).all() for v in (boxes, logits, presence, masks)):
                raise ValueError("Non-finite reference output")
            scores = (
                1 / (1 + np.exp(-logits)) * 1 / (1 + np.exp(-presence)).reshape(1, 1, 1)
            ).reshape(-1)
            selected = np.flatnonzero(scores > 0.5)
            if not len(selected):
                raise ValueError("Trained reference produced no confident detections")
            actual = native_frames[frame]
            detections = actual["detections"]
            if len(detections) != len(selected):
                raise ValueError("Native/reference selected detection counts differ")
            reference_masks = []
            for query in selected:
                value = (
                    np.asarray(
                        Image.fromarray(masks[0, query]).resize(
                            current.size, Image.Resampling.BILINEAR
                        )
                    )
                    > 0
                )
                if not value.any():
                    raise ValueError("Empty reference mask")
                reference_masks.append(value)
            remaining = set(range(len(reference_masks)))
            ious = []
            for detection in detections:
                if not np.isfinite([detection["score"], *detection["box_xyxy"]]).all():
                    raise ValueError("Non-finite native score/box")
                actual_mask = np.asarray(Image.open(destination / detection["mask"])) > 0
                candidates = {
                    index: float(
                        np.logical_and(actual_mask, reference_masks[index]).sum()
                        / np.logical_or(actual_mask, reference_masks[index]).sum()
                    )
                    for index in remaining
                }
                best = max(candidates, key=candidates.get)
                if candidates[best] < 0.90:
                    raise ValueError(f"Native mask IoU below 0.90: {candidates[best]}")
                remaining.remove(best)
                ious.append(
                    {
                        "query_id": int(selected[best]),
                        "iou": candidates[best],
                        "score_error": abs(float(scores[selected[best]]) - detection["score"]),
                        "foreground_pixels": int(actual_mask.sum()),
                    }
                )
            report = {
                "frame_index": frame,
                "masks": ious,
                "native_ms": actual["elapsed_ms"],
                "reference_ms": reference_ms,
            }
            reports.append(report)
            print(f"native frame {frame}: min IoU {min(m['iou'] for m in ious):.6f}", flush=True)
    result = {
        "passed": True,
        "backend": "ggml-cpu",
        "sequence_mode": "independent_detection",
        "checkpoint_sha256": MODEL_SHA256,
        "source_revision": SOURCE_REVISION,
        "native_path": "/NO_EXECUTABLES",
        "threads": threads,
        "frames": reports,
        "native_sequence_process_ms": sequence_ms,
    }
    (output / "e2e.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/efficientsam3/source")
    p.add_argument("--checkpoint", default="models/efficientsam3_ev_m.pt")
    p.add_argument("--gguf", default="models/efficientsam3_ev_m.gguf")
    p.add_argument("--binary", default="build/examples/efficientsam3")
    p.add_argument("--output", default="evidence/efficient-e2e")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    run_e2e(a.source, a.checkpoint, a.gguf, a.binary, a.output, a.threads)


if __name__ == "__main__":
    main()
