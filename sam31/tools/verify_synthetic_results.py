#!/usr/bin/env python3
"""Validate real inference outputs against colors of the independent input scene.

This runs after C++ inference and never supplies masks to the model. It checks
artifact structure, finite logits, stable IDs and per-object segmentation IoU.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(output: Path, inputs: Path, frames: int, min_iou: float) -> dict:
    report = json.loads((output / "report.json").read_text())
    require(report["complete"] and not report["model"]["test_fixture"], "real checkpoint required")
    require(not report["python_inference"] and report["model"]["backend"] == "native-cpp-cpu", "native C++ inference required")
    require(len(report["frames"]) == frames, "unexpected frame count")
    metrics = []
    for index, frame in enumerate(report["frames"]):
        require(frame["frame_index"] == index, "frame order differs")
        require([obj["object_id"] for obj in frame["objects"]] == ["1", "2"], "object IDs differ")
        image_path = inputs / ("image.png" if frames == 1 else f"frames/{index:06}.png")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        require((frame["height"], frame["width"]) == (height, width), "output dimensions differ from input")
        logits = np.fromfile(output / frame["logits_file"], dtype="<f4").reshape(2, height, width)
        require(np.isfinite(logits).all(), "non-finite logits")
        for slot, obj in enumerate(frame["objects"]):
            mask = np.asarray(Image.open(output / obj["mask"]))
            require(mask.shape == (height, width) and np.isin(mask, [0, 255]).all(), "invalid mask dimensions or values")
            pred = mask > 0
            require(np.array_equal(pred, logits[slot] > 0) and pred.any(), "mask is empty or differs from logits")
            truth = (image == ([225, 45, 30] if slot == 0 else [40, 195, 225])).all(axis=2)
            iou = float(np.count_nonzero(pred & truth) / np.count_nonzero(pred | truth))
            require(iou >= min_iou, f"frame {index} object {slot+1}: IoU {iou} < {min_iou}")
            metrics.append({"frame": index, "object_id": obj["object_id"], "iou": iou,
                            "foreground_pixels": int(pred.sum())})
    require(len(list(output.glob("*.png"))) == frames * 2, "unexpected number of masks")
    if frames > 1:
        require(report["stats"]["memory_attention"] == frames - 1, "memory attention count differs")
        require(report["stats"]["memory_blocks"] == 4 * (frames - 1), "memory block count differs")
        require(report["model"]["used_tensors"] == report["model"]["required_tensors"] == 867, "required video parameters were not used")
    else:
        require(report["model"]["used_tensors"] == 608, "required image parameters were not used")
    return {"passed": True, "frames": frames, "objects": 2, "threshold": min_iou,
            "minimum_iou": min(item["iou"] for item in metrics),
            "source_sha256": report["model"]["source_sha256"], "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--min-iou", type=float, default=0.8)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.inputs, args.frames, args.min_iou), indent=2))


if __name__ == "__main__":
    main()
