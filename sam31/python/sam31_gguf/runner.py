"""Point-prompted images and stateful multi-image video, with explicit provenance."""
from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .format import FormatError, sha256_file
from .model import build_reference_model

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PointPrompt:
    object_id: int
    x: float
    y: float

    @classmethod
    def parse(cls, text: str) -> "PointPrompt":
        try:
            fields = text.split(":")
            if len(fields) != 3:
                raise ValueError("wrong number of fields")
            result = cls(int(fields[0]), float(fields[1]), float(fields[2]))
        except ValueError as exc:
            raise ValueError("Point must be ID:X:Y with normalized coordinates") from exc
        if not 0 <= result.object_id <= 2**31 - 1:
            raise ValueError("Object ID must be between 0 and 2^31-1")
        if not all(math.isfinite(n) and 0 <= n <= 1 for n in (result.x, result.y)):
            raise ValueError("Point coordinates must be finite and within [0, 1]")
        return result


def validate_prompts(prompts: list[PointPrompt]) -> None:
    if not 1 <= len(prompts) <= 16:
        raise ValueError("This entry point supports 1-16 point-prompted objects in one multiplex bucket")
    if len({point.object_id for point in prompts}) != len(prompts):
        raise ValueError("Each object must have exactly one positive point and a unique ID")
    for point in prompts:
        PointPrompt.parse(f"{point.object_id}:{point.x}:{point.y}")


def natural_key(path: Path) -> tuple[tuple[int, Any], ...]:
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r"(\d+)", path.name))


def input_frames(path: Path, *, video: bool, max_frames: int | None = None) -> list[Path]:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if video:
        if not path.is_dir():
            raise FileNotFoundError(f"Frame directory does not exist: {path}")
        frames = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                        key=natural_key)
    else:
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            raise FileNotFoundError(f"Unsupported or missing input image: {path}")
        frames = [path]
    frames = frames[:max_frames] if max_frames else frames
    if not frames:
        raise ValueError("No supported image frames found")
    if video and len(frames) < 2:
        raise ValueError("Video validation requires at least two images; use image for a single frame")
    return frames


def stage_frames(frames: list[Path], directory: Path) -> tuple[int, int]:
    """Normalize names/formats for the official JPEG-folder loader.

    Existing JPEG bytes are copied unchanged. Other RGB formats are encoded as
    quality-100, non-subsampled JPEG; both GGUF and oracle use this same policy.
    """
    directory.mkdir(parents=True, exist_ok=False)
    dimensions = None
    for index, path in enumerate(frames):
        with Image.open(path) as image:
            if dimensions is None:
                dimensions = image.size
            if image.size != dimensions or image.width < 1 or image.height < 1:
                raise ValueError("All frames must have identical nonzero dimensions")
            if image.width * image.height > 25_000_000:
                raise ValueError("Input image exceeds the 25-million-pixel safety limit")
            target = directory / f"{index:06d}.jpg"
            if image.format == "JPEG" and image.mode == "RGB":
                shutil.copyfile(path, target)
            else:
                image.convert("RGB").save(target, quality=100, subsampling=0)
    if dimensions is None:
        raise ValueError("Cannot stage an empty sequence")
    return dimensions


class CallAudit:
    """Count actual module calls and optionally measure temporal-memory influence."""
    def __init__(self, model: Any, *, ablate_memory: bool):
        self.calls: Counter[str] = Counter()
        self.originals: list[tuple[Any, Any]] = []
        self.memory_delta: float | None = None
        self.ablate_memory = ablate_memory
        self.modules = {
            "image_encoder": model.backbone,
            "memory_attention": model.transformer.encoder,
            "multiplex_decoder": model.sam_mask_decoder,
            "memory_encoder": model.maskmem_backbone,
        }

    def __enter__(self) -> "CallAudit":
        for name, module in self.modules.items():
            original = module.forward
            self.originals.append((module, original))

            def wrapped(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
                self.calls[_name] += 1
                result = _original(*args, **kwargs)
                if _name == "memory_attention" and self.ablate_memory and self.memory_delta is None:
                    import torch
                    if args or "memory" not in kwargs or "memory_image" not in kwargs:
                        raise RuntimeError("Official attention call does not expose the audited memory inputs")
                    changed = dict(kwargs)
                    changed["memory"] = torch.zeros_like(kwargs["memory"])
                    changed["memory_image"] = torch.zeros_like(kwargs["memory_image"])
                    ablated = _original(**changed)
                    self.memory_delta = float((result["memory"] - ablated["memory"]).abs().mean())
                    if not math.isfinite(self.memory_delta):
                        raise RuntimeError("Non-finite temporal-memory ablation result")
                return result

            module.forward = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        for module, original in reversed(self.originals):
            module.forward = original


def checked_frame(entry: Any, ids: list[int], shape: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    frame_index, actual_ids, _, masks, scores = entry
    if list(actual_ids) != ids:
        raise RuntimeError(f"Object identities changed: expected {ids}, got {list(actual_ids)}")
    logits = masks.detach().cpu().float().numpy()
    scores_np = scores.detach().cpu().float().numpy()
    width, height = shape
    if logits.shape != (len(ids), 1, height, width):
        raise RuntimeError(f"Unexpected mask shape: {logits.shape}")
    if scores_np.shape[0] != len(ids):
        raise RuntimeError("Object score count does not match the requested objects")
    if not np.isfinite(logits).all() or not np.isfinite(scores_np).all():
        raise RuntimeError("Inference produced NaN or infinity")
    return int(frame_index), logits.copy(), scores_np.copy()


def infer_frames(model: Any, staged: Path, prompts: list[PointPrompt], *, count: int,
                 dimensions: tuple[int, int], video: bool, ablate_memory: bool) -> tuple[list[Any], dict[str, Any]]:
    import torch
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

    ids = [p.object_id for p in prompts]

    def new_state() -> dict[str, Any]:
        return VideoTrackingMultiplexDemo.init_state(
            model, video_path=str(staged), offload_video_to_cpu=True, offload_state_to_cpu=True
        )

    with torch.inference_mode(), CallAudit(model, ablate_memory=ablate_memory) as audit:
        prompt_state = new_state()
        for point in prompts:
            model.add_new_points(prompt_state, 0, point.object_id,
                                 torch.tensor([[point.x, point.y]], dtype=torch.float32),
                                 torch.tensor([1], dtype=torch.int32), clear_old_points=True)
        first = checked_frame(next(model.propagate_in_video(
            prompt_state, 0, 0, False, tqdm_disable=True)), ids, dimensions)
        if not video:
            results = [first]
            bucket_report = {"mode": "interactive-image"}
        else:
            # Recondition all point-derived masks together, as in the supplied
            # SAM 3.1 reference test. This exercises a real shared 16-slot bucket.
            masks = torch.from_numpy((first[1][:, 0] > 0).astype(np.float32))
            del prompt_state
            gc.collect()
            state = new_state()
            model.add_new_masks(state, 0, ids, masks)
            multiplex = state["multiplex_state"]
            expected_assignment = list(range(len(ids))) + [-1] * (16 - len(ids))
            if multiplex.num_buckets != 1 or multiplex.assignments != [expected_assignment]:
                raise RuntimeError("Objects were not assigned to the expected shared multiplex bucket")
            bucket_report = {"mode": "shared-multiplex-video", "buckets": 1,
                             "capacity": 16, "assignments": multiplex.assignments}
            results = [checked_frame(entry, ids, dimensions) for entry in model.propagate_in_video(
                state, 0, count - 1, False, tqdm_disable=True)]
        if [entry[0] for entry in results] != list(range(count)):
            raise RuntimeError(f"Missing, duplicated, or out-of-order frames: {[entry[0] for entry in results]}")
        if video:
            for operation in ("memory_attention", "multiplex_decoder", "memory_encoder"):
                if audit.calls[operation] < 1:
                    raise RuntimeError(f"Video did not execute its required {operation} module")
        report = {"module_calls": dict(audit.calls), "multiplex": bucket_report,
                  "memory_ablation_mean_abs": audit.memory_delta}
    return results, report


def run_inference(*, model_path: Path, source_root: Path, input_path: Path, output: Path,
                  prompts: list[PointPrompt], video: bool, threads: int = 4,
                  max_frames: int | None = None, ablate_memory: bool = False,
                  safetensors_oracle: bool = False) -> dict[str, Any]:
    validate_prompts(prompts)
    frames = input_frames(input_path, video=video, max_frames=max_frames)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite inference output: {output}")
    model, loading = build_reference_model(model_path, source_root, threads=threads,
                                           safetensors_oracle=safetensors_oracle)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        dimensions = stage_frames(frames, temporary / "input_frames")
        started = time.perf_counter()
        results, trace = infer_frames(model, temporary / "input_frames", prompts, count=len(frames),
                                     dimensions=dimensions, video=video, ablate_memory=ablate_memory)
        elapsed = time.perf_counter() - started
        ids = [point.object_id for point in prompts]
        np.savez_compressed(temporary / "result.npz",
                            frame_indices=np.asarray([r[0] for r in results], dtype=np.int64),
                            object_ids=np.asarray(ids, dtype=np.int64),
                            mask_logits=np.stack([r[1] for r in results]),
                            object_scores=np.stack([r[2] for r in results]))
        mask_dir = temporary / "masks"
        mask_dir.mkdir()
        for frame, masks, _ in results:
            for object_index, object_id in enumerate(ids):
                Image.fromarray((masks[object_index, 0] > 0).astype(np.uint8) * 255).save(
                    mask_dir / f"{frame:06d}_object_{object_id}.png")
        report = {**loading, **trace, "frame_count": len(frames), "object_ids": ids,
                  "image_dimensions": list(dimensions), "elapsed_inference_seconds": elapsed,
                  "input_files": [str(p.resolve()) for p in frames],
                  "checkpoint_sha256": sha256_file(model_path), "synthetic_weights": False,
                  "mask_threshold": 0.0,
                  "preprocessing": "official normalization and resize; RGB JPEG staging at quality=100, subsampling=0"}
        (temporary / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        os.rename(temporary, output)
        return report
    finally:
        del model
        gc.collect()
        if temporary.exists():
            shutil.rmtree(temporary)
