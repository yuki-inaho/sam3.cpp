"""Deterministic two-object image/video inputs and separate ground truth.

These files are test INPUTS, never fabricated SAM predictions. Ground truth is
used only by the scorer, not supplied to the point-prompted inference command.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def make_scene(output: Path, *, frame_count: int = 6) -> dict[str, Any]:
    if not 2 <= frame_count <= 32:
        raise ValueError("Synthetic scene supports 2-32 frames")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frames = output / "frames"
    truth_dir = output / "ground_truth"
    frames.mkdir()
    truth_dir.mkdir()
    width, height = 640, 384
    yy, xx = np.mgrid[:height, :width]
    background = np.stack([
        40 + (xx // 32 + yy // 32) % 2 * 7,
        48 + (xx // 32 + yy // 32) % 2 * 7,
        56 + (xx // 32 + yy // 32) % 2 * 7,
    ], axis=-1).astype(np.uint8)
    masks = []
    for index in range(frame_count):
        phase = index / max(5, frame_count - 1)
        circle = (150 + round(40 * phase), 110 + round(10 * phase))
        square = (452 - round(30 * phase), 272 - round(15 * phase))
        image = Image.fromarray(background.copy())
        draw = ImageDraw.Draw(image)
        circle_box = (circle[0] - 46, circle[1] - 46, circle[0] + 46, circle[1] + 46)
        square_box = (square[0] - 44, square[1] - 41, square[0] + 44, square[1] + 41)
        draw.ellipse(circle_box, fill=(223, 68, 49))
        draw.rectangle(square_box, fill=(55, 133, 224))
        draw.polygon([(315, 50), (345, 50), (330, 78)], fill=(71, 127, 82))
        image.save(frames / f"{index:06d}.jpg", quality=100, subsampling=0)
        object_masks = []
        for object_id, box, shape in ((1, circle_box, "ellipse"), (2, square_box, "rectangle")):
            truth = Image.new("L", (width, height), 0)
            getattr(ImageDraw.Draw(truth), shape)(box, fill=255)
            truth.save(truth_dir / f"{index:06d}_object_{object_id}.png")
            object_masks.append(np.asarray(truth) > 0)
        masks.append(np.stack(object_masks)[:, None])
    shutil.copyfile(frames / "000000.jpg", output / "image.jpg")
    np.savez_compressed(output / "ground_truth.npz", masks=np.stack(masks),
                        object_ids=np.asarray([1, 2], dtype=np.int64))
    scene = {"kind": "synthetic-test-input-not-model-output", "frame_count": frame_count,
             "width": width, "height": height, "object_ids": [1, 2],
             "points": [f"1:{150 / width}:{110 / height}", f"2:{452 / width}:{272 / height}"],
             "ground_truth_used_as_model_input": False, "moving_objects": ["circle", "rectangle"]}
    (output / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return scene
