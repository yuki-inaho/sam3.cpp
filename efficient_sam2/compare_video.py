"""Compare native masks/logits with an independently produced Python oracle.

Public numerical parity evidence only: agreement with another implementation
is not a claim about human-reviewed tracking accuracy or identity recovery.
"""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--expected-compute", choices=["restored_fp32_cpu", "experimental_ggml_q8_0_cpu"])
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--mean-iou", type=float, default=0.99)
    parser.add_argument("--min-iou", type=float, default=0.98)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_path = args.native / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    if args.expected_compute and (not runtime or runtime.get("compute") != args.expected_compute):
        raise ValueError("native compute provenance missing or mismatched")
    reference_report_path = args.reference / "report.json"
    reference_report = json.loads(reference_report_path.read_text()) if reference_report_path.exists() else None
    reference_variant = reference_report.get("variant") if reference_report else None
    if runtime and isinstance(reference_variant, dict):
        reference_model = reference_variant.get("gguf_loaded_sha256")
        if reference_model and runtime.get("model_sha256") != reference_model:
            raise ValueError("native and reference GGUF contents differ")
    metrics = []
    for frame in range(args.frames):
        prefix = f"frame_{frame:05d}_object_{args.object_id:03d}"
        actual = np.asarray(Image.open(args.native / f"{frame:05d}.png")) > 0
        expected = np.asarray(Image.open(args.reference / f"{prefix}.png")) > 0
        if actual.shape != expected.shape:
            raise ValueError(f"frame {frame}: mismatched image geometry")
        union = np.logical_or(actual, expected).sum()
        iou = float(np.logical_and(actual, expected).sum() / union) if union else 1.0
        actual_logits = np.fromfile(args.native / f"{frame:05d}.f32", dtype="<f4")
        expected_logits = np.load(args.reference / f"{prefix}_low_res.npy").reshape(-1)
        if actual_logits.shape != expected_logits.shape or actual_logits.size != 256 * 256:
            raise ValueError(f"frame {frame}: missing or malformed raw logits")
        metrics.append({"frame": frame, "mask_iou": iou,
                        "logit_rmse": float(np.sqrt(np.mean((actual_logits - expected_logits) ** 2))),
                        "finite": bool(np.isfinite(actual_logits).all())})
    report = {"count": len(metrics), "mean_iou": float(np.mean([m["mask_iou"] for m in metrics])),
              "min_iou": min(m["mask_iou"] for m in metrics), "frames": metrics,
              "protocol": "independent single-object contiguous forward box initialization"}
    report["native_runtime"] = runtime
    report["reference_variant"] = reference_variant
    report["reference_report_sha256"] = hashlib.sha256(reference_report_path.read_bytes()).hexdigest() if reference_report else None
    report["passed"] = (report["mean_iou"] >= args.mean_iou and report["min_iou"] >= args.min_iou
                        and all(m["finite"] for m in metrics))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("frames", "native_runtime", "reference_variant")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
