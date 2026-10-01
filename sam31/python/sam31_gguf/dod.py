"""A fail-closed real-checkpoint acceptance gate, separate from unit tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .format import GGUFFile, SafeTensorFile, compare_stores, convert
from .source import verify_prepared_source
from .synthetic import make_scene

# Fixed before execution; not relaxed automatically in response to failures.
THRESHOLDS = {
    "reference_mask_iou_min": 0.999,
    "reference_logit_max_abs": 0.0001,
    "image_ground_truth_iou_min": 0.60,
    "video_ground_truth_iou_min": 0.50,
    "memory_ablation_mean_abs_min": 0.000001,
}


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def score_outputs(actual_dir: Path, oracle_dir: Path, truth_path: Path, *, video: bool) -> dict[str, Any]:
    with np.load(actual_dir / "result.npz", allow_pickle=False) as actual, np.load(
        oracle_dir / "result.npz", allow_pickle=False
    ) as oracle, np.load(truth_path, allow_pickle=False) as truth:
        frames = 6 if video else 1
        for data in (actual, oracle):
            if not np.array_equal(data["frame_indices"], np.arange(frames)):
                raise AssertionError("Frame indices differ from the six-frame/single-image scenario")
            if not np.array_equal(data["object_ids"], [1, 2]):
                raise AssertionError("Two-object identities were not maintained")
        a, b = actual["mask_logits"], oracle["mask_logits"]
        if a.shape != b.shape or a.shape != (frames, 2, 1, 384, 640):
            raise AssertionError(f"Unexpected output dimensions: {a.shape}, {b.shape}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError("Non-finite mask logits")
        scores_a, scores_b = actual["object_scores"], oracle["object_scores"]
        if not np.isfinite(scores_a).all() or not np.isfinite(scores_b).all():
            raise AssertionError("Non-finite object scores")
        np.testing.assert_allclose(scores_a, scores_b, atol=1e-4, rtol=1e-5)
        reference_ious, truth_ious = [], []
        for frame in range(frames):
            for obj in range(2):
                predicted = a[frame, obj, 0] > 0
                if predicted.sum() == 0 or predicted.all():
                    raise AssertionError(f"Empty or all-image mask at frame={frame}, object={obj + 1}")
                reference_ious.append(mask_iou(predicted, b[frame, obj, 0] > 0))
                truth_ious.append(mask_iou(predicted, truth["masks"][frame, obj, 0]))
        error = float(np.max(np.abs(a - b)))
        if min(reference_ious) < THRESHOLDS["reference_mask_iou_min"] or error > THRESHOLDS["reference_logit_max_abs"]:
            raise AssertionError(f"GGUF/source reference mismatch: IoU={min(reference_ious)}, error={error}")
        threshold = THRESHOLDS["video_ground_truth_iou_min" if video else "image_ground_truth_iou_min"]
        if min(truth_ious) < threshold:
            raise AssertionError(f"Ground-truth IoU below fixed gate: {min(truth_ious)} < {threshold}")
    report = json.loads((actual_dir / "report.json").read_text())
    if report["synthetic_weights"] or not report["full_model_key_coverage"]:
        raise AssertionError("Test fixture or incomplete weight loading cannot satisfy DoD")
    if not video:
        for operation in ("image_encoder", "multiplex_decoder"):
            if report["module_calls"].get(operation, 0) < 1:
                raise AssertionError(f"Image did not execute its required {operation} module")
    if video:
        calls = report["module_calls"]
        for operation in ("image_encoder", "memory_attention", "multiplex_decoder", "memory_encoder"):
            minimum = frames if operation == "image_encoder" else frames - 1
            if calls.get(operation, 0) < minimum:
                raise AssertionError(f"Insufficient real module calls for {operation}: {calls}")
        if report["multiplex"].get("buckets") != 1:
            raise AssertionError("Video did not use one shared multiplex bucket")
        delta = report["memory_ablation_mean_abs"]
        if delta is None or delta <= THRESHOLDS["memory_ablation_mean_abs_min"]:
            raise AssertionError(f"No measured temporal-memory influence: {delta}")
    return {"frame_count": frames, "object_count": 2, "reference_iou_min": min(reference_ious),
            "logit_max_abs": error, "ground_truth_iou_min": min(truth_ious),
            "ground_truth_ious": truth_ious, "module_calls": report["module_calls"],
            "memory_ablation_mean_abs": report["memory_ablation_mean_abs"]}


def run_gate(*, checkpoint: Path, gguf: Path, source_root: Path, binary: Path,
             work_dir: Path, report_path: Path, threads: int = 4,
             reference_only: bool = False, timeout_seconds: int = 3600) -> int:
    report: dict[str, Any] = {
        "request_complete": False, "reference_e2e_passed": False,
        "scope": "historical-reference-backend-only",
        "thresholds": THRESHOLDS, "gates": [],
        "native_sam31_graph_status": "not_evaluated_by_reference_gate",
    }

    def gate(name: str, status: str, detail: Any) -> None:
        report["gates"].append({"name": name, "status": status, "detail": detail})

    started = time.perf_counter()
    exit_code = 2
    try:
        missing = [str(path) for path in (checkpoint, source_root, binary) if not path.exists()]
        if missing:
            gate("real_model_prerequisites", "BLOCKED", {"missing_paths": missing})
            return 2
        if work_dir.exists():
            raise FileExistsError(f"Use a new DoD work directory: {work_dir}")
        verify_prepared_source(source_root)
        source = SafeTensorFile(checkpoint)
        if source.metadata.get("sam31.test_fixture") == "true":
            raise ValueError("Synthetic weights are forbidden in the real-model acceptance gate")
        if not gguf.exists():
            conversion = convert(checkpoint, gguf)
        else:
            conversion = {"existing_gguf": str(gguf)}
        container = GGUFFile(gguf)
        if container.is_fixture:
            raise ValueError("Synthetic GGUF weights are forbidden in DoD")
        conversion["exact_payload_comparison"] = compare_stores(source, container)
        gate("lossless_real_checkpoint_conversion", "PASS", conversion)
        native_inspect = subprocess.run([str(binary), "inspect", str(gguf), "--read-all"],
                                        capture_output=True, text=True, check=True, timeout=timeout_seconds)
        gate("native_gguf_read", "PASS", json.loads(native_inspect.stdout))
        work_dir.mkdir(parents=True)
        scene = make_scene(work_dir / "synthetic_inputs")
        gate("independent_synthetic_inputs", "PASS", scene)
        entry = Path(__file__).resolve().parents[2] / "tools/sam31.py"
        env = dict(os.environ, SAM31_PYTHON=sys.executable)
        for mode in ("image", "video"):
            input_flag = "--frames" if mode == "video" else "--image"
            input_path = work_dir / "synthetic_inputs" / ("frames" if mode == "video" else "image.jpg")
            for kind, model_path in (("source_oracle", checkpoint), ("gguf", gguf)):
                output = work_dir / f"{mode}_{kind}"
                command = ([sys.executable, str(entry), mode] if kind == "source_oracle" else [str(binary), mode])
                command += ["--model", str(model_path), "--source-root", str(source_root),
                            input_flag, str(input_path), "--output", str(output), "--threads", str(threads)]
                for point in scene["points"]:
                    command += ["--point", point]
                if kind == "source_oracle":
                    command += ["--oracle-safetensors"]
                if mode == "video" and kind == "gguf":
                    command += ["--ablate-memory"]
                with (work_dir / f"{mode}_{kind}.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=env, timeout=timeout_seconds)
                if completed.returncode:
                    raise RuntimeError(f"Real inference failed for {mode}/{kind}; see {log.name}")
            metrics = score_outputs(work_dir / f"{mode}_gguf", work_dir / f"{mode}_source_oracle",
                                    work_dir / "synthetic_inputs/ground_truth.npz", video=mode == "video")
            gate(f"real_{mode}_inference_and_reference_parity", "PASS", metrics)
        report["reference_e2e_passed"] = True
        if reference_only:
            gate("native_cpp_graph", "OUT_OF_SCOPE_EXPLICITLY", "Reference-only gate was requested")
            exit_code = 0
        else:
            gate("native_cpp_graph", "NOT_EVALUATED", "Use native CTest acceptance and evidence-native; this gate evaluates only the Python reference")
            exit_code = 1
        return exit_code
    except Exception as exc:
        gate("execution", "FAIL", f"{type(exc).__name__}: {exc}")
        exit_code = 1
        return exit_code
    finally:
        if not any(item["name"] == "native_cpp_graph" for item in report["gates"]):
            gate("native_cpp_graph", "NOT_EVALUATED", "This historical reference gate does not evaluate the native C++ graph")
        report["elapsed_seconds"] = time.perf_counter() - started
        report["exit_code"] = exit_code
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=False))
