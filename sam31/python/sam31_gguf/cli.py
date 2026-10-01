"""Command-line entry points. Heavy inference imports are lazy and explicit."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .format import GGUFFile
from .runner import PointPrompt


def doctor(model: Path | None, source_root: Path | None) -> dict[str, Any]:
    modules = ("numpy", "PIL", "torch", "torchvision", "timm", "einops", "iopath",
               "ftfy", "huggingface_hub", "decord", "pycocotools")
    present = {name: importlib.util.find_spec(name) is not None for name in modules}
    python_ready = (3, 10) <= sys.version_info[:2] <= (3, 12)
    report: dict[str, Any] = {
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "modules": present, "storage_dependencies_ready": present["numpy"],
        "reference_python_version_supported": python_ready,
        "native_sam31_graph_status": "not_evaluated_by_reference_gate",
    }
    if model is not None:
        try:
            container = GGUFFile(model)
            report["gguf"] = {"tensor_count": len(container.tensors), "test_fixture": container.is_fixture}
        except (OSError, ValueError) as exc:
            report["gguf_error"] = str(exc)
    if source_root is not None:
        from .source import verify_prepared_source
        try:
            prepared = verify_prepared_source(source_root)
            report["source_commit"] = prepared["official_commit"]
        except (OSError, ValueError) as exc:
            report["source_error"] = str(exc)
    report["reference_dependencies_ready"] = python_ready and all(present.values())
    report["real_inference_prerequisites_ready"] = (
        report["reference_dependencies_ready"] and "source_commit" in report
        and "gguf" in report and not report["gguf"]["test_fixture"]
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SAM 3.1 lossless GGUF tools and explicit PyTorch reference inference")
    commands = result.add_subparsers(dest="command", required=True)
    for mode in ("image", "video"):
        command = commands.add_parser(mode, help="Official CPU reference backend (not native ggml)")
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--source-root", type=Path, required=True, help="Prepared official CPU source directory")
        command.add_argument("--frames" if mode == "video" else "--image", type=Path, required=True)
        command.add_argument("--point", type=PointPrompt.parse, action="append", required=True)
        command.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--max-frames", type=int)
        command.add_argument("--ablate-memory", action="store_true", help="Measure the effect of zeroing temporal memory once")
        command.add_argument("--oracle-safetensors", action="store_true", help="Explicit comparison oracle: read original SafeTensors instead of GGUF")
    command = commands.add_parser("synthetic", help="Generate inputs and separate ground truth; no model predictions")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--frame-count", type=int, default=6)
    command = commands.add_parser("prepare-source", help="Prepare a clean pinned official checkout for CPU")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("doctor", help="Report capabilities without running or downloading a model")
    command.add_argument("--model", type=Path)
    command.add_argument("--source-root", type=Path)
    command.add_argument("--storage-only", action="store_true")
    command = commands.add_parser("dod", help="Run real-model acceptance gates; missing assets never count as passing")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--gguf", type=Path, required=True)
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--binary", type=Path, required=True)
    command.add_argument("--work-dir", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--timeout-seconds", type=int, default=3600)
    command.add_argument("--reference-only", action="store_true", help="Explicitly scope the gate to the reference backend, not completion of the requested native port")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ("image", "video"):
            from .runner import run_inference
            report = run_inference(
                model_path=args.model, source_root=args.source_root,
                input_path=args.frames if args.command == "video" else args.image,
                output=args.output, prompts=args.point, video=args.command == "video",
                threads=args.threads, max_frames=args.max_frames,
                ablate_memory=args.ablate_memory, safetensors_oracle=args.oracle_safetensors,
            )
        elif args.command == "synthetic":
            from .synthetic import make_scene
            report = make_scene(args.output, frame_count=args.frame_count)
        elif args.command == "prepare-source":
            from .source import prepare_source
            report = prepare_source(args.source, args.output)
        elif args.command == "doctor":
            report = doctor(args.model, args.source_root)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["storage_dependencies_ready" if args.storage_only else "real_inference_prerequisites_ready"] else 2
        elif args.command == "dod":
            from .dod import run_gate
            if args.timeout_seconds < 1:
                raise ValueError("Timeout must be positive")
            return run_gate(checkpoint=args.checkpoint, gguf=args.gguf, source_root=args.source_root,
                            binary=args.binary.resolve(), work_dir=args.work_dir.resolve(),
                            report_path=args.report, threads=args.threads,
                            reference_only=args.reference_only, timeout_seconds=args.timeout_seconds)
        else:
            raise ValueError("Unknown command")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as exc:
        print(f"sam31 reference backend: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
