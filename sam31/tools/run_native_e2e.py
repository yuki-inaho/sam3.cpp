#!/usr/bin/env python3
"""Opt-in trained-checkpoint E2E: C++ input generation, image, video and validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sam31_gguf.format import GGUFFile, sha256_file
from verify_synthetic_results import verify

EXPECTED_GGUF = "e59728d18a3e0f6bfd603eee86490b6116b9141d4e4cd3acdc0309ea443ec370"
EXPECTED_SOURCE = "21fc2308ef4bf82a1d372e147170e2d48ed8f3688922b4eaa3e341828333debd"


def run(binary: Path, model: Path, output: Path, threads: int) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    store = GGUFFile(model)
    if store.is_fixture or store.metadata["sam31.source.sha256"] != EXPECTED_SOURCE:
        raise ValueError("E2E requires the pinned trained SAM 3.1 ConvRot INT8 checkpoint")
    if sha256_file(model) != EXPECTED_GGUF:
        raise ValueError("Trained GGUF bytes do not match the verified lossless conversion")
    output.mkdir(parents=True)
    environment = dict(os.environ, PATH="/NO_EXECUTABLES", SAM31_PYTHON="/NOT_PYTHON",
                       SAM31_SCRIPT="/NOT_A_SCRIPT", OPENBLAS_NUM_THREADS=str(threads))
    commands = [
        ["synthetic", "--output", output / "inputs", "--frames", 6],
        ["image", "--image", output / "inputs/image.png", "--output", output / "image"],
        ["video", "--frames", output / "inputs/frames", "--output", output / "video"],
    ]
    for command in commands:
        mode = command[0]
        if mode != "synthetic":
            command += ["--model", model, "--threads", threads, "--cache-mb", 4096,
                        "--point", "1:0.25:0.32", "--point", "2:0.73:0.68", "--profile"]
        with (output / f"{mode}.log").open("w") as log:
            subprocess.run([str(binary), *map(str, command)], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    image = verify(output / "image", output / "inputs", 1, 0.8)
    video = verify(output / "video", output / "inputs", 6, 0.8)
    report = {"passed": True, "checkpoint_source_sha256": EXPECTED_SOURCE,
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "gguf_sha256": EXPECTED_GGUF, "threads": threads, "image": image, "video": video,
              "image_elapsed_ms": json.loads((output / "image/report.json").read_text())["elapsed_ms"],
              "video_elapsed_ms": json.loads((output / "video/report.json").read_text())["elapsed_ms"]}
    (output / "e2e.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    try:
        if not 1 <= args.threads <= 256:
            raise ValueError("threads must be 1..256")
        binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
        if args.output is None:
            with tempfile.TemporaryDirectory(prefix="sam31-e2e-") as temporary:
                report = run(binary, model, Path(temporary) / "results", args.threads)
        else:
            report = run(binary, model, args.output.resolve(), args.threads)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Native E2E failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
