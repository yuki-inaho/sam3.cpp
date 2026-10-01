"""Prepare a pinned official source tree without editing the user's checkout."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .format import FormatError, sha256_file

OFFICIAL_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
REFERENCE_COMMIT = "c10425825604b2ad6939f0fc4e0966b4381ee473"
MARKER = ".sam31-prepared.json"


def python_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path)
            for path in sorted((root / "sam3").rglob("*.py"))}


def verify_prepared_source(root: Path) -> dict[str, Any]:
    marker = root / MARKER
    if not marker.is_file():
        raise FileNotFoundError(f"CPU source is not prepared: {marker}; use prepare-source")
    data = json.loads(marker.read_text(encoding="utf-8"))
    if data.get("official_commit") != OFFICIAL_COMMIT or data.get("patch_schema") != 1:
        raise FormatError("CPU source does not match the pinned SAM 3.1 revision")
    if data.get("python_sha256") != python_hashes(root):
        raise FormatError("Prepared source changed after preparation; regenerate it explicitly")
    if not (root / "sam3/model/video_tracking_multiplex.py").is_file():
        raise FormatError("Prepared source lacks the SAM 3.1 multiplex tracker")
    return data


def prepare_source(source: Path, output: Path) -> dict[str, Any]:
    from ._cpu_source_patch import create_sam31_cpu_source_copy

    source, output = source.resolve(strict=True), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Official checkout and generated tree must be separate, non-nested directories")
    if output.exists():
        return verify_prepared_source(output)
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormatError("Source must be a Git checkout at the documented official commit") from exc
    if revision != OFFICIAL_COMMIT or dirty:
        raise FormatError(f"Expected clean source at {OFFICIAL_COMMIT}; got {revision}, dirty={bool(dirty)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        result = create_sam31_cpu_source_copy(source, temporary)
        # The reference patcher disables antialiasing for ONNX. This backend is
        # PyTorch-only, so restore the original model's antialiasing semantics.
        memory = temporary / "sam3/model/memory.py"
        text = memory.read_text(encoding="utf-8")
        if text.count("antialias=False,") != 1:
            raise FormatError("Cannot identify the reference-only antialiasing rewrite")
        memory.write_text(text.replace("antialias=False,", "antialias=True,", 1), encoding="utf-8")
        report = {
            "official_commit": OFFICIAL_COMMIT, "reference_commit": REFERENCE_COMMIT,
            "patch_schema": 1, "antialias_preserved": True,
            "modified_files": [str(p.relative_to(temporary)) for p in result.modified_files],
            "python_sha256": python_hashes(temporary),
        }
        (temporary / MARKER).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        os.rename(temporary, output)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
