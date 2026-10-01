#!/usr/bin/env python3
"""Explicitly fetch the named public checkpoint and the pinned official source.

No execution occurs on import. Downloads are streamed, checksum-verified and
published without replacing an existing model. Terms are not bypassed: an HTTP
401/403 is an error, and no gated checkpoint fallback is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

OFFICIAL_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
MODEL_REPO = "ussoewwin/SAM3.1-ConvRot-INT8"
MODEL_REVISION = "32c74f1ec2a9b615ddb9346d2155a999296c23b7"
MODEL_FILE = "sam3.1_multiplex_convrot_int8.safetensors"


class TokenSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward a Hugging Face bearer token to a CDN or other host."""
    def redirect_request(self, request, response, code, message, headers, new_url):
        redirected = super().redirect_request(request, response, code, message, headers, new_url)
        if redirected is not None and urllib.parse.urlparse(new_url).hostname != "huggingface.co":
            redirected.remove_header("Authorization")
        return redirected


def download_model(directory: Path) -> dict[str, object]:
    metadata_url = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true"
    headers = {"User-Agent": "sam31-gguf-source-fetch/0.1"}
    # Only this named token is read, and it is never written to logs or files.
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(TokenSafeRedirectHandler())
    with opener.open(urllib.request.Request(metadata_url, headers=headers), timeout=60) as response:
        metadata = json.load(response)
    file = next((entry for entry in metadata["siblings"] if entry["rfilename"] == MODEL_FILE), None)
    if file is None:
        raise RuntimeError("Pinned model revision does not contain the expected checkpoint")
    lfs = file.get("lfs", {})
    expected_hash = lfs.get("sha256") or lfs.get("oid")
    expected_size = lfs.get("size") or file.get("size")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or not isinstance(expected_size, int):
        raise RuntimeError("Hugging Face did not supply the required checkpoint SHA-256 and size")
    if expected_size <= 0 or expected_size > 4 * 1024**3:
        raise RuntimeError("Unexpected checkpoint size")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / MODEL_FILE
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists; refusing to overwrite: {destination}")
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{metadata['sha']}/{MODEL_FILE}"
    descriptor, name = tempfile.mkstemp(prefix=MODEL_FILE + ".", suffix=".partial", dir=directory)
    temporary = Path(name)
    checksum, size = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as output, opener.open(
            urllib.request.Request(url, headers=headers), timeout=120
        ) as response:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > expected_size:
                    raise RuntimeError("Download exceeded the checkpoint's declared size")
                checksum.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if size != expected_size or checksum.hexdigest() != expected_hash:
            raise RuntimeError("Checkpoint size/SHA-256 verification failed")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"repository": MODEL_REPO, "revision": metadata["sha"], "file": MODEL_FILE,
              "sha256": expected_hash, "size": size, "verified": True}
    (directory / "checkpoint.provenance.json").write_text(json.dumps(report, indent=2))
    return report


def download_source(destination: Path) -> dict[str, str]:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace source directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", "https://github.com/facebookresearch/sam3.git"], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", OFFICIAL_COMMIT], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != OFFICIAL_COMMIT:
        raise RuntimeError("Source checkout does not match the pinned revision")
    return {"path": str(destination), "revision": actual}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if args.models_dir is None and args.source_dir is None:
        parser.error("Specify --models-dir and/or --source-dir")
    try:
        report = {}
        if args.models_dir is not None:
            report["model"] = download_model(args.models_dir)
        if args.source_dir is not None:
            report["source"] = download_source(args.source_dir)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        print(f"Asset fetch failed; no successful verification is claimed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
