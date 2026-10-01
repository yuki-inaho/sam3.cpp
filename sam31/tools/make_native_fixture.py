#!/usr/bin/env python3
"""Generate explicitly UNTRAINED weights for the native SAM 3.1 graph.

This is fixture preparation, not an inference backend. All image/video execution
is performed by the C++ binary. The 32/4/2 layer counts and 16 slots are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from make_test_fixture import write_fixture
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sam31_gguf.format import convert, compare_stores, SafeTensorFile, GGUFFile


def hadamard(size: int) -> np.ndarray:
    base = np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=np.float32) / 2
    result = np.ones((1,1), dtype=np.float32)
    while result.shape[0] < size:
        result = np.kron(result, base)
    return result


def generate(manifest: dict, output: Path) -> dict:
    if not manifest["test_fixture"]:
        raise ValueError("Only the explicitly tagged smoke manifest is allowed")
    tensors: dict[str, np.ndarray] = {}
    count = rotated = 0
    for spec in manifest["tensors"]:
        name, shape = spec["name"], tuple(spec["shape"])
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        kind = spec["kind"]
        if kind == "norm":
            value = np.ones(shape, dtype=np.float32)
        elif kind == "gamma":
            value = np.full(shape, 0.1, dtype=np.float32)
        elif kind == "bias":
            value = rng.normal(0, 0.015, shape).astype(np.float32)
            if name.endswith("pred_obj_score_head.layers.2.bias"):
                value.fill(2.0)  # keep the untrained decoder's mask branch observable
        else:
            fan_in = math.prod(shape[1:]) if len(shape)>1 else 1
            std = 0.55 / math.sqrt(fan_in) if kind in {"linear", "conv", "deconv"} else 0.20
            if "gaussian_matrix" in name:
                std = 1.0
            value = rng.normal(0, std, shape).astype(np.float32)
        # Exercise aliases used by the requested merged multiplex checkpoint.
        source_name = ("detector." if name.startswith("backbone.") else "tracker.model.") + name
        if not spec["quantizable"]:
            tensors[source_name] = value.astype(np.float16) if kind == "deconv" else value
            continue
        group = 256
        while group >= 4 and shape[1] % group:
            group //= 4
        config: dict = {"format": "int8_tensorwise"}
        if group >= 4:
            h = hadamard(group)
            if value.ndim == 2:
                value = (value.reshape(shape[0], -1, group) @ h.T).reshape(shape)
            else:
                flat = value.transpose(0,2,3,1).copy()
                value = (flat.reshape(-1, shape[1]//group, group) @ h.T).reshape(flat.shape).transpose(0,3,1,2).copy()
            config.update(convrot=True, convrot_groupsize=group)
            rotated += 1
        scale = np.maximum(np.abs(value).max(axis=tuple(range(1,value.ndim)), keepdims=True), 1e-6) / np.float32(127)
        quant = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
        base = source_name.removesuffix(".weight")
        tensors[source_name] = np.ascontiguousarray(quant)
        tensors[base+".weight_scale"] = scale.astype(np.float32)
        tensors[base+".comfy_quant"] = np.frombuffer(json.dumps(config, separators=(",", ":")).encode(), dtype=np.uint8).copy()
        count += 1
    output.mkdir(parents=True, exist_ok=True)
    source = output/"TEST_ONLY_native_sam31.safetensors"
    gguf = output/"TEST_ONLY_native_sam31.gguf"
    write_fixture(source, {name: ({np.dtype("float16"): "F16", np.dtype("float32"): "F32", np.dtype("int8"): "I8", np.dtype("uint8"): "U8"}[value.dtype], value) for name, value in tensors.items()}, metadata={
        "sam31.test_fixture": "true",
        "sam31.native.config": json.dumps(manifest["config"], separators=(",", ":")),
        "purpose": "UNTRAINED full-depth native graph smoke fixture, NOT the trained SAM3.1 checkpoint",
    })
    report = convert(source, gguf, allow_test_fixture=True, overwrite=True)
    report["roundtrip"] = compare_stores(SafeTensorFile(source), GGUFFile(gguf))
    report["native_required_parameters"] = len(manifest["tensors"])
    report["quantized_layers"] = count
    report["rotated_layers"] = rotated
    report["learned_model"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = generate(json.loads(args.manifest.read_text()), args.output)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Fixture generation failed: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
