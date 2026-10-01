#!/usr/bin/env python3
"""Make small storage/math fixtures. These tensors are NOT SAM model weights.

The writer is intentionally separate from the production readers/converter.
Only NumPy and the standard library are required. Real inference rejects the
explicit fixture marker carried into GGUF during conversion.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


def fixture_tensors() -> dict[str, tuple[str, np.ndarray]]:
    random = np.random.default_rng(310031)
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    for prefix, shape, group in (("test.linear16", (5, 32), 16),
                                 ("test.linear256", (3, 512), 256),
                                 ("test.plain", (2, 6), None),
                                 ("test.conv16", (3, 16, 3, 2), 16)):
        tensors[prefix + ".weight"] = ("I8", random.integers(-127, 128, shape, dtype=np.int8))
        scale_shape = () if group is None else (shape[0],) + (1,) * (len(shape) - 1)
        scale = np.asarray(random.uniform(0.002, 0.015, scale_shape), dtype=np.float32)
        tensors[prefix + ".weight_scale"] = ("F32", scale)
        descriptor: dict[str, Any] = {"format": "int8_tensorwise"}
        if group is not None:
            descriptor.update(convrot=True, convrot_groupsize=group)
        encoded = json.dumps(descriptor, separators=(",", ":")).encode()
        tensors[prefix + ".comfy_quant"] = ("U8", np.frombuffer(encoded, dtype=np.uint8).copy())
        tensors[prefix + ".bias"] = ("F16", random.normal(size=shape[0]).astype(np.float16))
    tensors["test." + "long_source_name_" * 10 + ".weight"] = ("F32", np.arange(20, dtype=np.float32).reshape(4, 5))
    tensors["test.scalar"] = ("F32", np.asarray(0.75, dtype=np.float32))
    tensors["test.bfloat16"] = ("BF16", np.array([0x3F80, 0x4000, 0xBF80], dtype=np.uint16))
    tensors["test.integer64"] = ("I64", np.array([0, 1, 2**40], dtype=np.int64))
    tensors["test.five_dimensional"] = ("F16", np.arange(32, dtype=np.float16).reshape(2, 2, 2, 2, 2))
    tensors["test.boolean"] = ("BOOL", np.array([True, False, True], dtype=np.bool_))
    return tensors


def write_fixture(path: Path, tensors: dict[str, tuple[str, np.ndarray]] | None = None,
                  metadata: dict[str, str] | None = None) -> None:
    tensors = fixture_tensors() if tensors is None else tensors
    header: dict[str, Any] = {"__metadata__": metadata if metadata is not None else {
        "sam31.test_fixture": "true", "purpose": "Storage and numerical tests only; NOT trained SAM 3.1 weights"
    }}
    payload = bytearray()
    for name, (dtype, tensor) in sorted(tensors.items()):
        data = tensor.tobytes(order="C")
        header[name] = {"dtype": dtype, "shape": list(tensor.shape),
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        file.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_fixture(args.output)
    print(json.dumps({"test_fixture": True, "trained_sam31_weights": False,
                      "output": str(args.output), "tensor_count": len(fixture_tensors())}, indent=2))


if __name__ == "__main__":
    main()
