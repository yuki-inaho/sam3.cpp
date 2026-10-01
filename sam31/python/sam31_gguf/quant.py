"""Decode per-layer ConvRot metadata and recover float-space weights.

Regular H4 is J4 - 2*reverse(I4), normalized by 2. Its Kronecker powers are
symmetric and orthogonal. This is NOT the usual Sylvester/Walsh basis. The
implementation below follows that mathematical definition; it does not import
or vendor the quantizer's implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .format import FormatError, TensorStore, decode_json


@dataclass(frozen=True)
class QuantSpec:
    weight_name: str
    scale_name: str
    descriptor_name: str
    group_size: int | None


def is_power_of_four(value: Any) -> bool:
    return type(value) is int and value >= 4 and value & (value - 1) == 0 and (
        value.bit_length() - 1) % 2 == 0


def inspect_quantization(store: TensorStore) -> dict[str, QuantSpec]:
    result: dict[str, QuantSpec] = {}
    descriptors: set[str] = set()
    for name, info in store.tensors.items():
        if not name.endswith(".weight") or info.dtype != "I8":
            continue
        prefix = name[:-len(".weight")]
        scale_name, descriptor_name = prefix + ".weight_scale", prefix + ".comfy_quant"
        if scale_name not in store.tensors or descriptor_name not in store.tensors:
            raise FormatError(f"INT8 weight lacks scale/quantization descriptor: {name}")
        descriptor = store.tensors[descriptor_name]
        if descriptor.dtype not in ("U8", "I8") or descriptor.size > 65_536:
            raise FormatError(f"Invalid quantization descriptor tensor: {descriptor_name}")
        config = decode_json(store.raw(descriptor_name))
        if not isinstance(config, dict) or config.get("format") != "int8_tensorwise":
            raise FormatError(f"Unsupported quantization scheme at {name}")
        allowed = {"format", "convrot", "convrot_groupsize"}
        if set(config) - allowed:
            raise FormatError(f"Unknown quantization semantics at {name}: {set(config) - allowed}")
        rotated = config.get("convrot", False)
        if type(rotated) is not bool:
            raise FormatError(f"convrot must be a boolean at {name}")
        if len(info.shape) not in (2, 4):
            raise FormatError(f"INT8 weights must be a Linear or Conv2d tensor: {name}")
        group = config.get("convrot_groupsize") if rotated else None
        if rotated and (not is_power_of_four(group) or info.shape[1] % group != 0):
            raise FormatError(f"Invalid ConvRot group size at {name}: {group}")
        scale_info = store.tensors[scale_name]
        if scale_info.dtype not in ("F16", "F32", "F64", "BF16"):
            raise FormatError(f"Scale is not floating point: {scale_name}")
        row_shape = (info.shape[0],) + (1,) * (len(info.shape) - 1)
        # Accept scalar and unambiguous per-output-channel encodings only.
        if scale_info.shape not in ((), (1,), (info.shape[0],), row_shape):
            raise FormatError(f"Scale does not match output channels at {name}: {scale_info.shape}")
        scale = float_array(store, scale_name)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise FormatError(f"Scale must be finite and positive: {scale_name}")
        result[name] = QuantSpec(name, scale_name, descriptor_name, group)
        descriptors.add(descriptor_name)
    unmatched = {name for name in store.tensors if name.endswith(".comfy_quant")} - descriptors
    if unmatched:
        raise FormatError(f"Quantization descriptor has no supported INT8 weight: {sorted(unmatched)[:3]}")
    return result


def float_array(store: TensorStore, name: str) -> np.ndarray:
    info = store.tensors[name]
    value = store.array(name)
    if info.dtype == "BF16":
        value = (value.astype(np.uint32) << 16).view(np.float32)
    return value.astype(np.float32, copy=False)


def regular_hadamard(size: int) -> np.ndarray:
    """Explicit reference matrix, intended for tests and small sizes."""
    if not is_power_of_four(size):
        raise ValueError("Hadamard group size must be a power of four, at least four")
    result = np.ones((1, 1), dtype=np.float32)
    base = (np.ones((4, 4), dtype=np.float32) - 2 * np.fliplr(np.eye(4, dtype=np.float32))) / 2
    while result.shape[0] < size:
        result = np.kron(result, base)
    return result


def rotate_last_axis(value: np.ndarray, group_size: int) -> np.ndarray:
    """Apply the regular orthonormal transform in O(N log(group_size)) work.

    Each radix-four stage computes sum(a,b,c,d) minus twice the reversed input.
    The same operation is the inverse because the basis is symmetric.
    """
    if not is_power_of_four(group_size):
        raise ValueError("Hadamard group size must be a power of four")
    if value.ndim < 1 or value.shape[-1] % group_size:
        raise ValueError("Last dimension is not divisible by the ConvRot group size")
    result = np.asarray(value, dtype=np.float32).copy()
    original_shape = result.shape
    width = 1
    while width < group_size:
        stage = result.reshape(-1, 4, width)
        totals = stage.sum(axis=1, keepdims=True, dtype=np.float32)
        result = (totals - np.float32(2) * stage[:, ::-1, :]).reshape(original_shape)
        width *= 4
    return result / np.float32(np.sqrt(group_size))


def inverse_weight_rotation(weight: np.ndarray, group_size: int) -> np.ndarray:
    if weight.ndim == 2:
        return rotate_last_axis(weight, group_size)
    if weight.ndim == 4:
        # Channel axis, NOT flattened input-channel/kernel coordinates.
        return np.ascontiguousarray(
            rotate_last_axis(weight.transpose(0, 2, 3, 1), group_size).transpose(0, 3, 1, 2)
        )
    raise ValueError("Only Linear (2D) and Conv2d (4D) weights support ConvRot")


def restored_tensor(store: TensorStore, name: str, specs: dict[str, QuantSpec]) -> np.ndarray:
    """Return an owned tensor; no original SafeTensors file is needed for GGUF."""
    if name not in specs:
        if store.tensors[name].dtype == "BF16":
            return float_array(store, name).copy()
        return store.array(name)
    spec = specs[name]
    weight = store.array(name).astype(np.float32)
    scale = float_array(store, spec.scale_name)
    scale = scale.reshape((-1,) + (1,) * (weight.ndim - 1)) if scale.size > 1 else scale.reshape(())
    weight *= scale
    if spec.group_size is not None:
        weight = inverse_weight_rotation(weight, spec.group_size)
    if not np.isfinite(weight).all():
        raise FormatError(f"Non-finite reconstructed weights: {name}")
    return weight
