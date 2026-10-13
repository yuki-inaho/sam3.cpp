"""Optional native-ordered FP32 inverse ConvRot; no Q8 quantizer implementation.

The H4 matrix has rows [1,1,1,-1], [1,1,-1,1], [1,-1,1,1],
[-1,1,1,1]. Its tensor powers are evaluated in the same operation order as
sam31/src/quant.cpp. Dense and butterfly forms are mathematically equivalent
but may round differently before the subsequent lossy Q8 quantization.
"""

from copy import copy

import numpy as np

POLICY = {
    "restore_policy": "native-h4-f32-v1",
    "enabled_by_default": False,
    "precision": "IEEE-754 FP32 at each indicated operation",
    "scope": "Every metadata-marked ConvRot Linear, including K112 FP32 fallbacks",
    "order": "I8-to-F32 times row-scale; stride1,4,... H4 butterfly; final multiply by1/sqrt(group)",
    "butterfly": "sum=((a+b)+c)+d; outputs=sum-2*[d,c,b,a]",
    "weight_quantizer": "ggml-Q8_0 block32; unrounded FP32 maxabs/127; round-away-from-zero integers; FP16 stored scale",
    "activation_quantizer": "ggml-Q8_0 x86 AVX2; FP32 127/maxabs division; nearest-even integers; FP16 stored scale",
}


def restore_native_h4(weight_int8, weight_scale, group_size):
    """Return original-coordinate FP32 weights using native butterfly ordering."""
    weight = np.asarray(weight_int8)
    scale = np.asarray(weight_scale)
    if weight.dtype != np.int8 or weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("Expected nonempty I8 [out,in] weights")
    if scale.dtype != np.float32 or scale.shape != (weight.shape[0], 1):
        raise ValueError("Expected FP32 [out,1] scales")
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Scales must be finite and positive")
    group = group_size
    if type(group) is not int or group < 1 or weight.shape[1] % group:
        raise ValueError("Invalid ConvRot group")
    power = group
    while power > 1 and power % 4 == 0:
        power //= 4
    if power != 1:
        raise ValueError("Group must be a power of four or identity one")
    output = weight.astype(np.float32) * scale
    if group > 1:
        grouped = output.reshape(-1, group)
        stride = 1
        while stride < group:
            block = grouped.reshape(-1, group // (4 * stride), 4, stride)
            a, b, c, d = [block[:, :, index, :].copy() for index in range(4)]
            total = ((a + b) + c) + d
            for index, value in enumerate([d, c, b, a]):
                block[:, :, index, :] = total - np.float32(2) * value
            stride *= 4
        output *= np.float32(1) / np.sqrt(np.float32(group))
    if not np.isfinite(output).all():
        raise ValueError("Restoration produced nonfinite weights")
    return output


class _NativeRecord:
    def __init__(self, record):
        self._record = record

    def __getattr__(self, name):
        return getattr(self._record, name)

    def dequantized_weight(self):
        return restore_native_h4(self.weight_int8, self.weight_scale, self.group_size)


def native_h4_artifact(artifact):
    """Copy the artifact view; reuse its validated residual/state-dict handling.

    Both state loading and manifest fallback checks see native-ordered weights.
    Original records/tensors are untouched; this does not edit or export GGUF.
    """
    result = copy(artifact)
    result.quantized = {
        name: _NativeRecord(record) for name, record in artifact.quantized.items()
    }
    return result
