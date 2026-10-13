"""Exact candidate selection shared by evaluation and native integration review.

Call before replacing nn.Linear modules. Never select by .weight suffix alone.
"""
def build_manifest(model, artifact):
    from torch import nn
    import numpy as np

    layers = []
    for name, record in sorted(artifact.quantized.items()):
        target = model.get_submodule(name) if name else model
        if not isinstance(target, nn.Linear):
            raise ValueError(f"Quantization target is not a Linear: {name}")
        if (target.in_features, target.out_features) != (record.in_features, record.out_features):
            raise ValueError(f"Linear dimensions differ from metadata: {name}")
        k = record.in_features
        reason = "eligible-linear" if k % 32 == 0 else "input-width-not-block-aligned"
        if k % 32 == 0:
            restored = record.dequantized_weight()
            if not np.isfinite(restored).all():
                raise ValueError(f"Non-finite restored weights: {name}")
            maximum = np.abs(restored.reshape(-1,32)).max(axis=-1)
            if np.any(maximum > np.float32(65504)*np.float32(127)):
                reason = "fp16-scale-overflow"
            elif np.any((maximum > 0) & ((maximum/np.float32(127)).astype(np.float16) == 0)):
                reason = "fp16-scale-underflow"
        layers.append({"source": name + ".weight" if name else "weight", "module": name,
                       "in_features": k, "out_features": record.out_features,
                       "convrot_group": record.group_size,
                       "candidate": "q8_0-second-quantization" if reason == "eligible-linear" else "f32",
                       "reason": reason})
    return {"schema": 1, "enabled_by_default": False,
            "candidate": "inverse-convrot-f32-then-ggml-q8_0-avx2",
            "selection": "GGUF quantization.layers intersect verified nn.Linear consumers",
            "layers": layers}
