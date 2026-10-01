"""Complete-key GGUF loading into the pinned official CPU reference model.

This module is a reference-backend integration, NOT a native ggml model graph.
It never silently downloads a model or runs with random/missing parameters.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .format import FormatError, GGUFFile, SafeTensorFile, TensorStore
from .quant import inspect_quantization, restored_tensor
from .source import OFFICIAL_COMMIT, verify_prepared_source


def match_model_keys(expected: Mapping[str, Any], store: TensorStore) -> tuple[dict[str, str], set[str]]:
    derived = {name for name in expected if name.endswith((".freqs_cis_real", ".freqs_cis_imag"))}
    mapping: dict[str, str] = {}
    missing = []
    for name, value in expected.items():
        if name in derived and name not in store.tensors:
            continue
        candidates = ["tracker." + name, "tracker.model." + name, name]
        if name.startswith("backbone."):
            candidates.insert(0, "detector." + name)
        candidates = [wrapper + candidate for wrapper in ("", "model.", "diffusion_model.", "model.diffusion_model.")
                      for candidate in candidates]
        matches = [candidate for candidate in candidates if candidate in store.tensors]
        matches = list(dict.fromkeys(matches))
        if not matches:
            missing.append(name)
            continue
        # Multiple aliases are allowed only when their tensors are identical.
        selected = matches[0]
        for alias in matches[1:]:
            a, b = store.tensors[selected], store.tensors[alias]
            if (a.dtype, a.shape) != (b.dtype, b.shape) or store.raw(selected) != store.raw(alias):
                raise FormatError(f"Ambiguous checkpoint aliases for {name}: {matches}")
            if a.dtype == "I8" and selected.endswith(".weight"):
                for suffix in (".weight_scale", ".comfy_quant"):
                    left = selected[:-len(".weight")] + suffix
                    right = alias[:-len(".weight")] + suffix
                    if left not in store.tensors or right not in store.tensors:
                        raise FormatError(f"Quantized alias lacks {suffix}: {name}")
                    li, ri = store.tensors[left], store.tensors[right]
                    if (li.dtype, li.shape) != (ri.dtype, ri.shape) or store.raw(left) != store.raw(right):
                        raise FormatError(f"Ambiguous quantization for checkpoint aliases: {name}")
        if tuple(value.shape) != store.tensors[selected].shape:
            raise FormatError(f"Shape mismatch for {name}: expected {tuple(value.shape)}, got {store.tensors[selected].shape}")
        mapping[name] = selected
    if missing:
        raise FormatError(f"Checkpoint lacks {len(missing)} required SAM 3.1 tensors: {missing[:12]}")
    if len(set(mapping.values())) != len(mapping):
        raise FormatError("A source tensor was mapped to multiple model keys")
    return mapping, set(expected) - mapping.keys()


def build_reference_model(path: Path, source_root: Path, *, threads: int = 4,
                          safetensors_oracle: bool = False) -> tuple[Any, dict[str, Any]]:
    # Validate storage and fixture status BEFORE importing the heavy backend.
    store: TensorStore = SafeTensorFile(path) if safetensors_oracle else GGUFFile(path)
    fixture = (store.metadata.get("sam31.test_fixture") == "true") if safetensors_oracle else store.is_fixture
    if fixture:
        raise FormatError("Synthetic test weights cannot be used for SAM 3.1 image/video inference")
    if not 1 <= threads <= 256:
        raise ValueError("Thread count must be between 1 and 256")
    source_root = source_root.resolve(strict=True)
    verify_prepared_source(source_root)
    specs = inspect_quantization(store)
    if not specs:
        raise FormatError("No INT8 layers in the requested checkpoint")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch reference backend is missing; see sam31/requirements-reference.txt") from exc
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("The pinned reference stack requires Python 3.10-3.12; storage tools also support 3.13")
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    # Do not mix an already-imported unrelated sam3 installation with this tree.
    for name in list(sys.modules):
        if name == "sam3" or name.startswith("sam3."):
            del sys.modules[name]
    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    from sam3.model_builder import build_sam3_multiplex_video_model

    model = build_sam3_multiplex_video_model(
        checkpoint_path=None, load_from_HF=False, multiplex_count=16,
        use_fa3=False, use_rope_real=True, device="cpu", compile=False,
    )
    model.eval().requires_grad_(False)
    expected = model.state_dict()
    mapping, derived = match_model_keys(expected, store)
    if len(mapping) < 900:
        raise FormatError(f"Unexpected incomplete SAM 3.1 model: only {len(mapping)} tensors mapped")
    loaded_elements = 0
    with torch.inference_mode():
        for name, origin in mapping.items():
            data = restored_tensor(store, origin, specs)
            if data.dtype.kind == "f" and not np.isfinite(data).all():
                raise FormatError(f"Non-finite checkpoint tensor: {origin}")
            destination = expected[name]
            value = torch.from_numpy(np.ascontiguousarray(data)).reshape(destination.shape)
            destination.copy_(value.to(dtype=destination.dtype, device=destination.device))
            loaded_elements += destination.numel()
            del data, value
    del expected
    gc.collect()
    report = {
        "backend": "official-pytorch-cpu-reference", "native_ggml_graph": False,
        "official_commit": OFFICIAL_COMMIT, "loaded_tensors": len(mapping),
        "loaded_elements": loaded_elements, "derived_buffers": sorted(derived),
        "int8_layers_in_checkpoint": len(specs), "source_format": "safetensors" if safetensors_oracle else "gguf",
        "full_model_key_coverage": True, "torch_version": torch.__version__,
        "precision": "float32 reference inference from restored INT8 weights",
        "memory_window": "official defaults (not the bounded ONNX window)",
    }
    return model, report
