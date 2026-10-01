"""Losslessly store the pinned EV-M trained tensors and BPE in GGUF v3."""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "efficientsam3"))
from contract import CONTEXT, HF_REVISION, MODEL_SHA256, SOURCE_REVISION, VARIANT
from reference import build_reference

# Reuse the exact PCS key mapping already consumed by sam3.cpp.
from convert_sam3_to_ggml import rename_key


def rename(name):
    vision = "backbone.vision_backbone.trunk.model."
    language = "backbone.language_backbone."
    if name.startswith(vision):
        result = "ev." + name[len(vision) :]
        result = result.replace("backbone.model.input_stem.op_list.", "stem.")
        result = result.replace("backbone.model.stages.", "s.").replace(
            ".op_list.", "."
        )
        result = result.replace(".context_module.main.", ".att.").replace(
            ".local_module.main.", ".mb."
        )
        result = result.replace(".main.", ".")
    elif name.startswith(language):
        result = "et." + name[len(language) :]
        result = result.replace("encoder.transformer.", "b.")
        for old, new in (
            (".pre_norm_mha.0.", ".ln1."),
            (".pre_norm_mha.1.qkv_proj.", ".qkv."),
            (".pre_norm_mha.1.out_proj.", ".out."),
            (".pre_norm_ffn.0.", ".ln2."),
            (".pre_norm_ffn.1.", ".fc1."),
            (".pre_norm_ffn.4.", ".fc2."),
            (".token_mixer.", ".tm."),
            (".convffn.", ".ff."),
        ):
            result = result.replace(old, new)
    else:
        result = rename_key("detector." + name)
        if result is None:
            raise ValueError(f"Unexpected discarded inference tensor: {name}")
    result = result.replace("running_mean", "rm").replace("running_var", "rv")
    result = result.replace("num_batches_tracked", "nb")
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"Tensor name exceeds ggml limit: {result}")
    return result


def convert_arrays(state):
    arrays = {}
    for source_name, value in state.items():
        name = rename(source_name)
        if name in arrays:
            raise ValueError(f"Duplicate tensor name: {name}")
        data = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        if np.issubdtype(data.dtype, np.floating) and not np.isfinite(data).all():
            raise ValueError(f"Non-finite tensor: {name}")
        if name == "geom.boxes_pool_project.weight":
            data = data.reshape(256, 256, 7, 7)
        if data.ndim == 0:
            data = data.reshape(1)
        arrays[name] = np.ascontiguousarray(data)
    return arrays


def convert(source, checkpoint, output):
    torch.set_num_threads(16)
    model = build_reference(source, checkpoint)
    arrays = convert_arrays(model.state_dict())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.gguf")
    writer = gguf.GGUFWriter(temporary, "efficientsam3")
    writer.add_string("efficientsam3.variant", VARIANT)
    writer.add_string("efficientsam3.source_revision", SOURCE_REVISION)
    writer.add_string("efficientsam3.hf_revision", HF_REVISION)
    writer.add_string("efficientsam3.checkpoint_sha256", MODEL_SHA256)
    writer.add_string("efficientsam3.sequence_mode", "independent_detection")
    writer.add_uint32("efficientsam3.context", CONTEXT)
    tokenizer = model.backbone.language_backbone.tokenizer
    tokens = [
        token
        for token, _ in sorted(tokenizer.encoder.items(), key=lambda pair: pair[1])
    ]
    merges = [
        " ".join(pair)
        for pair, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda pair: pair[1])
    ]
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)
    for name, data in arrays.items():
        writer.add_tensor(name, data)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reader = gguf.GGUFReader(temporary)
        for tensor in reader.tensors:
            original = arrays[tensor.name]
            if tensor.data.tobytes() != original.tobytes():
                raise ValueError(f"Payload mismatch: {tensor.name}")
        if len(reader.tensors) != len(arrays):
            raise ValueError("Tensor count mismatch")
        del reader
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "passed": True,
        "tensor_count": len(arrays),
        "payload_bytes": sum(v.nbytes for v in arrays.values()),
        "checkpoint_sha256": MODEL_SHA256,
        "source_revision": SOURCE_REVISION,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/efficientsam3/source")
    parser.add_argument("--checkpoint", default="models/efficientsam3_ev_m.pt")
    parser.add_argument("--output", default="models/efficientsam3_ev_m.gguf")
    a = parser.parse_args()
    print(json.dumps(convert(a.source, a.checkpoint, a.output), indent=2))


if __name__ == "__main__":
    main()
