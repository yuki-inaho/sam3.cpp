#!/usr/bin/env python3
"""Offline numerical test-data generation, not an inference backend.

The native test executable consumes the saved JSON without importing Python or
PyTorch. These operator tests do not assert official end-to-end model parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
from torch.nn import functional as F


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw = load_file(str(args.source))
    rng = np.random.default_rng(912)
    cases: list[dict] = []

    def weight(name: str) -> torch.Tensor:
        key = ("detector." if name.startswith("backbone.") else "tracker.model.") + name
        array = raw[key].astype(np.float32)
        if raw[key].dtype == np.int8:
            prefix = key.removesuffix(".weight")
            array = array * raw[prefix + ".weight_scale"]
            config = json.loads(raw[prefix + ".comfy_quant"].tobytes())
            if config.get("convrot", False):
                size = config["convrot_groupsize"]
                h4 = np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], np.float32)/2
                matrix = np.ones((1,1), dtype=np.float32)
                while matrix.shape[0] < size: matrix = np.kron(matrix, h4)
                if array.ndim == 2:
                    array = (array.reshape(-1,size) @ matrix).reshape(array.shape)
                else:
                    moved = array.transpose(0,2,3,1).copy()
                    array = (moved.reshape(-1,size) @ matrix).reshape(moved.shape).transpose(0,3,1,2).copy()
        return torch.from_numpy(array.copy())

    def sample(shape: tuple[int,int,int]) -> torch.Tensor:
        return torch.from_numpy(rng.normal(0,.7,shape).astype(np.float32))

    def save_tensor(value: torch.Tensor) -> dict:
        value = value.detach().float().contiguous()
        return {"shape": list(value.shape), "values": value.flatten().tolist()}

    def nchw(x: torch.Tensor) -> torch.Tensor: return x.permute(2,0,1).unsqueeze(0)
    def hwc(x: torch.Tensor) -> torch.Tensor: return x.squeeze(0).permute(1,2,0).contiguous()
    def add(name: str, operation: str, inputs: list[torch.Tensor], output: torch.Tensor, **options: object) -> None:
        cases.append({"name":name,"operation":operation,"inputs":[save_tensor(x) for x in inputs],
                      "output":save_tensor(output),**options})

    trunk = "backbone.vision_backbone.trunk"
    prefix = trunk+".blocks.0.attn.qkv"
    x=sample((3,2,32));add("convrot_linear", "linear", [x], F.linear(x,weight(prefix+".weight"),weight(prefix+".bias")), prefix=prefix)
    prefix = trunk+".blocks.0.norm1"
    x=sample((3,7,32));add("layer_norm", "norm", [x], F.layer_norm(x,(32,),weight(prefix+".weight"),weight(prefix+".bias"),1e-6), prefix=prefix)
    for label,prefix,shape,stride,padding,groups,bias in [
        ("patch_convolution",trunk+".patch_embed.proj",(12,16,3),4,0,1,False),
        ("convrot_convolution","backbone.vision_backbone.interactive_convs.2.conv_3x3",(7,9,32),1,1,1,True),
        ("depthwise_convolution","maskmem_backbone.fuser.layers.0.dwconv",(7,9,32),1,3,32,True),
    ]:
        x=sample(shape); y=F.conv2d(nchw(x),weight(prefix+".weight"),weight(prefix+".bias") if bias else None,stride,padding,groups=groups)
        add(label,"conv",[x],hwc(y),prefix=prefix,stride=stride,padding=padding,depthwise=groups!=1,bias=bias)
    prefix="backbone.vision_backbone.interactive_convs.1.dconv_2x2"
    x=sample((3,5,32));y=F.conv_transpose2d(nchw(x),weight(prefix+".weight"),weight(prefix+".bias"),stride=2)
    add("transpose_convolution_fp16_weights","deconv",[x],hwc(y),prefix=prefix)
    for shape,out,aa in [((11,13,3),(4,5),True),((11,13,3),(4,5),False),((3,5,4),(9,12),False)]:
        x=sample(shape);y=F.interpolate(nchw(x),size=out,mode="bilinear",align_corners=False,antialias=aa)
        add(f"resize_{shape}_{out}_{aa}","resize",[x],hwc(y),height=out[0],width=out[1],antialias=aa)
    for n,m,scale in [(35,19,1.0),(17,37,25.0)]:
        q=sample((n,1,32))*scale;k=sample((m,1,32))*scale;v=sample((m,1,16));heads=4
        pack=lambda x:x.reshape(1,x.shape[0],heads,x.shape[-1]//heads).transpose(1,2)
        y=F.scaled_dot_product_attention(pack(q),pack(k),pack(v)).transpose(1,2).reshape(n,1,16)
        add(f"attention_{n}_{m}_{scale}","attention",[q,k,v],y,heads=heads)
    x=sample((48,1,32));y=x.clone();heads=4;dim=8;grid=4
    pos=np.arange(16,dtype=np.float32)
    frequencies=1.0/np.power(10000.0,np.arange(0,dim,4,dtype=np.float32)/dim)
    phase=np.concatenate([np.outer(pos%grid,frequencies),np.outer(pos//grid,frequencies)],axis=-1)
    rotations=torch.polar(torch.ones((16,dim//2)),torch.from_numpy(phase))
    z=torch.view_as_complex(x[:16].reshape(16,heads,dim//2,2).contiguous())
    y[:16]=torch.view_as_real(z*rotations[:,None,:]).reshape(16,1,32)
    add("rotary_excludes_32_pointer_tokens","rope",[x],y,heads=heads,grid=grid,exclude=32)
    x=sample((7,9,32));add("gelu_exact","gelu",[x],F.gelu(x,approximate="none"))
    report={"purpose":"independent operator numerical checks; not full official model equivalence",
            "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),"torch_version":torch.__version__,
            "atol":2e-5,"rtol":2e-4,"cases":cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,separators=(",",":"),allow_nan=False)+"\n")
    print(json.dumps({"cases":len(cases),"output":str(args.output),"torch_version":torch.__version__}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
