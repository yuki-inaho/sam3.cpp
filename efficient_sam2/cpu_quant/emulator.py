"""Experimental ggml Q8_0 AVX2 numerical reference, not a native speed path.

Input weights MUST already be restored/inverse-ConvRot FP32. Quantizing these
weights is a second lossy quantization. The activation rounding below matches
vendored x86 AVX/AVX2; other ggml architectures may round differently.
"""

def fake_q8_0(value, *, weights):
    """Return dequantized torch tensor, preserving last-axis block32 layout.

    Float matmul of these values emulates integer block dots up to accumulation
    roundoff, not bitwise accumulation order. Test against the actual kernel.
    """
    import torch

    if value.ndim < 1 or value.shape[-1] == 0:
        raise ValueError("Nonempty feature axis required")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite Q8_0 input")
    if value.shape[-1] % 32:
        raise ValueError("Q8_0 requires input width divisible by 32")
    blocks = value.float().reshape(*value.shape[:-1], value.shape[-1] // 32, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    scale = maximum / 127.0
    if weights:
        inverse = torch.where(scale != 0, torch.div(torch.ones_like(scale), scale), 0.0)
    else:
        inverse = torch.where(maximum != 0, torch.div(torch.full_like(maximum, 127.0), maximum), 0.0)
    if not torch.isfinite(inverse).all() or not torch.isfinite(scale.half()).all():
        raise ValueError("Q8_0 activation scale is not representable")
    scaled = blocks * inverse
    if weights:
        whole = scaled.abs().floor()
        quantized = scaled.sign() * (whole + ((scaled.abs() - whole) >= 0.5))
    else:
        quantized = scaled.round()
    return (quantized * scale.half().float()).reshape(value.shape)


def linear(value, restored_weight, bias=None):
    """Emulate opt-in policy; K112 retains FP32, without padding or tail loss."""
    import torch
    import torch.nn.functional as F

    if restored_weight.ndim != 2 or min(restored_weight.shape) <= 0:
        raise ValueError("Nonempty matrix weights required")
    if not torch.isfinite(restored_weight).all() or not torch.isfinite(value).all():
        raise ValueError("Non-finite linear input")
    if bias is not None:
        if not torch.isfinite(bias).all():
            raise ValueError("Non-finite linear bias")
        bias = bias.float()
    blocks = restored_weight.float().reshape(-1, 32) if restored_weight.shape[-1] % 32 == 0 else None
    unsafe = False
    if blocks is not None:
        maximum = blocks.abs().amax(dim=-1)
        unsafe = bool(((maximum > 0) & ((maximum / 127.0).half() == 0)).any()) or bool((maximum > 65504.0 * 127.0).any())
    if restored_weight.shape[-1] % 32 or unsafe:
        return F.linear(value.float(), restored_weight.float(), bias)
    return F.linear(fake_q8_0(value, weights=False),
                    fake_q8_0(restored_weight, weights=True), bias)


def verify_fixtures(directory):
    import pathlib
    import numpy as np

    packed_dtype = np.dtype([("scale", "<f2"), ("q", "i1", (32,))])
    for path in sorted(pathlib.Path(directory).glob("*_w.f32")):
        k, n, m = map(int, path.stem.removesuffix("_w").split("_"))
        prefix = str(path).removesuffix("_w.f32")
        w = np.fromfile(path, "<f4").reshape(n,k)
        x = np.fromfile(prefix+"_x.f32", "<f4").reshape(m,k)
        y = np.fromfile(prefix+"_y.f32", "<f4").reshape(m,n)
        decoded = []
        for data, suffix, weights in [(w,"w",True),(x,"x",False)]:
            if k%32:
                decoded.append(data)
                continue
            blocks = data.reshape(-1,32)
            maximum = np.abs(blocks).max(axis=-1,keepdims=True)
            scale = maximum/np.float32(127)
            with np.errstate(divide="ignore", invalid="ignore"):
                inverse = np.where(scale!=0, np.float32(1)/scale, 0) if weights else np.where(maximum!=0,np.float32(127)/maximum,0)
            scaled = blocks*inverse
            whole = np.floor(np.abs(scaled))
            q = np.copysign(whole + ((np.abs(scaled)-whole)>=.5),scaled) if weights else np.rint(scaled)
            native = np.fromfile(prefix+f"_{suffix}.packed", packed_dtype)
            if not np.array_equal(q.astype(np.int8),native["q"]):
                raise AssertionError(f"Integer quantization mismatch: {path.name}/{suffix}")
            if not np.array_equal(scale.astype(np.float16).ravel(),native["scale"]):
                raise AssertionError(f"Scale mismatch: {path.name}/{suffix}")
            decoded.append((q*scale.astype(np.float16).astype(np.float32)).reshape(data.shape))
        expected=decoded[1]@decoded[0].T
        error=np.linalg.norm(expected-y)/max(np.linalg.norm(y),1e-30)
        if not np.isfinite(y).all() or error>1e-6:
            raise AssertionError(f"Kernel/emulator relativeL2 {error}: {path.name}")
        print(f"{k},{n},{m}: exact packed bytes; emulator relativeL2={error:.8g}")

if __name__ == "__main__":
    import sys
    verify_fixtures(sys.argv[1])
