"""Compare the torch evaluator against actual ggml fixtures and edge policies."""
import pathlib
import sys
import numpy as np
import torch
from emulator import fake_q8_0, linear, verify_fixtures

torch.set_num_threads(1)
verify_fixtures(sys.argv[1])
count = 0
for path in sorted(pathlib.Path(sys.argv[1]).glob("*_w.f32")):
    k,n,m = map(int,path.stem.removesuffix("_w").split("_"))
    prefix=str(path).removesuffix("_w.f32")
    w=torch.from_numpy(np.fromfile(path,"<f4").reshape(n,k))
    x=torch.from_numpy(np.fromfile(prefix+"_x.f32","<f4").reshape(m,k))
    y=torch.from_numpy(np.fromfile(prefix+"_y.f32","<f4").reshape(m,n))
    actual=linear(x,w)
    error=(torch.linalg.vector_norm(actual-y)/torch.linalg.vector_norm(y)).item()
    assert error<=1e-6, (path,error)
    # A noncontiguous input must preserve logical last-axis data, not raw storage order.
    transposed=x.T.contiguous().T
    assert not transposed.is_contiguous()
    torch.testing.assert_close(linear(transposed,w),actual,rtol=1e-5,atol=1e-6)
    print(f"torch {k},{n},{m}: relativeL2={error:.8g}")
    count += 1
assert count == 6, "Missing oracle fixtures"
for shape in [(0,32),(0,14,14,448)]:
    x=torch.empty(shape)
    out=linear(x,torch.ones(3,shape[-1]))
    assert out.shape == (*shape[:-1],3) and out.numel() == 0
for width in (16,32,33,112,224):
    x=torch.ones(3,width)
    w=torch.zeros(2,width)
    assert torch.equal(linear(x,w),torch.zeros(3,2))
    w.fill_(1e-30)
    assert torch.equal(linear(x,w),torch.nn.functional.linear(x,w))
    w.fill_(1e20)
    assert torch.equal(linear(x,w),torch.nn.functional.linear(x,w))
for value in (float("inf"),float("nan")):
    try:
        linear(torch.ones(1,32),torch.full((1,32),value))
        raise AssertionError("Non-finite accepted")
    except ValueError:
        pass
# Deliberate half ties: weights use round-away, AVX2 activations nearest-even.
x=torch.zeros(1,32); x[0,:5]=torch.tensor([127,.5,-.5,1.5,-1.5])
assert torch.equal(fake_q8_0(x,weights=True)[0,:5],torch.tensor([127.,1.,-1.,2.,-2.]))
assert torch.equal(fake_q8_0(x,weights=False)[0,:5],torch.tensor([127.,0.,0.,2.,-2.]))
print("PASS: torch fixture, noncontiguous, zero, tiny, huge, nonfinite, widths and tie tests")

# Root module names and FP16 bias must obey the reusable API contract.
from types import SimpleNamespace
from manifest import build_manifest
root = torch.nn.Linear(32, 2)
record = SimpleNamespace(in_features=32, out_features=2, group_size=16,
                         dequantized_weight=lambda: np.ones((2,32), np.float32))
manifest = build_manifest(root, SimpleNamespace(quantized={"":record}))
assert manifest["layers"][0]["source"] == "weight"
x = torch.ones(3,32); w = torch.ones(2,32); bias = torch.tensor([.5,-.5],dtype=torch.float16)
torch.testing.assert_close(linear(x,w,bias), linear(x,w,bias.float()),rtol=0,atol=0)
print("PASS: root Linear manifest and FP16 bias regressions")
