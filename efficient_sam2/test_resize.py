"""Independent Pillow oracle for uint8 RGB video preprocessing."""
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

binary=Path(sys.argv[1]).resolve()
rng=np.random.default_rng(20261013)
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    checks=0
    for w,h,ow,oh in [(7,5,19,23),(19,23,7,5),(1,1,17,19),(7,5,7,5),
                       (480,848,1024,1024),(1280,720,1024,1024)]:
        rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8)
        (root/'in.rgb').write_bytes(rgb.tobytes())
        subprocess.run([str(binary),str(root/'in.rgb'),str(root/'out.rgb'),str(w),str(h),str(ow),str(oh)],check=True)
        actual=np.fromfile(root/'out.rgb',np.uint8).reshape(oh,ow,3)
        expected=np.asarray(Image.fromarray(rgb).resize((ow,oh)))
        assert np.array_equal(actual,expected),(w,h,ow,oh,np.abs(actual.astype(int)-expected).max())
        checks+=1
    for kind in ['edge','gradient']:
        w,h,ow,oh=31,17,11,47
        rgb=np.zeros((h,w,3),np.uint8)
        if kind=='edge': rgb[:,w//2:]=255
        else: rgb[:]=np.linspace(0,255,w,dtype=np.uint8)[None,:,None]
        (root/'in.rgb').write_bytes(rgb.tobytes())
        subprocess.run([str(binary),str(root/'in.rgb'),str(root/'out.rgb'),str(w),str(h),str(ow),str(oh)],check=True)
        actual=np.fromfile(root/'out.rgb',np.uint8).reshape(oh,ow,3)
        assert np.array_equal(actual,np.asarray(Image.fromarray(rgb).resize((ow,oh)))),kind
        checks+=1
print(f'Pillow bicubic RGB resize: {checks} exact geometry/edge/gradient cases PASS')
