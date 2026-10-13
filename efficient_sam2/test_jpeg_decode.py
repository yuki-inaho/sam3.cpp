"""Byte-exact independent Pillow oracle; no model, torch or inference required.

Usage: uv run python efficient_sam2/test_jpeg_decode.py BINARY [JPEG_OR_DIR ...]
Optional inputs cover real/public JPEGs without copying their licensed assets.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, features

binary = Path(sys.argv[1]).resolve()
checks = 0
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    output = root / "decoded.ppm"

    def exact(path):
        global checks
        subprocess.run([str(binary), str(path), str(output)], check=True)
        with Image.open(path) as source, Image.open(output) as actual:
            expected = source.convert("RGB")
            assert actual.size == expected.size, path
            assert actual.tobytes() == expected.tobytes(), path
        checks += 1

    def reject(path):
        global checks
        result = subprocess.run(
            [str(binary), str(path), str(output)], capture_output=True
        )
        assert result.returncode == 3, (path, result.returncode, result.stderr)
        assert result.stderr, path
        checks += 1

    rgb = Image.frombytes("RGB", (37, 29), bytes((i * 73 + i // 11) % 256 for i in range(37 * 29 * 3)))
    for progressive in (False, True):
        for subsampling in (0, 1, 2):
            path = root / f"rgb-{progressive}-{subsampling}.jpg"
            rgb.save(path, quality=87, progressive=progressive, subsampling=subsampling)
            exact(path)
    for mode in ("L", "RGB"):
        path = root / f"{mode}.jpg"
        rgb.convert(mode).save(path, keep_rgb=mode == "RGB")
        exact(path)
    path = root / "one-pixel.jpg"
    Image.new("RGB", (1, 1), (14, 51, 213)).save(path)
    exact(path)
    # Image.open().convert() does not transpose EXIF or apply an ICC transform.
    exif = Image.Exif()
    exif[274] = 6
    path = root / "oriented-profile.jpg"
    rgb.save(path, exif=exif, icc_profile=b"ignored test profile")
    exact(path)
    with Image.open(path) as oriented:
        assert oriented.size == rgb.size and oriented.getexif()[274] == 6
    for name, data in (
        ("empty.jpg", b""),
        ("garbage.jpg", b"not a JPEG"),
        ("truncated.jpg", path.read_bytes()[:100]),
        ("missing-eoi.jpg", path.read_bytes()[:-2]),
    ):
        bad = root / name
        bad.write_bytes(data)
        reject(bad)
    reject(root / "missing.jpg")
    cmyk = root / "cmyk.jpg"
    rgb.convert("CMYK").save(cmyk)
    reject(cmyk)
    # Reject the alternate four-component Adobe YCCK interpretation, too.
    ycck_data = bytearray(cmyk.read_bytes())
    adobe = ycck_data.index(b"Adobe")
    ycck_data[adobe + 11] = 2
    ycck = root / "ycck.jpg"
    ycck.write_bytes(ycck_data)
    reject(ycck)
    # Dimension and precision header checks happen before pixel allocation.
    header = bytearray((root / "rgb-False-0.jpg").read_bytes())
    sof = header.index(b"\xff\xc0")
    for name, offset, replacement in (
        ("oversize", 5, b"\xff\xff\xff\xff"),
        ("zero-height", 5, b"\x00\x00"),
        ("precision-12", 4, b"\x0c"),
    ):
        data = header.copy()
        data[sof + offset : sof + offset + len(replacement)] = replacement
        bad = root / f"{name}.jpg"
        bad.write_bytes(data)
        reject(bad)
    png = root / "misnamed.jpg"
    rgb.save(png, format="PNG")
    reject(png)
    for argument in sys.argv[2:]:
        path = Path(argument)
        paths = sorted(path.glob("*.jpg")) if path.is_dir() else [path]
        assert paths, path
        for jpeg in paths:
            exact(jpeg)
print(f"JPEG RGB: {checks} exact-Pillow/error checks PASS; Pillow JPEG={features.version_codec('jpg')}, turbo={features.version_feature('libjpeg_turbo')}")
