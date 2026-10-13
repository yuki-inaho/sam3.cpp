# Opt-in video JPEG decoding

`decode_jpeg_rgb` is a small original adapter to the system libjpeg API. It is
used only for the Efficient-SAM2 video path. The legacy stb image loader stays
unchanged. Link the adapter with CMake's `JPEG::JPEG` target; installations need
the libjpeg development package (Debian: `libjpeg62-turbo-dev`). libjpeg-turbo
is the tested implementation. No codec implementation or third-party source is
copied into this repository.

## Pixel contract and limits

The supported input is a full-resolution, 8-bit grayscale, RGB or YCbCr JPEG,
including baseline and progressive coding. Output is width × height × 3
contiguous RGB bytes in encoded scanline order. Settings explicitly select
`JDCT_ISLOW`, fancy chroma upsampling, block smoothing and scale 1/1. These match
Pillow's ordinary non-draft decoder. Grayscale is replicated into RGB.

Like `Image.open(path).convert("RGB")`, this helper does not apply EXIF
orientation, ICC color management, rotations or resizing. The video caller
continues with the existing Pillow-compatible resize helper. CMYK/YCCK and
non-8-bit input are rejected with a diagnostic; this narrow adapter does not
claim to implement Pillow's separate CMYK-to-RGB conversion.

Malformed input, missing files, all libjpeg corruption warnings (including
missing EOI/truncation), allocation failure and decoded RGB larger than 512 MiB
are errors. Warning rejection is deliberately stricter than Pillow's handling
of some recoverable corrupt JPEGs. The helper never hands a partial frame to
inference. Failed calls clear RGB and dimensions and return a nonempty error.
Each call owns its independent decoder and file; no global codec state is used.
The longjmp boundary contains only trivial local variables and heap state.

## Verification on 2026-10-13

The frozen reference environment has Pillow 11.0.0, JPEG ABI 6.2 and bundled
libjpeg-turbo 3.0.4. The host reports libjpeg-turbo 2.1.5 (build 20250503), ABI
6.2. The adapter compiled against the latter passes exact byte comparisons
against Pillow for all 17 frozen 480×848 public video frames, `tests/cat.jpg`,
and synthetic baseline/progressive 4:4:4, 4:2:2 and 4:2:0, grayscale, true RGB,
1×1, EXIF orientation and embedded-profile cases. This is evidence for those
inputs, not a guarantee across all codecs or JPEG variants.

Run the focused oracle without model inference:

```
uv run python efficient_sam2/test_jpeg_decode.py /path/to/test_jpeg_decode \
  /path/to/public/frames tests/cat.jpg
```

The C++ adapter test emits a binary PPM for independent Pillow comparison and
checks the failure-output contract. The Python test also covers missing,
empty, non-JPEG, truncated, missing-EOI, CMYK and misnamed-PNG input.

## Dependency and source provenance

Official sources inspected for API behavior and default settings:

- [Pillow 11.0.0 JpegDecode.c](https://github.com/python-pillow/Pillow/blob/11.0.0/src/libImaging/JpegDecode.c)
- [Pillow 11.0.0 license](https://github.com/python-pillow/Pillow/blob/11.0.0/LICENSE)
- [libjpeg-turbo API documentation](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2.1.5/libjpeg.txt)
- [libjpeg-turbo license](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2.1.5/LICENSE.md)
- [Debian development package](https://deb.debian.org/debian/pool/main/libj/libjpeg-turbo/libjpeg62-turbo-dev_2.1.5-4_amd64.deb)

For validation, the official Debian development package was extracted into a
temporary dependency prefix outside the source repository without changing
system packages. Its SHA-256 is
`e687e3c8b24a3aa74974928b8e0b07f6df2442187492d85cbcea7dacd91e3aba`.
The original Debian copyright/license file remains under that extraction's
`usr/share/doc/libjpeg62-turbo-dev/copyright`. Runtime headers are not committed.
The build links the host `/usr/lib/x86_64-linux-gnu/libjpeg.so.62`; it does not
link private Pillow wheel internals. Distributors bundling libjpeg-turbo must
preserve its license notices; the codec has IJG, BSD-style and zlib-style
components as described by upstream. This original adapter is covered by the
repository license.

Final focused validation: 40 checks passed including `data/test_image.jpg`,
YCCK header rejection, oversized/zero dimensions and 12-bit precision rejection.
The same 40 checks passed under AddressSanitizer and UndefinedBehaviorSanitizer
with leak detection disabled. LeakSanitizer itself cannot run in this executor
because it reports a ptrace incompatibility; leak checking is not claimed.

As with other vector-returning preprocessing helpers, an exception from the
final `std::vector` allocation propagates after a scoped owner releases the
JPEG decoder, input file and raw pixels. The owner is created only after the
last fallible codec call, so libjpeg's longjmp cannot bypass its destructor.
This allocation-exception path is reviewed, not fault-injection tested.
