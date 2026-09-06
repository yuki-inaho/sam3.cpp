#!/bin/bash
# package_release.sh — build and package a portable Linux x86_64 release bundle.
#
# Produces dist/sam3-linux-x86_64-<version>.tar.gz containing:
#   bin/    sam3_seg, sam3_serve, sam3_benchmark, sam3_profile_edgetam, sam3_quantize
#   models/ vendored EdgeTAM weights (f16, q8_0, q4_0)
#   data/   sample image + video
#   README.md
#
# Portability strategy:
#   - GGML_NATIVE=OFF with AVX2+FMA fixed (no AVX-512 so the binary runs on
#     any x86_64 CPU since Haswell; the target container has AVX-512 but that
#     is a superset)
#   - GGML_OPENMP=OFF: removes the libgomp.so.1 dependency entirely; ggml's
#     own thread pool provides multithreading with comparable performance
#   - built with the conda-forge toolchain (pixi env) whose sysroot targets
#     old glibc (>= 2.17); libstdc++/libgcc are statically linked and no
#     RPATH is emitted, so the bundle does not depend on the build machine
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build-release"
DIST_DIR="$ROOT/dist"
BUNDLE_NAME="sam3-linux-x86_64-bundle"
BUNDLE="$DIST_DIR/$BUNDLE_NAME"

VERSION="$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo dev)"
ARCHIVE="$DIST_DIR/sam3-linux-x86_64-${VERSION}.tar.gz"

echo "=== Configure (Release, AVX2 fixed, no OpenMP, static libstdc++) ==="
cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSAM3_METAL=OFF \
    -DGGML_METAL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_OPENMP=OFF \
    -DCMAKE_SKIP_RPATH=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"

echo "=== Build ==="
cmake --build "$BUILD_DIR" --parallel

echo "=== Assemble bundle ==="
rm -rf "$BUNDLE" "$ARCHIVE"
mkdir -p "$BUNDLE/bin"

for bin in sam3_seg sam3_serve sam3_benchmark sam3_profile_edgetam sam3_quantize; do
    cp "$BUILD_DIR/examples/$bin" "$BUNDLE/bin/"
done

# The conda-forge toolchain injects an RPATH to the build env's lib dir
# (e.g. libgomp); strip it so the binaries only use system libraries.
command -v patchelf >/dev/null 2>&1 || {
    echo "ERROR: patchelf not found (install it or run inside the pixi env)" >&2
    exit 1
}
for f in "$BUNDLE"/bin/*; do
    patchelf --remove-rpath "$f"
done

cp -r "$ROOT/models" "$BUNDLE/models"
mkdir -p "$BUNDLE/data"
cp "$ROOT/data/test_image.jpg" "$ROOT/data/test_video.mp4" "$BUNDLE/data/"
cp "$ROOT/README-release.md" "$BUNDLE/README.md"

echo "=== Archive ==="
tar czf "$ARCHIVE" -C "$DIST_DIR" "$BUNDLE_NAME"

echo ""
ls -lh "$ARCHIVE"
echo "Bundle: $BUNDLE"
echo "Archive: $ARCHIVE"
