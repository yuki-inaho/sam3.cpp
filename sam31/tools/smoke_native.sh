#!/bin/sh
# All test execution and inference below use native C++ executables.
# Python is neither required nor invoked by this script.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build=${1:-"$root/build"}
output=${2:-"$root/sam31-smoke-output"}
case "$build" in /*) ;; *) build="$(pwd)/$build" ;; esac
case "$output" in /*) ;; *) output="$(pwd)/$output" ;; esac
if [ -e "$output" ] || [ -L "$output" ]; then
    printf '%s\n' "Output already exists: $output" >&2
    exit 2
fi
for executable in sam31 sam31_image sam31_video test_sam31_native test_sam31_quant test_sam31_ops; do
    if [ ! -x "$build/sam31/$executable" ]; then
        printf '%s\n' "Missing native executable: $build/sam31/$executable" >&2
        exit 2
    fi
done
mkdir -p "$output"
model="$root/sam31/fixtures/TEST_ONLY_native_sam31.gguf"
"$build/sam31/test_sam31_quant" > "$output/quant.txt"
"$build/sam31/test_sam31_native" "$model" > "$output/acceptance.json"
"$build/sam31/test_sam31_ops" "$model" "$root/sam31/fixtures/TEST_ONLY_operator_oracle.json" > "$output/operators.json"
"$build/sam31/sam31" synthetic --output "$output/inputs" --frames 6 > "$output/synthetic.json"
env PATH=/NO_EXECUTABLES SAM31_PYTHON=/NOT_PYTHON SAM31_SCRIPT=/NOT_A_SCRIPT \
    "$build/sam31/sam31_image" --model "$model" --allow-test-fixture \
    --image "$output/inputs/image.png" --point 1:0.25:0.32 --point 2:0.73:0.68 \
    --output "$output/image" > "$output/image.stdout.json"
env PATH=/NO_EXECUTABLES SAM31_PYTHON=/NOT_PYTHON SAM31_SCRIPT=/NOT_A_SCRIPT \
    "$build/sam31/sam31_video" --model "$model" --allow-test-fixture \
    --frames "$output/inputs/frames" --point 1:0.25:0.32 --point 2:0.73:0.68 \
    --output "$output/video" > "$output/video.stdout.json"
printf '%s\n' "PASS: native C++ image/video smoke; UNTRAINED weights, no accuracy claim. Results: $output"
