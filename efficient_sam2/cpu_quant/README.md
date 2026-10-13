# Experimental portable CPU quantized Linear probe

## Status and exact candidate

This is an **opt-in experimental native candidate**, disabled by default.
The Efficient-SAM2 loader selects 191 verified Linear weights for Q8_0 and keeps
8 width-112 weights as FP32. Other architectures are rejected by this option.
The candidate remains **opt-in, pending practical performance and identity
assessment**. Python full-model quality passed, but native four-object temporal
parity failed the unchanged minimum mask-IoU threshold of 0.98. Synthetic Linear
gains do not establish end-to-end gains. The already accepted numerical path is
the compressed ConvRot GGUF restored to FP32, with separately validated exact
encoder-feature reuse.


On 2026-10-13, the frozen 68-case box-initialized native comparison against the
original Q8 emulator obtained mean/minimum IoU **0.9972512518 / 0.9734237740**.
A distinct, explicitly recorded `native-h4-f32-v1` Python restoration policy
matched actual native restored and packed bytes on seven real-layer fixtures,
including width112 fallback. Its unchanged Python full-mask quality gate passed,
but the matched 68-case native box comparison still failed: mean/minimum
**0.9959965973 / 0.9411180946**. Objects0 and4 failed, objects1 and2 passed.
All used SWR window IDs matched; the worst object0 discrepancy occurred before
SMR pruning. These observations do not identify the exact remaining numerical
cause. That fidelity threshold was not relaxed and the failed result is retained.
A separate direct native-versus-manual-GT assessment used the four fixed frame0
boxes and scored frames4/8/12/16: Q8 J/F/J&F was
0.9134281122/0.9751776370/0.9443028746; restored FP32 was
0.9108902637/0.9750822755/0.9429862696. Direct native Q8-versus-restored-FP32
mask agreement mean/p5/minimum was 0.9923943263/0.9768474100/0.9456099456.
These are practical-quality observations on 16 cases, not a retroactive
fidelity-gate pass or proof of private fruit-identity accuracy. No deployment
speed claim is made without whole-model measurements.

Candidate: **inverse-ConvRot FP32 → ggml Q8_0 weights + dynamic Q8_0 activations**.
This is a second lossy quantization, not the existing ConvRot W8A16 format and
not storage-only compression. Existing `sam31/src/quant.cpp` was inspected:
`sam31_linear_w8a32` rotates floating activations and performs scalar double
accumulation with INT8 weights. It is useful as an oracle, but is not a SIMD
integer backend. Reuse ggml dispatch rather than introducing another backend.

`weights.h` uses public ggml quantization APIs; the consumer uses ordinary
`ggml_mul_mat` with persistent packed weights. No per-inference weight conversion,
custom instruction set, padding, or dropped tail. Q8_0 has 32 weights plus an FP16
scale (34 bytes). Width 112 is not divisible by 32 and remains exact FP32;
224/448/896 and aligned expansion widths are eligible. ConvRot groups16/16/64/64
belong to the original inverse rotation and do not change Q8_0's block size.

Selection MUST intersect GGUF `quantization.layers` with verified Linear consumers,
then registered native weight tensors. `manifest.py` verifies original nn.Linear
module identity/dimensions and records every candidate and fallback. Do not apply
the policy to every 2D `.weight` tensor: embeddings/positional tensors are not Linear.
Native `sam2_gguf_runtime_name` maps source names to registered native names.
Keep the evaluated selection manifest with each candidate result.

Raw pointers in `pack` require contiguous row-major FP32; strided callers must
materialize explicitly. Invalid shapes/count/overflow/NaN/Inf reject. Nonzero
blocks whose FP16 scale rounds to zero, or weights exceeding representable scale,
fall back for the entire matrix to exact FP32. All-zero blocks remain valid Q8.
No native activation-range fallback has been added: ggml's stock behavior applies.
The emulator rejects nonfinite/overflowing activation conversion rather than
silently passing it. Finite-output model gates remain mandatory.

## Official source review and portability

Official llama.cpp checkout read on 2026-10-13:
`bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444` (MIT).
No code copied from that checkout; implementation uses this repository's existing
vendored ggml (vendor commit `b0b5ee969e7dc1327aaddf6f1b1eab80303543fe`).

- [Q8_0 layout](https://github.com/ggml-org/llama.cpp/blob/bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444/ggml/src/ggml-common.h)
- [Deterministic weight quantization](https://github.com/ggml-org/llama.cpp/blob/bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444/ggml/src/ggml-quants.c)
- [CPU type traits and dispatch](https://github.com/ggml-org/llama.cpp/blob/bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444/ggml/src/ggml-cpu/ggml-cpu.c)
- [x86 quantization and integer dots](https://github.com/ggml-org/llama.cpp/blob/bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444/ggml/src/ggml-cpu/arch/x86/quants.c)
- [Generic fallback](https://github.com/ggml-org/llama.cpp/blob/bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444/ggml/src/ggml-cpu/quants.c)

Vendored files actually tested (git blob identifiers):
- ggml-common.h: `92cf739e7a7beab7c4ae1717ab270f73d41a01ff`
- ggml-quants.c: `48695a61ea3360bec9b0a1de74c80c3c305494b0`
- ggml-cpu.c: `5fbca925967e89f01a2ef0f04bc58816bf514053`
- arch/x86/quants.c: `74d699f633d3ace911c16428ff73c6ef5f2cd355`

The same ggml dispatch provides architecture kernels and generic fallback.
A separate no-AVX/no-AVX2/no-FMA/no-F16C build passed the numerical CTest on this
x86 host. This establishes the generic path here; it is not a tested ARM/Windows
performance claim. Benchmark RSS reporting uses POSIX getrusage.

## Arithmetic and independent oracle

Weight blocks use FP32 maxabs/127; integer quantization uses reciprocal of this
unrounded FP32 scale and C roundf (ties away from zero). Scale is stored FP16.
AVX2 activation blocks use FP32 127/maxabs with nearest-even integer rounding and
FP16 scale. Q8 integer dot products are scaled and accumulated in FP32. Bias stays
FP32 in actual Linear consumers. ISA-specific ties/accumulation can differ.
`emulator.py` targets the measured x86 AVX/AVX2 behavior explicitly, not all ISAs.

NumPy independently matched every packed integer and FP16 scale byte for six
representative matrices; matmul output relative L2 <=4.05e-7. Torch evaluator
relative L2 <=2.91e-7. A detected Torch scalar-over-tensor reciprocal-multiply
rounding discrepancy was fixed by explicit tensor/tensor division, before model
evaluation. Do not simplify that division. Emulator dequantized FP32 matmul is
mathematically equivalent, not a bitwise imitation of SIMD accumulation order.

`test_emulator.py` also checks noncontiguous activations, empty SWR batches, zero, tiny/huge weights,
NaN/Inf, widths16/32/33/112/224 and deliberately chosen half-integer ties.
The C++ probe checks default-disabled and width112 bit-exact FP32 fallback,
shape/count/overflow rejection, finite outputs and scalar-double output oracles.

## Preregistered benchmark protocol

Registered with workdoc owner before collecting results:
(K input width,N output width,M tokens) = (112,336,64), (224,672,64),
(448,1344,64), (896,2688,64), (448,1792,256), (1792,448,256).
One and four threads, same deterministic seed20261013, normally distributed
weights scaled by1/sqrt(K), identical activation/weight samples for both paths.
Three warmups each,11 alternating paired AB/BA samples. Report median paired
FP32/candidate ratio, not a ratio of independently selected best samples.
Scalar-double oracle: FP32 relativeL2<=1e-5, candidate<=.02, all finite.
These tolerances are microbenchmark checks, never model acceptance criteria.

Measurement excludes graph creation and weight packing; it includes ordinary
`ggml_mul_mat` dynamic activation quantization. Persistent threadpool and graph
workspace are reused. Pack cost and combined two-graph setup cost are reported
separately in CSV. No model/image preprocessing, attention, resizing, stage
allocation, or graph rebuilds are timed.

Host AMD EPYC9V74, container CPU0–8 available. Existing Release ggml build:
GGML_NATIVE=OFF, AVX=ON, AVX2=ON, FMA=ON, F16C=ON,
AVX512/VNNI=OFF, BLAS=OFF, OPENMP=ON, CPU_REPACK=ON.
The benchmark explicitly selects1/4 ggml threads, no BLAS is involved. Background
work shares the host: ratios fluctuate, including the identical112 fallback.
Preliminary shared-host microbenchmark run (noisy, not a controlled end-to-end gain):

|K|N|M|threads|candidate|FP32 ms|candidate ms|paired speedup|pack ms|candidate relL2|
|---|---|---|---|---|---|---|---|---|---|
|112|336|64|1|f32|0.2447|0.2624|0.980|0.067|1.19581e-07|
|224|672|64|1|q8_0|0.5932|0.4489|1.309|0.709|0.00756108|
|448|1344|64|1|q8_0|2.2503|1.7200|1.382|2.796|0.00759491|
|896|2688|64|1|q8_0|9.3294|6.1359|1.511|14.510|0.0075761|
|448|1792|256|1|q8_0|14.9537|11.0625|1.270|4.071|0.00754828|
|1792|448|256|1|q8_0|12.5980|8.4568|1.454|5.290|0.00754505|
|112|336|64|4|f32|0.0829|0.0839|0.990|0.064|1.19581e-07|
|224|672|64|4|q8_0|0.1804|0.1500|1.154|1.146|0.00756108|
|448|1344|64|4|q8_0|0.6350|0.5086|1.249|3.135|0.00759491|
|896|2688|64|4|q8_0|2.5884|1.8994|1.309|13.141|0.0075761|
|448|1792|256|4|q8_0|4.1577|3.3084|1.208|5.396|0.00754828|
|1792|448|256|4|q8_0|3.9663|2.6940|1.397|3.772|0.00754505|

For eligible matrices, persistent weight bytes are34/128=26.5625% of FP32
(73.4375% reduction for those matrices only). CSV reports exact bytes, not a
whole-model RAM claim. Process peak RSS was approximately42MiB because both
FP32 and candidate copies, oracle arrays and graph buffers coexist. getrusage
reports a cumulative process high-water mark; it is **not** isolated candidate
peak or net memory savings. No whole-model load conversion/RSS/end-to-end latency
has been measured by this probe.

## Reproduction

From repository root (all weights, builds and results remain outside Git):

```sh
: "${EVALUATION_DIR:?Set an output directory outside the repository}"
R="$EVALUATION_DIR"
mkdir -p "$R/results/cpu-quant-oracle"
c++ -O3 -std=c++14 -Wall -Wextra -Werror -Iggml/include   efficient_sam2/cpu_quant/benchmark.cpp -Lbuild/ggml/src   -Wl,-rpath,"$PWD/build/ggml/src" -lggml-cpu -lggml-base   -o "$R/cpu-quant-benchmark"
"$R/cpu-quant-benchmark" 1 11 "$R/results/cpu-quant-oracle"   > "$R/results/cpu-quant-1.csv" 2> "$R/results/cpu-quant-1-samples.csv"
"$R/cpu-quant-benchmark" 4 11   > "$R/results/cpu-quant-4.csv" 2> "$R/results/cpu-quant-4-samples.csv"
python efficient_sam2/cpu_quant/test_emulator.py   "$R/results/cpu-quant-oracle"
cmake -S efficient_sam2/cpu_quant -B "$R/cpu-quant-portable-build"   -DCMAKE_BUILD_TYPE=Release -DGGML_AVX=OFF -DGGML_AVX2=OFF   -DGGML_AVX512=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_OPENMP=OFF
cmake --build "$R/cpu-quant-portable-build" -j4
ctest --test-dir "$R/cpu-quant-portable-build" --output-on-failure
```

Use an existing CMake installation on PATH and a Python environment with NumPy
and Torch. Build output and test logs are `results/cpu-quant-portable-{build,test}.log`; raw timing pairs are
`cpu-quant-{1,4}-samples.csv`. The already-restored original GGUF used for future
model comparison has SHA256
`0c58eac7150d1b24970bc2ff1537a8847faec9e971be585bfd55411ef8a6b32b`.

## Acceptance requirements for any future candidate

- Fixed public SA-V TRAIN16 cases, identical manual-mask initialization:
  mean/p5/min FP32-mask IoU >=.98/.95/.90; J/F losses<=1percentagepoint; finite.
- Kernel-validated emulator selection must exactly match native integration.
- Native temporal box trajectory parity is a separate implementation check.
- Downstream private-data identity checks remain separate and are not inspected by this work.
- Integration exists behind the explicit experimental option. Any future
  accepted candidate must measure load conversion, persistent/peak memory and
  end-to-end warmed latency separately.
  Profile attention, allocation/build, resize and matmul before further changes.

Additional sanitizer check: ASan+UBSan on the helper/probe (vendored ggml not
instrumented) initially reached process exit, where LeakSanitizer was unsupported
under the executor's tracing environment. Leak checking is not claimed as passed.


## Whole-model retest on another CPU

The retained runner is `../benchmark_video.py`; its fixture-only orchestration
checks are `../test_benchmark_video.py`. The checked-in
`../benchmark_protocol.public.json` preserves the frozen CPU4, full17-source,
first4-frame object1 protocol and public input/model hashes. It excludes one
2-frame warmup per variant, then measures three paired orders AB/BA/AB with
cache disabled in both modes. A and B use the same executable and loaded library
contents. It records every raw sample, model load/setup separately, phase events,
per-frame wall time, full process wall time and peak RSS where supported. Nested
phase totals overlap. Reported weight-buffer MiB are rounded log observations.

```sh
python efficient_sam2/test_benchmark_video.py
python efficient_sam2/benchmark_video.py \
  --executable build/efficient_sam2_track \
  --model /path/to/efficient-sam2-base-plus-convrot-w8a16.gguf \
  --frames /path/to/public/frames_17 \
  --protocol efficient_sam2/benchmark_protocol.public.json \
  --output /path/to/new-empty-benchmark-directory
```

This whole-model timing protocol was prepared but not run for the delivery;
accepted encoder-cache workflow timing was prioritized. A faster/newer CPU does
not guarantee Q8 gains, identical ISA rounding or acceptable model accuracy.
Retest both correctness and practical performance there. Keep Q8 opt-in and
preserve the recorded fidelity failures, even if direct task-quality is adequate.
For different input/model/thread protocols, preregister a separate protocol and
adapt the explicit validation runner rather than relabeling these fixed cases.

The companion Efficient-SAM2 repository's
`tools/quantization/NATIVE_Q8_EXPERIMENT.md` documents the optional
`--restore-policy native-h4-f32-v1` evaluator and its explicit source/manifest
provenance. This repository provides `restore_policy.py` and synthetic
`test_restore_policy.py`; they reuse the existing GGUF state loader and Q8
quantizer. The default dense restoration is unchanged. Exact restoration
fixtures do not override the recorded native temporal fidelity failures.
