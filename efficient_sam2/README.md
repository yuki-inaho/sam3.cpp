# Efficient-SAM2 ConvRot GGUF CPU integration

This is a distinct Efficient-SAM2 model path. It loads the complete validated
`efficientsam2` GGUF v1 state, including the six learned bypass tensors. It does
not rename EdgeTAM or silently execute ordinary dense SAM2 instead.

## Compute and supported scope

The default implementation inverse-rotates/dequantizes ConvRot INT8 weights
at load time and executes FP32 GGML CPU graphs. The GGUF file is compressed;
resident weights are FP32. There is no INT8-compute or speedup claim for this
path. W8A8 activation semantics are rejected. Experimental quantized CPU
arithmetic, if enabled separately, requires its own precision/speed gate.

The implemented video profile is **SAM2.1 base-plus, 1024-square preprocessing,
single object, forward, consecutive frames, stride 1**. Independent fruit
tracks can begin at any source frame; a frontend maps local frame zero back
to that source frame. Sparse routing and memory queues are independent per
track. Reverse tracking, shared multi-object state, new conditioning frames
inside an existing track, arbitrary memory stride, and large-model sparse
video are not supported. Reset and reseed rather than silently changing mode.

The packed image path supports the explicitly validated base-plus/large
architecture geometry, but only base-plus has numerical model evidence.
Large video is rejected even with the experimental flag.

SWR preserves padded windows through local blocks, handles selected valid
tokens at intermediate global attention, applies the learned residual
LayerNorm/down/ReLU/up bypass to cached background, then restores the full
grid before the final global block. SMR gathers different original spatial
indices for each of four memory layers and gathers the corresponding original
RoPE rows. Conditioning/latest memory and object pointers remain dense.
Memory features are rounded through BF16 to match the Python storage boundary.
Top-k ties use deterministic index order; PyTorch tie order is unspecified.

## Preprocessing and temporal contract

Video inputs use full-resolution libjpeg RGB decoding (`JDCT_ISLOW`, fancy
upsampling) and the exact uint8 Pillow bicubic resize pipeline before FP32
normalization. The tested decoder matches Pillow byte-for-byte on all 17 frozen
public JPEG frames. Standalone image prediction and legacy architectures keep
their existing image preprocessing. See [JPEG_CODEC.md](JPEG_CODEC.md) for
supported JPEG variants and explicit corruption/CMYK failures.

Efficient video preserves FP32 in decoder transposed convolutions, uses exact
ERF GELU and supplies both empty prompt tokens on unprompted propagation (the
explicit empty point plus PromptEncoder padding). It applies the object-presence gate to both
conditioning and propagated masks before resizing or memory encoding. For
known-length inputs, set `efficient_sequence_length` to the full initialized
source length and `efficient_source_start_frame` to the original seed position.
The pointer-position denominator is `min(sequence_length,16)-1`. A one-frame
sequence permits conditioning only; out-of-range propagation is rejected.
Unknown streaming length zero uses denominator 15. The number of frames chosen
for a test run is not the initialized source length.

## Build and tests

Install a C++ compiler, CMake, and the system libjpeg development package
(for example, Debian's `libjpeg62-turbo-dev`). Python/Pillow/NumPy enable the
independent preprocessing fixture tests.

```sh
cmake -S . -B build -DSAM3_METAL=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
build/efficient_sam2_inspect /path/to/model.gguf
```

The storage tests reject inconsistent layer/count metadata, malformed bounds,
unsupported activation modes, incomplete state, duplicate JSON keys and v1
non-Linear INT8 tensors. The independent policy tests cover padded layouts,
routing, frame-indexed warmup, 204-token masks, original RoPE coordinates,
temporal positions, sequence bounds, absent conditioning and BF16 boundaries.
Loaded-library provenance and persistent-cache integrity have separate tests.

## Public temporal parity

The explicit validation runner selects the verified base-plus video preprocessing path:

```sh
mkdir -p /private/evidence/native
EFFICIENT_SAM2_DUMP_LOGITS=1 EFFICIENT_SAM2_TRACE_DIR=/private/evidence/native \
  build/efficient_sam2_track model.gguf /public/frames_17 /private/evidence/native \
  73 245 309 783 17 > /private/evidence/native/run.jsonl 2> /private/evidence/native/run.log
python efficient_sam2/compare_video.py /private/evidence/native /private/evidence/python \
  --frames 17 --output /private/evidence/native/comparison.json
```

The frozen public example uses RGB frames 480×848 and an original-image box
`[73,245,309,783]`. Both implementations directly resize to a 1024×1024 square;
this is not SAM1-style aspect-ratio padding. The Python oracle must use the
Efficient WB_Hiera all-window conditioning path and the same prompt. A
full-mask-initialized oracle is a different protocol.

The acceptance protocol is four independent 17-frame box tracks (68 cases),
mean mask IoU ≥0.99, minimum ≥0.98 and all finite raw logits, using fixed public
objects 0, 1, 2 and 4. It is separate from the full-mask-initialized quantization
quality gate. The historical one-object result is not the expanded gate.

The restored-FP32 native implementation passed the expanded 68-case gate on
2026-10-13: mean mask IoU **0.9999975421**, minimum **0.9998328598**, all finite
raw logits. Objects 0, 1 and 4 had pixel-identical masks on all 17 frames.
The tested GGUF SHA256 is
`0c58eac7150d1b24970bc2ff1537a8847faec9e971be585bfd55411ef8a6b32b`;
the frozen runner SHA256 is
`e5c2a3ce81c318710288241b6a0b4f715b285b999cb4cd412477546228c89d61`.
The experimental Q8 second-quantization mode failed the separate native temporal
gate. It remains opt-in pending the separately stated practical-quality and
private identity assessment. See the candidate record below.
SWR is active from frame 2; SMR begins at frame 7 and retains 9,212 spatial keys
per layer. Numerical agreement does not establish human-reviewed identity or
occlusion accuracy on a private fruit dataset.

## Annotation protocol

`sam3_serve` reports runtime capabilities in its ready JSON. The verified
base-plus video mode is available on normal startup; no research override is
needed. `--experimental-efficient-video` remains an optional compatibility
alias for older callers. `--efficient-image-only` selects the separate image
preprocessing contract and disables video. Large-model sparse video stays
unsupported and is rejected explicitly.
The loopback frontend bridge must expose the returned capability rather than
inferring tracking support from the file extension.

After `load`, set `track_config <full-sequence-length> <source-seed-index>`
when known, queue `box` or positive `point`, then `track_add <positive-id>`.
Configuration is rejected during an active track; reset clears it.
`track_frame <path>` returns compact RLE masks and preserves that external ID.
A second `track_add` is rejected until `track_reset`. Reset clears prompts,
temporal state and the loaded image, so reload before reseeding. Keep the
source frame index, fruit ID and segment ID in the project. An empty/absent
mask is not evidence that a different fruit has the same identity.

Debug logits and routing traces are opt-in environment variables. Normal
annotation should persist compact masks, scores, frame identities and timing,
not per-frame encoder or FP32 logit arrays. Tracking responses expose `absent`
when the observed mask has no foreground; this is not a claim that the fruit's
identity ceased to exist. Empty Efficient masks use a valid zero bounding box. Review model-assisted annotations
before calling them ground truth. See [NOTICE.md](NOTICE.md) for licensing.

## Opt-in CPU optimization and reusable features

`--experimental-efficient-q8-0` selects the separately gated GGML Q8_0 candidate.
It is a second lossy quantization with dynamic quantized activations. It must
not be confused with the default restored-FP32 compressed-storage runtime.
Both native-reference comparisons failed the fixed minimum mask-IoU gate: the original emulator
reference reached 0.9734237740, and a weight-byte-matched native-H4 reference
reached 0.9411180946 (required minimum 0.98). Its Python-only quantization quality
checks passed, which did not establish native temporal parity. A separate direct
native/manual-GT box evaluation obtained J/F 0.9134281/0.9751776 for Q8 and
0.9108903/0.9750823 for restored FP32 on 16 sampled cases. These small-sample
quality results are being considered separately from fidelity; the original
failed gate remains failed. Practical Q8 acceptance still requires measured
whole-model performance and private identity review. No end-to-end Q8 speedup
is claimed here.
See [cpu_quant/README.md](cpu_quant/README.md) for its exact selection, arithmetic,
portable fallback, fixtures and benchmark limitations.

`--efficient-prefix-cache` enables one 49 MiB FP32 RAM entry. Add
`--prefix-cache-dir PATH --prefix-cache-budget-mib N` for explicitly bounded,
reusable disk artifacts. The prefix is track independent; every fruit still
keeps its own full temporal state and routing. Frame-major scheduling permits
same-frame reuse. Actual peak memory includes graph workspaces and copies and
can exceed the 49 MiB cache residency increase.

Features are keyed by model contents, original RGB and geometry, preprocessing,
precision, policy, actual resolved thread count, CPU features, source/build
configuration and loaded executable/library contents. Changing these invalidates
persistent features. Cache failures recompute safely. Linux runtime provenance
is exposed in ready responses and by `--runtime-provenance`; unverified runtime
identity disables disk caching without disabling inference. See
[prefix_cache/README.md](prefix_cache/README.md) for export/import, integrity,
trusted-directory and single-writer requirements.

### Exact full-FPN RAM reuse

`--efficient-fpn-cache` is an additional, default-off base-plus option. It keeps
up to four unconditioned encoder-output entries for the current image, keyed by
the complete existing model/input/preprocessing/precision/backend identity and
**exact ordered SWR window IDs**. Empty IDs mean all25 windows. The cache never
unions selections or approximates features. It stores neither decoder outputs
nor temporal memory, pointers, masks or routing state.

Each entry contains FP32 `neck_trk[0..2]`, shapes `[256,256,256,1]`,
`[256,128,128,1]`, `[256,64,64,1]`: **84 MiB per entry, at most336 MiB resident
payload**, plus normal state, PE, graph workspaces and temporary copies. The
cache is RAM-only and session-owned; the prefix cache remains the separate
persistent/exportable feature format. Image-identity changes clear entries,
and LRU eviction bounds residency. Frame-major scheduling is needed to reuse
same-image entries across independent objects. Different routes may eliminate
most hits; cache presence alone is not a speedup claim.

The public candidate test exercised46 predictions, including duplicate and
mixed routes, late seeding, drop/reseed, active SMR and real LRU eviction. All
logits, masks and full routing/SMR traces were exact against the frozen native
reference or a matching cache-off late-seed run. All three restored FPN tensor
hashes matched on every hit. Standalone helper tests cover invalid shapes,
nonfinite values, exact signed-zero bits, key ordering and bounded residency.
Use `full_encoder` counters returned by `cache_stats`/batch `stats` to observe
actual hits separately from prefix-cache lookups. Debug FPN hashes and complete
routing traces are opt-in via `EFFICIENT_SAM2_FPN_TRACE` and
`EFFICIENT_SAM2_TRACE_STDERR`; leave them off for ordinary annotation.
