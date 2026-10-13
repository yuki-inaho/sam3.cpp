# Exact Efficient-SAM2 prefix feature cache

This helper caches only **track-independent FP32 prefix tensors**, immediately
before SWR packing. It never serializes live ggml pointers, decoder outputs,
conditioned features, object pointers, masks, SMR queues or tracker state.
It supports session-owned one-entry RAM reuse and optional reusable disk files.
There is no database dependency or global feature cache.

## Exact boundary and measured motivation

Base-plus, trained 1024-square geometry, channel-fast layout:

1. Output after block5: `[448,64,64,1]`, 7MiB
2. Stage0 skip output after block1: `[112,256,256,1]`, 28MiB
3. Stage1 skip output after block4: `[224,128,128,1]`, 14MiB

Together these are exactly49MiB, unchanged IEEE FP32. No FP16 compression or
rounding is permitted for this exact cache. A native CPU4 profile found prefix
costs3368.8/9702.2ms and2874.9/8505.9ms of dense-frame encoder time, then
3203.7/6611.5ms for a sparse frame (34–49%). Concurrent unrelated CPU work was
present: these are phase observations, **not a measured cache speedup**.

The model/session owns the cache so it can survive independent tracker resets.
One-frame RAM reuse needs frame-major scheduling. Each object retains its own
router, memory and SMR state and processes frames in order. Identical RGB alone
does not permit reusing the rest of the encoder: full-feature reuse would also
require the exact ordered SWR selection, in a different key/representation.
This helper intentionally implements only the prefix representation.

## API and key responsibilities

`cache.h` and `digest.h` are C++14-compatible. The isolated library implementation
uses C++17 filesystem; link target `efficient_sam2_prefix_cache`. All input tensors
are owned CPU vectors. `lookup` returns an immutable shared entry; mutable native
work must copy the required tensors. `store` takes its own copy for RAM residency.
Only one entry is retained; external shared-pointer holders can extend old entry
lifetimes beyond the helper's resident-byte budget.

Key fields are all mandatory:
- SHA256 of the actual loaded model/GGUF contents
- SHA256 of canonical original RGB bytes **and geometry**
- SHA256 of full model/preprocessing/policy/quantization configuration
- Explicit preprocessing representation, e.g. image-bilinear-u8-v1 versus
  sam2-video-libjpeg-rgb-islow-fancy-pil-bicubic-u8-v2
- Precision/quantization-selection identity, CPU/backend/ISA identity,
  implementation/build fingerprint, resolved encoder thread count and original input dimensions
- On Linux, the content identity of the executable and all loaded file-backed
  ELF libraries after backend initialization; unsupported/unverifiable identity
  disables persistent caching while RAM reuse and inference remain available

Never substitute a filename, modification time, guessed hash, or only a clean
Git commit when the compiled source has uncommitted changes. A source/build/flags
fingerprint must invalidate old features when semantics change. Python orchestration
may pass hashlib-generated verified identities. Direct native callers can reuse
`sha256_file`, `sha256_bytes` and `sha256_rgb` from the same reviewed digest wrapper.

The RGB digest is precisely SHA256 over:
`b"efficient-sam2-rgb-v1\n" + struct.pack("<QQ", width, height) + contiguous_RGB`.
Policy serialization belongs to the caller and must be deterministic and complete.
A prefix key does not depend on object ID, tracker frame index or SWR selection.
The stored key is length-delimited and includes the fixed representation domain.
The filename is SHA256 of this complete canonical key; reads also verify the full
stored key, shapes, length, version and CRC32.

Misses/rejections must **recompute normally**, with counters and the returned
reason available for diagnostics. Cache failures must not make otherwise-valid
inference fail. `Hit` distinguishes RAM, disk, absent and rejected entries.

## Persistent storage, export/import and safety

`Options.disk_directory` empty and disk budget0 disable disk. Set both explicitly
to enable it. The helper owns only its `efficient-sam2-prefix-v1` subdirectory
and validated feature filenames within it. `write_file`/`read_file` support direct
artifact export/import with the same strict key/format checks, across sessions.
`artifact_path` locates a canonical entry; copying that immutable file preserves
its representation without requiring a DB. Exports require an existing parent.

- Disk byte budget includes complete entry headers. Oldest-accessed entries are
  evicted before a new temporary file is written, so normal writes fit the budget.
- Only a validated owned header/key/shape and matching payload checksum permit
  deletion. Unrelated, corrupt or unknown-version files are not removed or
  overwritten automatically; they can block the budget and cause a safe miss.
  `clear(include_disk=true)` reports when unvalidated entries were preserved.
- Temporary files use C11 exclusive-create mode (`wbx`), never truncating a
  planted file or symlink. Publication is a same-directory rename after complete
  writes/flush/close. Interrupted temporaries are reclaimed only if their complete
  ownership header validates; unrecognizable partial files remain untouched.
- Symlink components and parent traversal are rejected. The cache directory must
  remain a trusted local directory under the caller's control. This is not a
  defense against a hostile concurrent filesystem administrator.
- Only one process may write/evict/clear a directory at a time; separate processes
  may read published immutable files. There is no multi-writer lock/database.
- CRC32 detects accidental corruption, **not adversarial authenticity**. Source
  identity uses SHA256; cache content is not signed. Power-loss durability/fsync
  is not promised. Failed imports are never treated as valid data.
- Format v1 uses little-endian IEEE FP32 and exact fixed shapes; unsupported
  endian/representation is rejected. x86/ARM little-endian portability does not
  imply cross-backend bit-identical inference or permission to reuse different keys.

An LRU smaller than the working set can thrash during serial full-video sweeps.
For F distinct frames, the raw prefix set alone is F×49 MiB; H disk hits
read H×49 MiB of tensor payload, plus headers.
Prefer one-frame RAM reuse for a frame-major driver when feasible. Budget and
measure disk mode separately, including content hashing, validation, copying and
I/O. The helper reports resident/disk bytes, hits, misses, rejections and evictions.

## Verification and reproducibility

```sh
: "${EVALUATION_DIR:?Choose an output directory outside the repository}"
cmake -S efficient_sam2/prefix_cache -B "$EVALUATION_DIR/prefix-cache-build" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$EVALUATION_DIR/prefix-cache-build" -j4
ctest --test-dir "$EVALUATION_DIR/prefix-cache-build" --output-on-failure
python efficient_sam2/prefix_cache/test_hashlib.py \
  "$EVALUATION_DIR/prefix-cache-build/test_efficient_sam2_prefix_digest" \
  "$EVALUATION_DIR/hashlib-test.bin"
```

Tests cover exact FP32/signed-zero round trips, independent changes to every key
domain including preprocessing, RAM/disk reopening, corrupted data/header/shape,
old versions, truncated/trailing payloads, empty/mismatched identities, budgets,
clear/invalidation, unrelated filenames, symlinks/path traversal, interrupted
writes and concurrent reads during atomic publication. Tests create a new private
test directory and remove only that generated directory on success.

SHA256 known answers cover empty/abc/multiblock/million-a plus streamed files and
RGB geometry. `test_hashlib.py` independently cross-checks Python hashlib.
The SHA256 and rotate-bits implementation is copied unchanged from official
llama.cpp commit `bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444`; public-domain and MIT
license/provenance are preserved in `third_party/NOTICE.md` and original files.

Release and ASan/UBSan tests are required after final edits. LeakSanitizer is not
available in the traced execution environment; no leak-check pass is claimed.
Native cached/uncached logits, masks, SWR and SMR parity and actual end-to-end
latency/RSS remain separate integration gates owned by the native implementation.
