# Independent frame-major tracking

Prepare a manifest with `batch-track.py` (omit its `--run`). Then use:

```sh
python tools/annotator/tools/frame-major-track.py \
  --manifest /outside-repository/prepared/manifest.json \
  --driver /path/to/build/examples/sam3_track_batch \
  --output /outside-repository/frame-major
```

This only validates inputs and writes a plan. Add `--run` to explicitly start
CPU inference after applicable model quality gates have passed. Experimental
video permission is inherited from the prepared manifest. The optional
`--experimental-efficient-q8-0` changes native computation and requires its own
quality gate; quantized model storage alone does not establish INT8 computation.

`--efficient-prefix-cache` enables bounded exact RAM prefix reuse. Persistent
features additionally require `--prefix-cache-dir /outside-repository/features`
and an explicit `--prefix-cache-budget-mib N`. Cache keys and eviction are owned
by the native model cache. Cache contents are not tracker checkpoints. The wrapper holds an exclusive
writer lock on a persistent cache directory, including warm reads that may
trigger eviction; separate output directories cannot race on that cache.

One process loads one model and one transient state. Each active fruit owns one
independent tracker. The wrapper loads each image once, runs every active object
sequentially, then advances to the next source frame. Different seed frames map
to local tracker frame zero. The wrapper passes the full initialized manifest
frame count and each original seed frame to temporal position encoding; a
`max_frames` sweep limit never substitutes for full sequence length. Direct
streaming use can leave sequence length unknown (zero), retaining its streaming
position-encoding policy. The conditioning API clears stale object-specific
window selection; propagation selects windows using that object's own memory.
Only an exact reusable image prefix is shared, not object-specific SWR/SMR state.
Completed trackers are released immediately.

## Outputs and restart

The existing `batch-track.py` project, stable fruit ID, uncompressed column-major
RLE, segment, unresolved-state, and atomic JSON-write contracts are reused.
`runtime.json` records the driver, wrapper, shared annotation-contract and bridge hashes, resolved
Linux runtime-library hashes, architecture, and compute configuration. The driver uses the native loaded-ELF provenance resolver before model load.
Unavailable provenance or dependency changes after model load block persisted
resume rather than silently trusting an incomplete identity. Per-frame
causal dependency hashes cover the model, driver, thread count, compute mode,
that object's seed, and input bytes through that frame. Paths do not establish
identity, and changing another object's prompt does not invalidate this object.
A changed frame invalidates the affected causal suffix; changing model/runtime
or an object's prompt invalidates its whole track. Use a newly prepared manifest
when source bytes change. A different sequence requires another output directory.

An interrupted track replays from its seed to reconstruct temporal state.
Valid existing result prefixes are compared against replay and preserved,
including their original timings. Divergent replay stops rather than silently
replacing old output. Fully completed matching tracks are skipped. There is no
claim of native temporal checkpoint serialization or zero-cost resume.
`cache-statistics.json` stores current aggregate hit/miss/write/eviction counts
and native process peak RSS (`peak_rss_bytes`, zero if unsupported), and
`status.json` records progress. Compact SHA256 fingerprints of native low-resolution
logits and decoder saliency, vector lengths, and object logits accompany each
result for exact replay/equivalence checks without tensor dumps. Native diagnostics stay in `native.stderr.log`.

## Validation gates

`python tests/test_frame_major_track.py` runs synthetic tests without model
weights. These establish scheduler/output/resume contracts, not numerical
parity, cache correctness, RSS limits, or end-to-end tracking quality.

Before scaling, run identical public two-object, eight-frame manifests into
separate cached and uncached output directories, compare every RLE and score,
and record wall time, per-frame latency, cache counters, and peak RSS. Verify a
late seed after another object's propagation. Then measure eight objects before
larger private runs. Large runs remain blocked until these empirical gates and
the separate native video/Q8 quality gates have passed.

### Experimental full-encoder RAM reuse

`--efficient-fpn-cache` is an opt-in, default-off exact full-FPN cache prototype. It retains at most four ordered-window configurations for one image (approximately 84 MiB per entry, 336 MiB maximum resident, excluding transient/state/positional-encoding copies), independently of the persistent prefix cache. The driver reports its counters under `full_encoder`; the wrapper preserves those counters and includes the flag in output dependency/provenance identity. It does not persist tracker state or full-FPN entries across processes. Existing frozen runs remain unchanged; enable only for separately reviewed candidate experiments.
