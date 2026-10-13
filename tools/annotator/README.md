# sam3 Annotator

SAM-assisted image and video annotation on top of `sam3.cpp`. Click a point,
save a draft mask, retain explicit fruit IDs across frames, and export COCO.

The frontend is plain HTML/CSS/JS with **no dependencies and no build step**;
the desktop shell is Tauri 2. Segmentation runs in `sam3_serve`, a long-lived
child process.

## Why a server process

`sam3_seg` re-encodes the image on every invocation. On this machine that is
about 3 seconds before any prompt is even considered, which makes
click-to-segment unusable.

`sam3_serve` encodes once and then stays alive:

| step | measured |
|------|----------|
| model load + image encode | ~2.9 s, once |
| each prompt afterwards | **~0.5 s** |

## Build

```bash
# 1. the segmentation server (from the repository root)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sam3_serve --parallel

# 2. the desktop app
cd tools/annotator/src-tauri
cargo run                 # development
cargo build --release     # release binary in target/release/
```

Linux needs the WebKitGTK development packages Tauri 2 requires
(`webkit2gtk-4.1`, `javascriptcoregtk-4.1`).

The app finds `sam3_serve` next to its own executable, then in `build/examples/`
resolved **relative to that executable** (not to the working directory, so it
does not matter where the app was launched from). Override with the
`SAM3_SERVE` environment variable. Which binary is spawned is deliberately not
something the page can choose.

## Use

1. **モデル開始** with a `.ggml` path (the vendored `models/edgetam_q8_0.ggml`
   works). This is the slow step.
2. **画像を開く**. The image is encoded once, which also takes a few seconds.
3. Click on an object. `Shift`+click (or right-click) marks background.
   Switch to 矩形 to drag a box prompt instead.
4. **下書き保存** (`Enter`) saves the current mask with its Fruit ID; set the
   label first if you want something other than `tomato`. This is not review approval.
5. **COCO を書き出す** downloads a COCO JSON.

| key / gesture | action |
|-----|--------|
| `Space` | run segmentation |
| `Enter` | accept the current mask (also works while typing the label) |
| `Ctrl+Z` | undo the last prompt point, then restore the last deleted instance |
| `Esc` | clear the prompt |
| wheel | zoom at the cursor |
| `Alt`+drag, middle-drag, or 移動 mode | pan |

Clicking an instance row selects it: the other masks dim and its bounding box
is drawn. Zoom and pan survive a window resize; 全体表示 re-fits.

The whole point set is re-sent on every run, because SAM conditions on all
points at once — adding a negative point has to re-run with the earlier
positives still attached or it segments something unrelated.

## Masks and export

Masks stay in COCO uncompressed RLE from the C++ side all the way to the
exported JSON, so **holes, detached pieces and overlaps between instances
survive** — none of them can be represented by a polygon approximation.

`segmentation.size` is `[height, width]` and `counts` is column-major, matching
COCO and CVAT.

## Tests

```bash
npm test                                      # RLE, project and tracking checks
PLAYWRIGHT_PATH=<path-to-playwright> \
  node tools/browser-test.mjs                  # UI checks, sandboxed Chromium
```

The node suite covers the RLE codec (including a mask with a hole and a
detached island, and malformed run lists that must be rejected rather than
decoded into the wrong pixels) and the COCO document shape.

The browser suite drives the real page with `window.__TAURI__` stubbed,
covering click → segment → accept → export, the colour/instance bookkeeping,
the in-flight guard, keyboard handling, pan/zoom, undo, the discard
confirmation, and the panel layout under 40+ instances.

Both replay `fixtures/serve-point-315-250.json`, a recorded `sam3_serve` reply
whose mask was verified pixel-for-pixel against `sam3_seg`'s PNG output. It is
vendored, so neither suite needs a model, a GPU, or anything in `/tmp`.

## Limits

- Only point and box prompts. No brush, eraser, or polygon editing yet.
- Tauri currently offers single-image selection. The loopback browser route adds
  sequence navigation, tracking, and project save/reload as described below.
- Undo covers prompt points and instance deletion, not label edits.
- Current-frame instances use full-image RGBA display layers. Other frames retain
  only portable RLE; very large current-frame instance counts still use more memory.
- A plain static browser page has no model/filesystem bridge. Use the dedicated
  loopback launcher below to enable the explicitly allowlisted local native backend.
- Only tested on Linux.

## Private loopback video projects

The browser frontend can run without Tauri/Cargo. This stdlib-only service binds
**127.0.0.1 only**. It exposes fixed UI assets, exactly allowlisted model files,
and images from explicitly selected roots. There are no external requests or
uploads. Never place private images, models, annotations, or screenshots in Git.

```bash
uv run --no-project python tools/annotator/tools/serve-browser.py \
  --serve build/examples/sam3_serve \
  --model /private/models/model.gguf \
  --image-root /private/sequence \
  --annotation-root /private/annotations \
  --port 8765
```

The output directory must already exist and be outside this repository. Omit
`--annotation-root` for download-only saves. The service saves atomically to
`annotation-project.json` within that explicit directory; browsers cannot choose
an arbitrary write path. Model/image roots are never inferred from a request.
Host and Origin checks restrict the API to the local page. Symlink changes,
traversal, newline paths, non-finite prompts, and non-CPU startup are rejected.
Do not proxy this service onto a public interface.

Each image directory is a separate sequence. Previous/next and the frame picker
retain per-frame masks. Fruit IDs are allocated per sequence and are never
recycled after deletion. Enter an existing Fruit ID to correct it on a frame.
`segmentId` identifies a continuous observation separately from `fruitId`;
re-seeding/reappearance gets a new segment and remains unverified/unresolved.
The UI does not infer re-identification from overlapping boxes.

Saving a project preserves every frame's uncompressed COCO RLE, fruit and segment
IDs, source-frame index, prompts, unresolved IDs, label, score and review state.
Source images and absolute input paths are not embedded. The loopback “保存済みを開く” button reads that fixed saved file without a browser
file upload. File-based reload accepts this
project format or the extended COCO export, and validates dimensions, exact RLE
coverage and unique IDs/references before replacing the active project. COCO
imports require explicit `fruit_id`/`track_id`; absent identity is not guessed.
Legacy missing segment IDs become `fruitId:legacy-unverified`; malformed explicit
segment IDs are rejected. The selected image sequence must match all frame keys.

Predictions are **drafts**. Saving/exporting never marks them reviewed. Use the
review button only after checking both the visible mask and its video identity.
An empty mask records the fruit as unresolved. Reappearing masks remain
unresolved and begin a fresh segment, even though the explicit fruit ID persists.

Native propagation is enabled only if `sam3_serve` reports `video_tracking:true`.
Each fruit runs independently: `track_reset`, `load` the seed frame, configure
sequence bounds with `track_config <fullSequenceLength> <seedPosition>`, queue
the prompt, `track_add <fruitId>`, then `track_frame <path>` for subsequent frames.
The full length is the initialized image-list length, never the remaining sweep
length. Seed position is zero-based within that list: source labels 400/401 in
a two-frame project still use seed positions 0/1. Invalid bounds fail before
seeding. Reset clears these bounds; the UI reconfigures each fruit sweep.
Legacy adapters without this metadata retain native unknown-length streaming
semantics (length 0, fixed temporal-pointer normalization).
The native frame index is local to that sweep; export preserves original source
frame indices. Existing annotations (including reviewed masks) are preserved.
No other model is substituted if loading or inference fails. Native-qualified
Efficient-SAM2 base-plus CPU video now works on normal startup. Capability still
comes from the native reply; unsupported modes are not enabled by the page.
`--experimental-efficient-video` is retained only for compatibility with older
servers. Native capability does not mark masks or fruit identities reviewed.

### Optional encoder prefix cache

Cache and experimental compute options are **launcher-only**. The browser cannot
choose a cache directory, change its budget, clear files, or pass arbitrary native
commands. Defaults remain cache off and restored-FP32 CPU compute.

For a one-entry RAM cache (approximately 49 MiB), add
`--efficient-prefix-cache` to the launch command. RAM entries disappear when the
native process stops. For persistent reuse across sessions, create an existing
private cache directory outside Git and use an explicit disk budget, for example
8192 MiB:

```bash
uv run --no-project python tools/annotator/tools/serve-browser.py \
  --serve build/examples/sam3_serve \
  --model /private/models/efficient-sam2-base-plus-convrot-w8a16.gguf \
  --image-root /private/sequence \
  --annotation-root /private/annotations \
  --prefix-cache-dir /private/cache/efficient-sam2 \
  --prefix-cache-budget-mib 8192 \
  --port 8765
```

All directories must already exist. Persistent caching also enables the bounded
RAM entry. Budgets must be integer 1..1048576 MiB; a budget without a directory,
symlinks, traversal, and cache directories inside Git worktrees are rejected.
The selected directory is rechecked at model startup. Native cache keys include
the actual image, model, preprocessing, runtime and build identity; no database
is needed. Cache content derives from the input images and must remain private.

The read-only cache status reports actual native RAM/disk hits, misses, invalid
entries, writes, evictions and resident/disk bytes. It refreshes after encoding
and propagation or with **キャッシュ統計を更新**. Unsupported older servers display
an unavailable/error message, never fabricated statistics. Cache mode and budget
are saved in per-mask provenance without the private cache path; live counters
are operational status and do not rewrite existing mask origins.

`--experimental-efficient-q8-0` is a separate, explicit research-only CLI opt-in.
It remains unqualified and is labelled experimental in the UI. Do not include it
in the standard validated launch command above.

Stop kills the current native process immediately rather than waiting for its
inference lock. Completed frames remain in the project; stale replies are ignored.
Restart the model and reseed to continue. Reset clears native tracking memory and
requires reloading a frame, while keeping saved annotations and IDs. Only current
frame canvases are retained, avoiding an RGBA cache for every mask in the video.

Additional dependency-free checks:

```bash
node tools/annotator/tools/run-tests.mjs
node tools/annotator/tools/test-project.mjs
node tools/annotator/tools/test-tracking.mjs
uv run --no-project python tools/annotator/tools/test-bridge.py
```

The synthetic tracking suite tests separate-object sweeps, cancellation/stale
replies, completed-frame retention, wrong IDs, missing/reappearing objects,
repeat-run preservation and native errors. Bridge tests cover deadlines even
when native stdout ends in a partial JSON line, cancellation/restart and security
boundaries. They do not claim private-data segmentation or identity quality.
Browser tests always request Chromium's sandbox; do not add `--no-sandbox` to
work around a restricted test environment.

### Prompt and edit provenance

Each candidate carries an immutable copy of the exact prompt and backend used by
its inference request. Editing the prompt invalidates old candidates, including
when auto-run is off. Prompt modes cannot change during inference. Saving a
candidate never pairs an old mask with a newer prompt.

Loading a different backend does not rewrite the saved project's model origin.
Per-mask provenance and model history retain the original model metadata; new
native masks record their actual backend. The bridge fingerprints the selected
model and native executable with SHA-256; the active native preprocessing mode
is displayed and retained in mask provenance, alongside native build identity,
runtime provenance and each propagation's sequence configuration. Manual annotation edits set the
project's raw-prediction flag to false, increment its intervention count and keep
an edit log; original per-mask origin records remain intact. Export/reload retains
this metadata. Undo refuses to restore an ID already occupied by a newer mask,
and the complete project is validated before every save or export.
