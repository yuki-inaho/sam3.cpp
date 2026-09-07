# sam3 Annotator

SAM-assisted image annotation on top of `sam3.cpp`. Click a point, get a mask,
accept it, export COCO.

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
4. **確定** (`Enter`) turns the current mask into an instance; name it in the
   label box first if you want something other than `object`.
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
node tools/run-tests.mjs                       # 16 checks, no browser
PLAYWRIGHT_PATH=<path-to-playwright> \
  node tools/browser-test.mjs                  # 39 checks, headless Chromium
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
- One image at a time; there is no project save/reload. Loading another image
  discards the current instances — it asks first, and leaving the page warns,
  but there is no recovery once confirmed.
- Undo covers prompt points and instance deletion, not label edits.
- Every accepted instance keeps a full-image RGBA layer, so memory grows with
  the instance count on very large images.
- Opened in a browser instead of Tauri, the SAM controls are disabled — there
  is no model and no filesystem there.
- Only tested on Linux.
