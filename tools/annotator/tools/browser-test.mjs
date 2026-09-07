/**
 * Headless UI test.
 *
 * app.js only ever runs inside a webview, so nothing here was exercised by the
 * node tests. This drives the real page in Chromium with window.__TAURI__
 * replaced by a stub that replays a recorded sam3_serve payload, which covers
 * the click -> segment -> accept -> export path without needing a model.
 */
import { createRequire } from 'node:module';
// playwright ships CommonJS; resolve it from wherever it is installed rather
// than pinning a pnpm path.
const require_ = createRequire(import.meta.url);
const { chromium } = require_(process.env.PLAYWRIGHT_PATH || 'playwright');
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const srcDir = join(here, '..', 'src');
const repoRoot = join(here, '..', '..', '..');

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const server = createServer((req, res) => {
  const name = req.url === '/' ? '/index.html' : req.url.split('?')[0];
  try {
    const body = readFileSync(join(srcDir, name));
    res.writeHead(200, { 'Content-Type': MIME[extname(name)] || 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/`;

// The same vendored payload the node tests pin, so this runs from a clean
// checkout with no model and no /tmp state.
const fixture = JSON.parse(readFileSync(join(here, '..', 'fixtures', 'serve-point-315-250.json'), 'utf8'));
const recorded = fixture.reply;
const imageDataUrl =
  'data:image/jpeg;base64,' + readFileSync(join(repoRoot, 'data', 'test_image.jpg')).toString('base64');

let pass = 0, fail = 0;
const check = (name, cond, detail = '') => {
  if (cond) console.log(`ok ${++pass + fail} - ${name}`);
  else { fail++; console.log(`not ok ${pass + fail} - ${name}${detail ? ' # ' + detail : ''}`); }
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });

const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));

// Stub the native side before any of the app's scripts run.
await page.addInitScript(({ det, dataUrl }) => {
  window.__INVOKES__ = [];
  window.__TAURI__ = {
    core: {
      invoke: async (cmd, argsObj) => {
        window.__INVOKES__.push([cmd, argsObj]);
        switch (cmd) {
          case 'read_image':
            return { data_url: dataUrl, width: 640, height: 480 };
          case 'sam_start':
            return { ok: true, event: 'ready', model_type: 3 };
          case 'sam_running':
            return true;
          case 'sam_load_image':
            return { ok: true, event: 'loaded', width: 640, height: 480 };
          case 'sam_segment':
            return det;
          case 'sam_stop':
            return null;
          default:
            throw new Error('unexpected command: ' + cmd);
        }
      },
    },
    dialog: { open: async () => '/repo/data/test_image.jpg' },
  };
}, { det: recorded, dataUrl: imageDataUrl });
await page.addInitScript((det) => { window.__FIXTURE__ = det; }, recorded);

await page.goto(base);
await page.waitForFunction(() => document.getElementById('status').textContent !== '起動中…');

check('native mode is detected', !(await page.evaluate(() => document.body.classList.contains('browser'))));
// Controls now reflect what is actually possible: the entry points are live,
// but 実行 stays disabled until a model, an image and a prompt all exist.
check('the entry-point controls are live under Tauri',
  !(await page.locator('#startModel').isDisabled()) &&
  !(await page.locator('#openImage').isDisabled()));
check('実行 is disabled before a model and a prompt exist',
  await page.locator('#segment').isDisabled());

// --- model ---
await page.click('#startModel');
await page.waitForFunction(() => document.getElementById('status').className.includes('ok'), null, { timeout: 15000 });
check('model start reaches a ready state', true);
check('sam_start received the JS-side argument names',
  await page.evaluate(() => {
    const call = window.__INVOKES__.find((c) => c[0] === 'sam_start');
    return !!call && 'modelPath' in call[1] && 'cpu' in call[1] && 'nThreads' in call[1];
  }));

// --- image ---
await page.click('#openImage');
await page.waitForFunction(() => document.getElementById('imageName').textContent.includes('test_image'), null, { timeout: 15000 });
check('image name is shown', (await page.textContent('#imageName')).includes('test_image.jpg'));
check('sam_load_image was called after the image loaded',
  await page.evaluate(() => window.__INVOKES__.some((c) => c[0] === 'sam_load_image')));

// --- click to segment ---
const boxEl = await page.locator('#canvas').boundingBox();
await page.mouse.click(boxEl.x + boxEl.width / 2, boxEl.y + boxEl.height / 2);
await page.waitForFunction(() => !document.getElementById('accept').disabled, null, { timeout: 15000 });
check('a click produces a candidate and enables 確定', true);
check('the candidate chip shows the recorded score',
  (await page.textContent('#candidates')).includes('0.511'),
  await page.textContent('#candidates'));
check('sam_segment was sent one positive point',
  await page.evaluate(() => {
    const call = [...window.__INVOKES__].reverse().find((c) => c[0] === 'sam_segment');
    return !!call && call[1].pos.length === 1 && call[1].neg.length === 0;
  }));

// --- shift-click adds a negative point and re-sends BOTH ---
await page.keyboard.down('Shift');
await page.mouse.click(boxEl.x + boxEl.width / 2 + 40, boxEl.y + boxEl.height / 2 + 30);
await page.keyboard.up('Shift');
await page.waitForTimeout(400);
check('shift-click re-sends the whole prompt, not just the new point',
  await page.evaluate(() => {
    const call = [...window.__INVOKES__].reverse().find((c) => c[0] === 'sam_segment');
    return !!call && call[1].pos.length === 1 && call[1].neg.length === 1;
  }),
  JSON.stringify(await page.evaluate(() => {
    const c = [...window.__INVOKES__].reverse().find((x) => x[0] === 'sam_segment');
    return c && { pos: c[1].pos.length, neg: c[1].neg.length };
  })));

// --- accept ---
await page.fill('#label', 'cat');
await page.click('#accept');
await page.waitForFunction(() => document.getElementById('count').textContent === '1', null, { timeout: 10000 });
check('accepting creates one instance', (await page.textContent('#count')) === '1');
check('the instance carries the typed label',
  (await page.locator('#instances .name').first().inputValue()) === 'cat');
check('the instance area matches the verified pixel count',
  (await page.textContent('#instances .meta')).includes(String(fixture.expected.area)),
  await page.textContent('#instances .meta'));
check('the prompt is cleared after accepting',
  await page.locator('#accept').isDisabled());

// --- export ---
const [download] = await Promise.all([
  page.waitForEvent('download', { timeout: 10000 }),
  page.click('#export'),
]);
const stream = await download.createReadStream();
let raw = '';
for await (const chunk of stream) raw += chunk;
const coco = JSON.parse(raw);
check('export produces one annotation', coco.annotations.length === 1);
check('COCO size is [height, width]',
  JSON.stringify(coco.annotations[0].segmentation.size) === JSON.stringify([480, 640]),
  JSON.stringify(coco.annotations[0].segmentation.size));
check('COCO bbox is [x, y, w, h] in pixels',
  JSON.stringify(coco.annotations[0].bbox) === JSON.stringify([91, 217, 257, 161]),
  JSON.stringify(coco.annotations[0].bbox));
check('the category comes from the typed label', coco.categories[0].name === 'cat');
check('image metadata matches the loaded image',
  coco.images[0].width === 640 && coco.images[0].height === 480);

// --- second instance keeps its own colour and deletion re-indexes ---
await page.mouse.click(boxEl.x + boxEl.width / 2 - 60, boxEl.y + boxEl.height / 2 - 40);
await page.waitForFunction(() => !document.getElementById('accept').disabled, null, { timeout: 15000 });
await page.click('#accept');
await page.waitForFunction(() => document.getElementById('count').textContent === '2', null, { timeout: 10000 });
const swatchesBefore = await page.locator('#instances .swatch').evaluateAll((els) => els.map((e) => e.style.background));
check('two instances get different colours', swatchesBefore[0] !== swatchesBefore[1], swatchesBefore.join(' / '));

await page.locator('#instances .icon.danger').first().click();
await page.waitForFunction(() => document.getElementById('count').textContent === '1', null, { timeout: 10000 });
const swatchAfter = await page.locator('#instances .swatch').evaluateAll((els) => els.map((e) => e.style.background));
// The mask layer is baked once, at accept time. If the swatch were derived
// from the array index it would change under the survivor after a delete and
// disagree with the mask still painted on the canvas.
check('the survivor keeps the colour its mask was baked with',
  swatchAfter[0] === swatchesBefore[1], `${swatchAfter[0]} vs ${swatchesBefore[1]}`);

// A third instance must not reuse a colour still on screen.
await page.mouse.click(boxEl.x + boxEl.width / 2 + 90, boxEl.y + boxEl.height / 2 + 60);
await page.waitForFunction(() => !document.getElementById('accept').disabled, null, { timeout: 15000 });
await page.click('#accept');
await page.waitForFunction(() => document.getElementById('count').textContent === '2', null, { timeout: 10000 });
const swatchesNow = await page.locator('#instances .swatch').evaluateAll((els) => els.map((e) => e.style.background));
check('a new instance does not reuse a colour already in use',
  new Set(swatchesNow).size === swatchesNow.length, swatchesNow.join(' / '));

// --- regressions for the review findings ---

// A malformed RLE must be refused, not drawn at the wrong offset.
// bridge.js captures `invoke` at load time, so stub the Bridge method that
// app.js actually calls rather than the underlying Tauri function.
await page.evaluate(() => {
  window.__ORIG_SEGMENT__ = window.Bridge.samSegment;
  window.Bridge.samSegment = async () => ({
    ok: true, event: 'segmented', width: 640, height: 480,
    detections: [{ score: 0.9, iou: 0.9, box: [0, 0, 10, 10],
                   mask_size: [640, 480], counts: [0, 5] }],  // runs do not cover the grid
  });
});
await page.mouse.click(boxEl.x + boxEl.width / 2 + 20, boxEl.y + boxEl.height / 2 + 20);
await page.waitForTimeout(500);
check('a malformed mask is rejected instead of rendered',
  await page.locator('#accept').isDisabled() &&
  (await page.textContent('#status')).includes('復号'),
  await page.textContent('#status'));

// A slow reply must not let a second request overlap on the single stream.
await page.evaluate(() => {
  const det = window.__FIXTURE__;
  window.__SEGMENT_CALLS__ = 0;
  window.Bridge.samSegment = async () => {
    window.__SEGMENT_CALLS__++;
    await new Promise((r) => setTimeout(r, 600));
    return det;
  };
});
await page.mouse.click(boxEl.x + boxEl.width / 2, boxEl.y + boxEl.height / 2);
await page.waitForTimeout(60);
await page.mouse.click(boxEl.x + boxEl.width / 2 + 10, boxEl.y + boxEl.height / 2);
await page.keyboard.press('Space');
await page.waitForTimeout(900);
check('an in-flight segmentation blocks a second request',
  (await page.evaluate(() => window.__SEGMENT_CALLS__)) === 1,
  'calls=' + (await page.evaluate(() => window.__SEGMENT_CALLS__)));

// The ↵ badge sits beside the label field, so Enter must work while typing there.
await page.waitForFunction(() => !document.getElementById('accept').disabled, null, { timeout: 10000 });
const before = await page.textContent('#count');
await page.click('#label');
await page.fill('#label', 'dog');
await page.keyboard.press('Enter');
await page.waitForTimeout(300);
check('Enter in the label field accepts, as its badge claims',
  (await page.textContent('#count')) !== before,
  `${before} -> ${await page.textContent('#count')}`);
check('the label typed at accept time is the one kept',
  (await page.locator('#instances .name').last().inputValue()) === 'dog');

// --- view: pan, and zoom that survives a resize ---
await page.evaluate(() => { window.__VIEW__ = null; });
const readView = () => page.evaluate(() => {
  // derive the view from where a known image point lands on screen
  const c = document.getElementById('canvas');
  return { w: c.width, h: c.height };
});
await page.mouse.move(boxEl.x + 300, boxEl.y + 300);
await page.mouse.wheel(0, -240);           // zoom in
await page.waitForTimeout(120);
const zoomed = await page.screenshot({ clip: { x: boxEl.x + 200, y: boxEl.y + 200, width: 120, height: 120 } });
await page.setViewportSize({ width: 1180, height: 780 });
await page.waitForTimeout(200);
await page.setViewportSize({ width: 1280, height: 820 });
await page.waitForTimeout(200);
const afterResize = await page.screenshot({ clip: { x: boxEl.x + 200, y: boxEl.y + 200, width: 120, height: 120 } });
check('a window resize does not reset the zoom',
  Buffer.compare(zoomed, afterResize) === 0);

// Alt+drag pans from any mode.
const beforePan = await page.screenshot({ clip: { x: boxEl.x + 200, y: boxEl.y + 200, width: 120, height: 120 } });
await page.keyboard.down('Alt');
await page.mouse.move(boxEl.x + 400, boxEl.y + 400);
await page.mouse.down();
await page.mouse.move(boxEl.x + 460, boxEl.y + 430, { steps: 4 });
await page.mouse.up();
await page.keyboard.up('Alt');
await page.waitForTimeout(150);
const afterPan = await page.screenshot({ clip: { x: boxEl.x + 200, y: boxEl.y + 200, width: 120, height: 120 } });
check('Alt+drag pans the image', Buffer.compare(beforePan, afterPan) !== 0);

await page.click('#fit');
await page.waitForTimeout(120);

// --- undo: prompt points, then instance deletion ---
await page.evaluate(() => { window.Bridge.samSegment = window.__ORIG_SEGMENT__; });
const instBefore = await page.textContent('#count');
await page.locator('#instances .icon.danger').first().click();
await page.waitForTimeout(150);
const instAfterDelete = await page.textContent('#count');
await page.keyboard.press('Control+z');
await page.waitForTimeout(200);
check('Ctrl+Z restores a deleted instance',
  Number(instAfterDelete) === Number(instBefore) - 1 &&
  (await page.textContent('#count')) === instBefore,
  `${instBefore} -> ${instAfterDelete} -> ${await page.textContent('#count')}`);

// A prompt point is undone before any instance is.
await page.click('[data-mode="point"]');
await page.evaluate(() => { document.getElementById('auto').checked = false; });
await page.mouse.click(boxEl.x + boxEl.width / 2, boxEl.y + boxEl.height / 2);
await page.waitForTimeout(100);
check('a queued prompt enables 実行', !(await page.locator('#segment').isDisabled()));
await page.keyboard.press('Control+z');
await page.waitForTimeout(150);
check('Ctrl+Z removes the last prompt point first',
  await page.locator('#segment').isDisabled());

// --- selecting an instance links the list to the canvas ---
await page.locator('#instances .meta').first().click();
await page.waitForTimeout(120);
check('clicking a row selects it', await page.evaluate(() =>
  !!document.querySelector('#instances li.sel')));

// --- discarding work asks first ---
let asked = null;
page.on('dialog', async (d) => { asked = d.message(); await d.dismiss(); });
await page.click('#openImage');
await page.waitForTimeout(300);
check('opening another image asks before discarding instances',
  asked !== null && asked.includes('破棄'), String(asked));
check('dismissing the prompt keeps the current image',
  (await page.textContent('#imageName')).includes('test_image'));

// --- the panel stays reachable no matter how many instances exist ---
await page.evaluate(() => {
  const list = document.getElementById('instances');
  for (let i = 0; i < 40; i++) list.appendChild(list.firstElementChild.cloneNode(true));
});
await page.waitForTimeout(120);
const visible = async (sel) => {
  const b = await page.locator(sel).boundingBox();
  const vp = page.viewportSize();
  return !!b && b.y >= 0 && b.y + b.height <= vp.height && b.x >= 0;
};
// At this window height the fixed sections plus a usable list do not both
// fit, so the setup controls scroll rather than being clipped away. What
// matters is that they stay reachable.
check('the model path stays reachable with 40+ instances',
  await (async () => {
    await page.locator('#modelPath').scrollIntoViewIfNeeded();
    return visible('#modelPath');
  })());
// Pinned to the bottom of the panel: visible even after scrolling the panel
// to the top for the check above.
check('the export button stays visible with 40+ instances', await visible('#export'));
check('the instance list scrolls inside itself rather than growing the panel',
  await page.evaluate(() => {
    const l = document.getElementById('instances');
    // It absorbs the spare space, keeps a usable height, and scrolls its own
    // overflow instead of pushing the export button off the panel.
    return l.scrollHeight > l.clientHeight && l.clientHeight >= 100 &&
           getComputedStyle(l).overflowY === 'auto';
  }),
  await page.evaluate(() => {
    const l = document.getElementById('instances');
    return `client=${l.clientHeight} scroll=${l.scrollHeight}`;
  }));

check('no uncaught page errors', consoleErrors.length === 0, consoleErrors.join(' | '));

await page.click('#fit');
await page.waitForTimeout(150);
await page.screenshot({ path: join(here, '..', 'evidence', 'browser-final.png'), fullPage: false });

await browser.close();
server.close();

console.log(`1..${pass + fail}`);
if (fail) { console.error(`\n${fail} check(s) failed`); process.exit(1); }
console.log(`\nall ${pass} browser checks passed`);
