// Dependency-free checks for the parts where a silent mistake would corrupt
// every exported mask: the RLE codec and the COCO document shape.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const sandbox = { window: {} };
vm.createContext(sandbox);
for (const f of ['src/rle.js', 'src/coco.js']) {
  vm.runInContext(readFileSync(new URL(`../${f}`, import.meta.url), 'utf8'), sandbox);
}
const { RLE, COCO } = sandbox.window;

let pass = 0, fail = 0;
const check = (name, cond, detail = '') => {
  if (cond) { console.log(`ok ${++pass + fail} - ${name}`); }
  else { fail++; console.log(`not ok ${pass + fail} - ${name}${detail ? ' # ' + detail : ''}`); }
};

// 1. Round-trip on a shape with a hole and a detached island — exactly the
//    cases a polygon export would lose.
{
  const w = 40, h = 30;
  const m = new Uint8Array(w * h);
  const put = (x, y, v = 1) => { m[y * w + x] = v; };
  for (let y = 5; y < 20; y++) for (let x = 5; x < 20; x++) put(x, y);      // block
  for (let y = 10; y < 15; y++) for (let x = 10; x < 15; x++) put(x, y, 0); // hole
  for (let y = 24; y < 28; y++) for (let x = 30; x < 36; x++) put(x, y);    // island

  const counts = RLE.encode(m, w, h);
  const back = RLE.decode(counts, w, h);
  check('RLE round-trip preserves hole and island', back.every((v, i) => v === m[i]));
  check('RLE counts sum to the pixel count', counts.reduce((a, b) => a + b, 0) === w * h);
  check('RLE starts with a background run', counts.length > 0 && m[0] === 0);
  check('area counts foreground only', RLE.area(m) === 15 * 15 - 25 + 24);
  check('bbox spans both components', JSON.stringify(RLE.bbox(m, w, h)) === JSON.stringify([5, 5, 31, 23]));
}

// 2. A recorded sam3_serve payload, vendored so this suite runs from a clean
//    checkout. Its mask was verified pixel-for-pixel against sam3_seg's PNG.
{
  const fx = JSON.parse(readFileSync(new URL('../fixtures/serve-point-315-250.json', import.meta.url), 'utf8'));
  const det = fx.reply.detections[0];
  const [w, h] = det.mask_size;
  const mask = RLE.decode(det.counts, w, h);
  check('server mask decodes to the verified pixel count', RLE.area(mask) === fx.expected.area, `got ${RLE.area(mask)}`);
  check('server mask bbox matches the reported box',
    JSON.stringify(RLE.bbox(mask, w, h)) === JSON.stringify(fx.expected.bbox_xywh),
    JSON.stringify(RLE.bbox(mask, w, h)));
  check('re-encoding the server mask reproduces its counts',
    JSON.stringify(RLE.encode(mask, w, h)) === JSON.stringify(det.counts));
}

// 2b. A malformed RLE must be rejected, not decoded into the wrong pixels.
{
  check('decode rejects runs that overshoot the grid', RLE.decode([0, 5, 100], 4, 4) === null);
  check('decode rejects runs that undershoot the grid', RLE.decode([0, 3], 4, 4) === null);
  check('decode rejects a negative run', RLE.decode([0, -1, 17], 4, 4) === null);
  check('decode accepts an exact cover', RLE.decode([0, 16], 4, 4) !== null);
}

// 3. COCO shape: size comes from the mask grid, and instances sharing a label
//    share a category.
{
  const doc = COCO.build({
    width: 640, height: 480, fileName: 'test.jpg',
    instances: [
      { label: 'cat', counts: [0, 5, 10], maskWidth: 640, maskHeight: 480, area: 5, bbox: [1, 2, 3, 4], score: 0.9, source: 'sam' },
      { label: 'cat', counts: [0, 7, 8], maskWidth: 640, maskHeight: 480, area: 7, bbox: [2, 3, 4, 5], score: 0.8, source: 'sam' },
      { label: 'dog', counts: [0, 2, 13], maskWidth: 640, maskHeight: 480, area: 2, bbox: [0, 0, 1, 1], score: 0.7, source: 'sam' },
    ],
  });
  check('COCO segmentation size is [height, width] of the MASK grid',
    JSON.stringify(doc.annotations[0].segmentation.size) === JSON.stringify([480, 640]));
  check('COCO dedupes categories by label', doc.categories.length === 2);
  check('COCO reuses the category id for the same label',
    doc.annotations[0].category_id === doc.annotations[1].category_id &&
    doc.annotations[2].category_id !== doc.annotations[0].category_id);
  check('COCO annotation ids are 1-based and unique',
    JSON.stringify(doc.annotations.map((a) => a.id)) === JSON.stringify([1, 2, 3]));
}

console.log(`1..${pass + fail}`);
if (fail) { console.error(`\n${fail} check(s) failed`); process.exit(1); }
console.log(`\nall ${pass} checks passed`);
