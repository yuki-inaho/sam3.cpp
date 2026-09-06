/**
 * COCO uncompressed RLE.
 *
 * Runs are column-major and alternate, starting with background, which is
 * what sam3_serve emits and what COCO/CVAT consumers expect.  Keeping the
 * masks in this form all the way through means export is a copy, not a
 * re-encode.
 */
(function (global) {
  'use strict';

  // Returns null rather than a half-written bitmap when the runs do not sum to
  // width*height. Writing past the intended end would silently alias into
  // other pixels and the caller would export a plausible-looking wrong mask.
  function decode(counts, width, height) {
    const total = width * height;
    const out = new Uint8Array(total); // row-major, for ImageData
    let idx = 0;
    let value = 0;
    for (let i = 0; i < counts.length; i++) {
      const run = counts[i];
      if (!(run >= 0) || idx + run > total) return null;
      if (value) {
        for (let k = 0; k < run; k++) {
          const pos = idx + k;
          const x = (pos / height) | 0;
          const y = pos % height;
          out[y * width + x] = 1;
        }
      }
      idx += run;
      value ^= 1;
    }
    return idx === total ? out : null;
  }

  function encode(mask, width, height) {
    const counts = [];
    let value = 0;
    let run = 0;
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        const v = mask[y * width + x] ? 1 : 0;
        if (v === value) {
          run++;
        } else {
          counts.push(run);
          value = v;
          run = 1;
        }
      }
    }
    counts.push(run);
    return counts;
  }

  function area(mask) {
    let n = 0;
    for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
    return n;
  }

  // COCO bbox is [x, y, w, h] in pixels.
  function bbox(mask, width, height) {
    let x0 = width, y0 = height, x1 = -1, y1 = -1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y * width + x]) {
          if (x < x0) x0 = x;
          if (y < y0) y0 = y;
          if (x > x1) x1 = x;
          if (y > y1) y1 = y;
        }
      }
    }
    if (x1 < 0) return [0, 0, 0, 0];
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
  }

  global.RLE = { decode, encode, area, bbox };
})(window);
