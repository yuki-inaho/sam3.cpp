/**
 * COCO export.
 *
 * Masks go out as uncompressed RLE in `segmentation`, holes, detached pieces
 * and overlaps between instances all preserved — a per-instance bitmap has no
 * way to lose them, unlike a polygon approximation.
 */
(function (global) {
  'use strict';

  function build(project) {
    const categories = [];
    const catId = new Map();
    project.instances.forEach((inst) => {
      const name = inst.label || 'object';
      if (!catId.has(name)) {
        catId.set(name, catId.size + 1);
        categories.push({ id: catId.size, name, supercategory: '' });
      }
    });

    const images = [{
      id: 1,
      file_name: project.fileName || 'image.jpg',
      width: project.width,
      height: project.height,
    }];

    // size must describe the grid `counts` was produced on, which is the
    // server's mask_size — not the browser-decoded image dimensions. They are
    // normally equal, but taking the wrong one would emit an RLE whose runs
    // sum to a different pixel count than its declared size.
    const annotations = project.instances.map((inst, i) => ({
      id: i + 1,
      image_id: 1,
      category_id: catId.get(inst.label || 'object'),
      segmentation: {
        size: [inst.maskHeight, inst.maskWidth], // COCO order is [h, w]
        counts: inst.counts,
      },
      area: inst.area,
      bbox: inst.bbox,
      iscrowd: 0,
      attributes: { score: inst.score, source: inst.source },
    }));

    return {
      info: {
        description: 'sam3.cpp annotator export',
        version: '1.0',
        date_created: new Date().toISOString(),
      },
      licenses: [],
      images,
      annotations,
      categories,
    };
  }

  global.COCO = { build };
})(window);
