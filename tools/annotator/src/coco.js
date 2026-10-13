/** COCO RLE plus explicit video identity and review attributes. */
(function (global) {
  'use strict';
  function build(project) {
    const frames = project.frames || [{ width: project.width, height: project.height,
      fileName: project.fileName || 'image.jpg', key: project.fileName || 'image.jpg', instances: project.instances }];
    const categories = [], catId = new Map(), annotations = [];
    const images = frames.map((f, index) => {
      f.instances.forEach((inst) => {
        const name = inst.label || 'object';
        if (!catId.has(name)) { catId.set(name, catId.size + 1); categories.push({ id: catId.size, name, supercategory: '' }); }
        const fruitId = inst.fruitId || inst.id || annotations.length + 1;
        annotations.push({ id: annotations.length + 1, image_id: index + 1,
          category_id: catId.get(name), fruit_id: fruitId, track_id: fruitId, segment_id: inst.segmentId === undefined ? `${fruitId}:legacy-unverified` : inst.segmentId,
          segmentation: { size: [inst.maskHeight, inst.maskWidth], counts: inst.counts },
          area: inst.area, bbox: inst.bbox, iscrowd: 0,
          attributes: { score: inst.score, source: inst.source, provenance: inst.provenance || null,
            review_status: inst.reviewStatus || 'draft', identity_status: inst.identityStatus || 'unverified',
            visible: inst.visible !== false, prompt: inst.prompt || null } });
      });
      return { id: index + 1, file_name: f.fileName, frame_key: f.key,
        frame_index: index, source_frame_index: f.sourceFrameIndex ?? f.index ?? index, sequence_id: project.sequenceId || 'single-image',
        width: f.width, height: f.height, unresolved_fruit_ids: f.unresolvedFruitIds || [] };
    });
    return { info: { description: 'sam3.cpp video annotation export; predictions remain drafts until explicitly reviewed',
      version: '2.0', sequence_id: project.sequenceId || 'single-image', next_fruit_id: project.nextFruitId, model_history: project.modelHistory || [], raw_prediction: project.rawPrediction ?? project.model?.raw_prediction ?? false, manual_interventions: project.manualInterventions ?? project.model?.manual_interventions ?? 0, edit_history: project.editHistory || [], model: project.model || null,
      date_created: new Date().toISOString() }, licenses: [], images, annotations, categories };
  }
  global.COCO = { build };
})(window);
