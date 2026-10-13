/** Portable, validated annotation projects. Fruit IDs belong to a sequence,
 * annotation IDs belong to an individual frame; neither is inferred by IoU. */
(function (global) {
  'use strict';
  const MAX_PIXELS = 64 * 1024 * 1024;
  const statuses = new Set(['draft', 'reviewed']);
  const identities = new Set(['unverified', 'confirmed', 'unresolved']);
  const integer = (n, min = 1) => Number.isSafeInteger(n) && n >= min;
  const fail = (s) => { throw new Error('Invalid annotation project: ' + s); };
  function newProject(sequenceId, frames = []) {
    return { schema: 'sam3-annotation-project', version: 1, sequenceId,
      nextFruitId: 1, model: null, modelHistory: [], frames: frames.map((f, i) => ({
        key: f.key || f.fileName, fileName: f.fileName, index: i, sourceFrameIndex: f.sourceFrameIndex ?? i,
        width: 0, height: 0, instances: [], unresolvedFruitIds: [],
      })) };
  }
  function serialise(project) {
    return { schema: 'sam3-annotation-project', version: 1,
      sequenceId: project.sequenceId, nextFruitId: project.nextFruitId,
      model: project.model || null, modelHistory: project.modelHistory || [], rawPrediction: project.rawPrediction ?? project.model?.raw_prediction ?? false, manualInterventions: project.manualInterventions ?? project.model?.manual_interventions ?? 0, editHistory: project.editHistory || [],
      frames: project.frames.map((frame) => ({
        key: frame.key, fileName: frame.fileName, index: frame.index, sourceFrameIndex: frame.sourceFrameIndex ?? frame.index,
        width: frame.width, height: frame.height,
        unresolvedFruitIds: [...(frame.unresolvedFruitIds || [])],
        instances: frame.instances.map((i) => ({
          id: i.id, fruitId: i.fruitId === undefined ? i.id : i.fruitId, segmentId: i.segmentId === undefined ? `${i.fruitId ?? i.id}:legacy-unverified` : i.segmentId, label: i.label,
          counts: [...i.counts], maskWidth: i.maskWidth, maskHeight: i.maskHeight,
          score: i.score, source: i.source, provenance: i.provenance || null, reviewStatus: i.reviewStatus === undefined ? 'draft' : i.reviewStatus,
          identityStatus: i.identityStatus === undefined ? 'unverified' : i.identityStatus,
          visible: i.visible !== false, colourIndex: i.colourIndex,
          area: i.area, bbox: [...i.bbox], prompt: i.prompt || null,
        })),
      })),
    };
  }
  function validate(value) {
    if (!value || value.schema !== 'sam3-annotation-project' || value.version !== 1) fail('unsupported schema/version');
    if (typeof value.sequenceId !== 'string' || !value.sequenceId || value.sequenceId.length > 512) fail('sequence ID');
    if (value.manualInterventions !== undefined && !integer(value.manualInterventions,0)) fail('manual intervention count');
    if (value.rawPrediction !== undefined && typeof value.rawPrediction !== 'boolean') fail('raw prediction flag');
    if (!Array.isArray(value.frames) || !value.frames.length || value.frames.length > 20000) fail('frame list');
    let maxId = 0;
    const keys = new Set(), sourceIndices = new Set();
    const frames = value.frames.map((f, index) => {
      if (!f || typeof f.key !== 'string' || !f.key || f.key.length > 1024 || keys.has(f.key)) fail('duplicate/invalid frame key');
      if (typeof f.fileName !== 'string' || !f.fileName || f.fileName.length > 1024) fail('frame filename');
      keys.add(f.key);
      const sourceIndex = f.sourceFrameIndex ?? f.index ?? index;
      if (sourceIndices.has(sourceIndex)) fail('duplicate source frame index');
      sourceIndices.add(sourceIndex);
      if ((f.sourceFrameIndex !== undefined && !integer(f.sourceFrameIndex, 0)) || (f.index !== undefined && !integer(f.index, 0))) fail('source frame index');
      if (!integer(f.width, 0) || !integer(f.height, 0) || f.width * f.height > MAX_PIXELS) fail('frame dimensions');
      if (!Array.isArray(f.instances) || f.instances.length > 10000) fail('instance list');
      const ids = new Set();
      const instances = f.instances.map((i) => {
        if (!i || !integer(i.fruitId) || i.fruitId > 2147483647 || ids.has(i.fruitId)) fail('duplicate/invalid fruit ID in frame');
        if (i.segmentId !== undefined && (typeof i.segmentId !== 'string' || !i.segmentId.trim() || i.segmentId.length > 255)) fail('segment ID');
        ids.add(i.fruitId); maxId = Math.max(maxId, i.fruitId);
        if (!integer(i.maskWidth) || !integer(i.maskHeight) || i.maskWidth * i.maskHeight > MAX_PIXELS || i.maskWidth !== f.width || i.maskHeight !== f.height) fail('mask/frame size mismatch');
        if (!Array.isArray(i.counts) || i.counts.length > 2 * i.maskWidth * i.maskHeight + 1) fail('RLE counts');
        const mask = global.RLE.decode(i.counts, i.maskWidth, i.maskHeight);
        if (!mask) fail('malformed RLE');
        if (!statuses.has(i.reviewStatus) || !identities.has(i.identityStatus)) fail('review/identity status');
        if (typeof i.label !== 'string' || i.label.length > 512 || !Number.isFinite(i.score)) fail('label/score');
        let prompt = null;
        if (i.prompt) {
          const point = (p, n) => Array.isArray(p) && p.length === n && p.every(Number.isFinite);
          if (!Array.isArray(i.prompt.pos) || !Array.isArray(i.prompt.neg) || i.prompt.pos.length + i.prompt.neg.length > 4096 || !i.prompt.pos.every(p => point(p, 2)) || !i.prompt.neg.every(p => point(p, 2)) || (i.prompt.box !== null && !point(i.prompt.box, 4))) fail('prompt');
          prompt = { pos: i.prompt.pos.map(p => [...p]), neg: i.prompt.neg.map(p => [...p]), box: i.prompt.box && [...i.prompt.box] };
        }
        return { id: i.fruitId, fruitId: i.fruitId, segmentId: i.segmentId === undefined ? `${i.fruitId}:legacy-unverified` : i.segmentId, label: i.label,
          counts: [...i.counts], maskWidth: i.maskWidth, maskHeight: i.maskHeight,
          provenance: i.provenance ? JSON.parse(JSON.stringify(i.provenance)) : value.model ? { model: JSON.parse(JSON.stringify(value.model)), inheritedFromProject: true } : null,
          score: i.score, source: typeof i.source === 'string' ? i.source.slice(0, 128) : 'import',
          reviewStatus: i.reviewStatus, identityStatus: i.identityStatus,
          visible: i.visible !== false, colourIndex: integer(i.colourIndex, 0) ? i.colourIndex : (i.fruitId - 1) % 8,
          area: global.RLE.area(mask), bbox: global.RLE.bbox(mask, i.maskWidth, i.maskHeight), prompt };
      });
      const unresolvedFruitIds = f.unresolvedFruitIds || [];
      if (!Array.isArray(unresolvedFruitIds) || unresolvedFruitIds.length > 10000 || !unresolvedFruitIds.every(n => integer(n) && n <= 2147483647)) fail('unresolved IDs');
      unresolvedFruitIds.forEach(n => { maxId = Math.max(maxId, n); });
      return { key: f.key, fileName: f.fileName, index, sourceFrameIndex: f.sourceFrameIndex ?? f.index ?? index, width: f.width, height: f.height, instances, unresolvedFruitIds: [...new Set(unresolvedFruitIds)] };
    });
    // Never recycle an ID after deletion or when a higher saved counter exists.
    if (value.nextFruitId !== undefined && (!integer(value.nextFruitId) || value.nextFruitId > 2147483648)) fail('next fruit ID');
    const nextFruitId = integer(value.nextFruitId) ? Math.max(maxId + 1, value.nextFruitId) : maxId + 1;
    return { schema: value.schema, version: 1, sequenceId: value.sequenceId, nextFruitId,
      rawPrediction: value.rawPrediction ?? value.model?.raw_prediction ?? false,
      manualInterventions: value.manualInterventions ?? value.model?.manual_interventions ?? 0,
      editHistory: Array.isArray(value.editHistory) ? JSON.parse(JSON.stringify(value.editHistory)) : [],
      modelHistory: Array.isArray(value.modelHistory) ? JSON.parse(JSON.stringify(value.modelHistory)) : [],
      model: value.model && typeof value.model === 'object' ? JSON.parse(JSON.stringify(value.model)) : null, frames };
  }
  function fromCoco(doc) {
    if (!doc || !Array.isArray(doc.images) || !Array.isArray(doc.annotations) || !Array.isArray(doc.categories)) fail('COCO structure');
    const imageIds = new Set(), categoryIds = new Set(), annotationIds = new Set();
    for (const i of doc.images) { if (!i || !integer(i.id) || imageIds.has(i.id)) fail('duplicate/invalid COCO image ID'); imageIds.add(i.id); }
    for (const c of doc.categories) { if (!c || !integer(c.id) || categoryIds.has(c.id) || typeof c.name !== 'string') fail('duplicate/invalid COCO category ID'); categoryIds.add(c.id); }
    for (const a of doc.annotations) {
      if (!a || !integer(a.id) || annotationIds.has(a.id)) fail('duplicate/invalid COCO annotation ID');
      annotationIds.add(a.id);
      if (!imageIds.has(a.image_id) || !categoryIds.has(a.category_id)) fail('unknown COCO image/category reference');
    }
    const categories = new Map(doc.categories.map(c => [c.id, c.name]));
    const project = newProject(doc.info && doc.info.sequence_id || 'imported-sequence', doc.images.map(i => ({ key: i.frame_key || i.file_name, fileName: i.file_name })));
    project.model = doc.info && doc.info.model || null;
    project.modelHistory = doc.info && doc.info.model_history || [];
    project.rawPrediction = doc.info?.raw_prediction ?? project.model?.raw_prediction ?? false;
    project.manualInterventions = doc.info?.manual_interventions ?? project.model?.manual_interventions ?? 0;
    project.editHistory = doc.info?.edit_history || [];
    if (doc.info && doc.info.next_fruit_id !== undefined) project.nextFruitId = doc.info.next_fruit_id;
    project.frames.forEach((frame, idx) => {
      const image = doc.images[idx]; frame.sourceFrameIndex = image.source_frame_index ?? image.frame_index ?? idx; frame.width = image.width; frame.height = image.height;
      frame.unresolvedFruitIds = image.unresolved_fruit_ids || [];
      frame.instances = doc.annotations.filter(a => a.image_id === image.id).map(a => {
        const attrs = a.attributes || {}, seg = a.segmentation || {};
        if (!Array.isArray(seg.size) || seg.size.length !== 2 || !Array.isArray(seg.counts)) fail('only uncompressed COCO RLE is supported');
        const fruitId = a.fruit_id ?? a.track_id ?? attrs.fruit_id;
        if ([a.fruit_id, a.track_id, attrs.fruit_id].some(id => id !== undefined && id !== fruitId)) fail('conflicting COCO fruit IDs');
        if (!integer(fruitId)) fail('COCO fruit_id required; identities cannot be guessed');
        const segmentId = a.segment_id === undefined ? attrs.segment_id : a.segment_id;
        if (a.segment_id !== undefined && attrs.segment_id !== undefined && a.segment_id !== attrs.segment_id) fail('conflicting COCO segment IDs');
        return { id: fruitId, fruitId, segmentId, label: categories.get(a.category_id) || 'object',
          maskWidth: seg.size[1], maskHeight: seg.size[0], counts: seg.counts,
          score: Number.isFinite(attrs.score) ? attrs.score : 0,
          provenance: attrs.provenance || null,
          source: attrs.source || 'coco-import', reviewStatus: attrs.review_status || 'draft',
          identityStatus: attrs.identity_status || 'unverified',
          visible: attrs.visible !== false, colourIndex: (fruitId - 1) % 8,
          bbox: a.bbox, area: a.area, prompt: attrs.prompt || null };
      });
    });
    return validate(project);
  }
  function restoreDeleted(instances, entry) {
    if (instances.some(i => i.fruitId === entry.inst.fruitId)) return false;
    instances.splice(Math.min(entry.index, instances.length), 0, entry.inst); return true;
  }
  function recordModel(project, model) {
    if (!model) return;
    if (!project.modelHistory) project.modelHistory = [];
    const encoded = JSON.stringify(model);
    if (!project.modelHistory.some(m => JSON.stringify(m) === encoded)) project.modelHistory.push(JSON.parse(encoded));
  }
  function recordManualEdit(project, action, frameKey, fruitId) {
    if (project.model?.raw_prediction === true) recordModel(project, project.model);
    const count = (project.manualInterventions ?? project.model?.manual_interventions ?? 0) + 1;
    project.rawPrediction = false; project.manualInterventions = count;
    if (project.model) project.model = {...project.model,raw_prediction:false,manual_interventions:count};
    if (!project.editHistory) project.editHistory = [];
    project.editHistory.push({action,frameKey,fruitId:fruitId??null,at:new Date().toISOString()});
  }
  function snapshotPrompt(pos, neg, box) { return {pos:pos.map(p=>[...p]),neg:neg.map(p=>[...p]),box:box&&[...box]}; }
  function candidateCurrent(candidate, revision) { return !!candidate && candidate.promptRevision === revision; }
  global.AnnotationProject = { newProject, serialise, validate, fromCoco, restoreDeleted, recordModel, recordManualEdit, snapshotPrompt, candidateCurrent };
})(window);
