/**
 * Native adapters for Tauri and the explicitly allowlisted loopback bridge.
 * A plain static page has no inference backend.
 */
(function (global) {
  'use strict';

  const tauri = global.__TAURI__ || null;
  const loopback = global.__SAM3_LOOPBACK__ || null;
  const native = !!(loopback || (tauri && tauri.core && tauri.core.invoke));
  // Outside Tauri these must reject, not throw synchronously on `null(...)`:
  // callers await them inside try/catch and would otherwise see a TypeError
  // instead of a message they can show the user.
  const invoke = loopback ? async (command, args = {}) => {
    const response = await fetch('/api/invoke', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command, args }),
    });
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || 'CPU bridge failed');
    return result.result;
  } : native ? tauri.core.invoke
    : () => Promise.reject(new Error('この機能は Tauri または専用ローカルサーバーで使えます'));

  if (loopback) {
    const addSelect = (id, items, before) => {
      const select = document.createElement('select');
      select.id = id;
      select.style.maxWidth = '100%';
      items.forEach((path) => {
        const option = document.createElement('option');
        option.value = path;
        option.textContent = path.split('/').slice(-2).join('/');
        select.appendChild(option);
      });
      before.parentNode.insertBefore(select, before);
      return select;
    };
    const models = addSelect('browserModels', loopback.models, document.getElementById('startModel'));
    addSelect('browserImages', loopback.images, document.getElementById('openImage'));
    const modelPath = document.getElementById('modelPath');
    modelPath.value = models.value;
    models.onchange = () => { modelPath.value = models.value; };
    document.getElementById('cpu').checked = true;
    document.getElementById('cpu').disabled = true;
  }

  async function openImageDialog() {
    if (!native) return null;
    if (loopback) return document.getElementById('browserImages').value;
    const selected = await tauri.dialog.open({
      multiple: false,
      filters: [{ name: 'Image', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'] }],
    });
    return typeof selected === 'string' ? selected : null;
  }

  async function openModelDialog() {
    if (!native) return null;
    if (loopback) return document.getElementById('browserModels').value;
    const selected = await tauri.dialog.open({
      multiple: false,
      filters: [{ name: 'SAM model', extensions: ['ggml', 'gguf'] }],
    });
    return typeof selected === 'string' ? selected : null;
  }

  const knownSequences = loopback ? loopback.sequences : [];
  function sequenceFor(path) {
    const known = knownSequences.find(s => s.frames.some(f => f.path === path));
    if (known) return known;
    const name = path.split(/[\\/]/).pop();
    const sequence = { id: name, frames: [{ path, key: name, fileName: name }] };
    knownSequences.push(sequence); return sequence;
  }
  function matchSequence(project) {
    const exact = knownSequences.find(s => s.id === project.sequenceId && s.frames.length === project.frames.length && s.frames.every((f, i) => f.key === project.frames[i].key));
    if (exact) return exact;
    const candidates = knownSequences.filter(s => s.frames.length === project.frames.length && s.frames.every((f, i) => f.key === project.frames[i].key));
    return candidates.length === 1 ? candidates[0] : null;
  }
  global.Bridge = {
    sequenceFor, matchSequence, trackingAvailable: !!loopback, canCancel: !!loopback,
    loadSavedProject: loopback && loopback.canSave ? () => invoke('load_saved_project') : null,
    saveProject: loopback && loopback.canSave ? (project) => invoke('save_project', { project }) : null,
    samTrackReset: () => invoke('sam_track_reset'),
    samTrackConfig: loopback ? (sequenceLength, sourceStartFrame) => invoke('sam_track_config', { sequenceLength, sourceStartFrame }) : null,
    samTrackAdd: (objectId, prompt) => invoke('sam_track_add', { objectId, ...prompt }),
    samTrackFrame: (path) => invoke('sam_track_frame', { path }),
    samCacheStats: loopback ? () => invoke('sam_cache_stats') : null,
    native,
    openImageDialog,
    openModelDialog,
    readImage: (path) => invoke('read_image', { path }),
    samStart: (opts) => invoke('sam_start', opts),
    samStop: () => invoke('sam_stop'),
    samRunning: () => (native ? invoke('sam_running') : Promise.resolve(false)),
    samLoadImage: (path) => invoke('sam_load_image', { path }),
    samSegment: (pos, neg, bbox, multimask) =>
      invoke('sam_segment', { pos, neg, bbox, multimask }),
  };
})(window);
