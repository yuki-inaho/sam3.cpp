/**
 * The one place that knows whether we are inside Tauri.
 *
 * In a plain browser there is no sam3_serve and no filesystem, so SAM is
 * reported unavailable and the UI falls back to manual annotation rather
 * than throwing.
 */
(function (global) {
  'use strict';

  const tauri = global.__TAURI__ || null;
  const native = !!(tauri && tauri.core && tauri.core.invoke);
  // Outside Tauri these must reject, not throw synchronously on `null(...)`:
  // callers await them inside try/catch and would otherwise see a TypeError
  // instead of a message they can show the user.
  const invoke = native
    ? tauri.core.invoke
    : () => Promise.reject(new Error('この機能はデスクトップ版 (Tauri) でのみ使えます'));

  async function openImageDialog() {
    if (!native) return null;
    const selected = await tauri.dialog.open({
      multiple: false,
      filters: [{ name: 'Image', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'] }],
    });
    return typeof selected === 'string' ? selected : null;
  }

  async function openModelDialog() {
    if (!native) return null;
    const selected = await tauri.dialog.open({
      multiple: false,
      filters: [{ name: 'ggml model', extensions: ['ggml'] }],
    });
    return typeof selected === 'string' ? selected : null;
  }

  global.Bridge = {
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
