/**
 * sam3 annotator — click to segment, review, accept, export COCO.
 *
 * The prompt is always re-sent whole. SAM conditions on the full point set,
 * so adding a negative point must re-run with every earlier point still
 * attached; sending only the new one would segment something unrelated.
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const PALETTE = [
    [239, 83, 80], [66, 165, 245], [102, 187, 106], [255, 167, 38],
    [171, 71, 188], [38, 198, 218], [255, 238, 88], [141, 110, 99],
  ];

  // One boolean cannot distinguish "no model" from "model but no encoded
  // image" from "the server died", yet each needs a different message and a
  // different set of enabled controls.
  const SERVER = { NONE: 'none', READY: 'ready', ENCODED: 'encoded', DEAD: 'dead' };

  const state = {
    project: null, frameIndex: -1, sequence: null, capabilities: {}, activeBackend: null, dirty: false, revision: 0, saving: false, operation: 0,
    imagePath: null,
    fileName: null,
    img: null,
    width: 0,
    height: 0,
    server: SERVER.NONE,
    busy: false,
    mode: 'point',
    pos: [],
    neg: [],
    box: null,
    promptRevision: 0,
    promptLog: [],          // ordered record of prompt edits, for undo
    dragStart: null,
    dragNow: null,
    panFrom: null,
    candidates: [],
    candidateIndex: 0,
    instances: [],
    selectedId: null,
    deleted: [],            // undo stack for instance deletion
    view: { scale: 1, x: 0, y: 0 },
    viewPlaced: false,
    dpr: 1,
  };

  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');

  /* ── status ─────────────────────────────────────────────────────────── */

  function setStatus(text, kind) {
    const el = $('status');
    el.textContent = text;
    el.className = 'status' + (kind ? ' ' + kind : '');
    el.title = text;   // the header is narrow; the full text stays reachable
  }

  // Anything that talks to the native side is disabled while a request is in
  // flight, so the only way to overlap requests is gone rather than merely
  // discouraged.

  function syncControls() {
    const native = window.Bridge.native;
    const busy = state.busy;
    const hasImage = !!state.img;
    const encoded = state.server === SERVER.ENCODED;

    document.querySelectorAll('[data-mode]').forEach(el => { el.disabled = busy; });
    document.querySelectorAll('#instances input, #instances button').forEach(el => { el.disabled = busy; });
    $('startModel').disabled = busy || !native;
    $('refreshCacheStats').disabled = busy || !window.Bridge.samCacheStats || state.server === SERVER.NONE || state.server === SERVER.DEAD;
    $('browseModel').disabled = busy || !native;
    $('openImage').disabled = busy || !native;
    $('segment').disabled = busy || !native || !encoded || !hasPrompt();
    $('accept').disabled = busy || !state.candidates.length;
    $('clear').disabled = busy || (!hasPrompt() && !state.candidates.length);
    $('undo').disabled = busy || (!state.promptLog.length && !state.deleted.length);
    $('export').disabled = busy || state.instances.length === 0;
    $('fit').disabled = !hasImage;
    $('previousFrame').disabled = busy || state.frameIndex <= 0;
    $('nextFrame').disabled = busy || !state.sequence || state.frameIndex >= state.sequence.frames.length - 1;
    $('propagate').disabled = !window.Bridge.trackingAvailable || busy || !state.sequence || state.frameIndex >= state.sequence.frames.length - 1 || !encoded || !state.capabilities.video_tracking || !state.project || !state.instances.some(i => i.prompt);
    $('cancel').disabled = !busy || !window.Bridge.canCancel;
    $('resetTracking').disabled = !window.Bridge.trackingAvailable || busy || !native || state.server === SERVER.NONE || state.server === SERVER.DEAD;
    $('saveProject').disabled = busy || state.saving || !state.project;
    $('importProject').disabled = busy;
    $('restoreSaved').disabled = busy || !window.Bridge.loadSavedProject;
    $('frameSelect').disabled = busy || !state.sequence;
    $('fruitId').disabled = busy;
    $('export').disabled = busy || !(state.project && state.project.frames.some(f => f.instances.length));
    $('unresolvedStatus').textContent = state.project && state.frameIndex >= 0 && state.project.frames[state.frameIndex].unresolvedFruitIds.length ? '見失い・未解決: Fruit ' + state.project.frames[state.frameIndex].unresolvedFruitIds.join(', ') : '';
    if (state.project) $('sequenceStatus').textContent = `${state.project.sequenceId} | frame ${state.frameIndex + 1}/${state.project.frames.length} | ${state.project.frames.reduce((n,f) => n + f.instances.length, 0)} masks${state.dirty ? ' • 未保存' : ''}`;

  }

  function hasPrompt() {
    return state.pos.length > 0 || state.neg.length > 0 || !!state.box;
  }

  function setBusy(on, label) {
    state.busy = on;
    document.body.classList.toggle('busy', on);
    if (on && label) setStatus(label, 'work');
    syncControls();
  }

  function serverDied(err) {
    state.server = SERVER.DEAD;
    $('cacheStatus').textContent = 'キャッシュ統計: モデル停止';
    setStatus('セグメンテーションサーバが停止しました: ' + err + ' — モデルを開始し直してください', 'err');
    $('startModel').textContent = 'モデル開始';
    syncControls();
  }

  async function updateCacheStatus(operation) {
    if (!window.Bridge.samCacheStats) { $('cacheStatus').textContent = 'キャッシュ統計: この接続では未対応'; return; }
    try {
      const stats = await window.Bridge.samCacheStats();
      if (operation !== state.operation) return;
      const mib = bytes => (bytes / (1024 * 1024)).toFixed(1);
      $('cacheStatus').textContent = stats.enabled
        ? `キャッシュ有効 | RAM hit ${stats.ram_hits}, disk hit ${stats.disk_hits}, miss ${stats.misses} | RAM ${mib(stats.resident_bytes)} MiB / disk ${mib(stats.disk_bytes)} MiB | write ${stats.writes}, eviction ${stats.evictions}, invalid ${stats.invalid_entries}`
        : 'キャッシュ無効 (native 応答)。設定は起動時の CLI のみです。';
    } catch (error) {
      if (operation === state.operation) $('cacheStatus').textContent = 'キャッシュ統計を取得できません: ' + error;
    }
  }

  async function refreshCacheStats() {
    if (state.busy || !window.Bridge.samCacheStats || state.server === SERVER.NONE || state.server === SERVER.DEAD) return;
    const operation = ++state.operation; setBusy(true);
    try { await updateCacheStatus(operation); }
    finally { if (operation === state.operation) setBusy(false); }
  }

  /* ── mask rendering ─────────────────────────────────────────────────── */

  // Rasterise a mask once into its own canvas; redraw then costs one drawImage
  // instead of touching width*height pixels every frame.
  function rasterise(counts, w, h, rgb, alpha) {
    const mask = window.RLE.decode(counts, w, h);
    if (!mask) return null;   // runs did not sum to w*h
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const octx = off.getContext('2d');
    const img = octx.createImageData(w, h);
    const d = img.data;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) {
        d[i * 4] = rgb[0];
        d[i * 4 + 1] = rgb[1];
        d[i * 4 + 2] = rgb[2];
        d[i * 4 + 3] = alpha;
      }
    }
    octx.putImageData(img, 0, 0);
    return { canvas: off, mask };
  }

  // Colours are assigned once, at accept time, and stored on the instance.
  // Deriving them from the array index instead would repaint every survivor
  // after a delete while their mask layers keep the colour they were baked
  // with, so the swatch and the mask on screen would disagree.
  function colourFor(index) {
    return PALETTE[index % PALETTE.length];
  }

  function nextColourIndex() {
    const used = new Set(state.instances.map((i) => i.colourIndex));
    for (let i = 0; i < PALETTE.length; i++) if (!used.has(i)) return i;
    return state.instances.length % PALETTE.length;
  }

  /* ── view ───────────────────────────────────────────────────────────── */

  function fitView() {
    if (!state.img) return;
    const pad = 24;
    const availW = canvas.clientWidth - pad * 2;
    const availH = canvas.clientHeight - pad * 2;
    const scale = Math.min(availW / state.width, availH / state.height, 4);
    state.view.scale = scale > 0 ? scale : 1;
    state.view.x = (canvas.clientWidth - state.width * state.view.scale) / 2;
    state.view.y = (canvas.clientHeight - state.height * state.view.scale) / 2;
    state.viewPlaced = true;
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.dpr = dpr;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }

  // devicePixelRatio can change without a resize event — dragging the window
  // to a different-density display fires only this.
  function watchDpr() {
    if (!window.matchMedia) return;
    const arm = () => {
      const mq = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
      const once = () => { resizeCanvas(); draw(); arm(); };
      if (mq.addEventListener) mq.addEventListener('change', once, { once: true });
      else if (mq.addListener) mq.addListener(once);
    };
    arm();
  }

  function toImage(ev) {
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left;
    const sy = ev.clientY - rect.top;
    return {
      x: (sx - state.view.x) / state.view.scale,
      y: (sy - state.view.y) / state.view.scale,
    };
  }

  function draw() {
    const dpr = state.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    if (!state.img) return;

    ctx.save();
    const { scale, x, y } = state.view;
    ctx.translate(x, y);
    ctx.scale(scale, scale);
    ctx.imageSmoothingEnabled = scale < 3;
    ctx.drawImage(state.img, 0, 0, state.width, state.height);

    state.instances.forEach((inst) => {
      if (!inst.visible) return;
      ctx.globalAlpha = state.selectedId === null || state.selectedId === inst.id ? 1 : 0.45;
      ctx.drawImage(inst.layer, 0, 0);
      ctx.globalAlpha = 1;
      if (state.selectedId === inst.id) {
        const rgb = colourFor(inst.colourIndex);
        ctx.lineWidth = 2 / scale;
        ctx.strokeStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.strokeRect(inst.bbox[0], inst.bbox[1], inst.bbox[2], inst.bbox[3]);
      }
    });

    const cand = state.candidates[state.candidateIndex];
    if (cand) {
      ctx.drawImage(cand.layer, 0, 0);
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = '#ffffff';
      ctx.setLineDash([6 / scale, 4 / scale]);
      ctx.strokeRect(cand.box[0], cand.box[1], cand.box[2] - cand.box[0], cand.box[3] - cand.box[1]);
      ctx.setLineDash([]);
    }

    if (state.box) {
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = '#42a5f5';
      ctx.strokeRect(state.box[0], state.box[1], state.box[2] - state.box[0], state.box[3] - state.box[1]);
    }
    if (state.dragStart && state.dragNow) {
      ctx.lineWidth = 1.5 / scale;
      ctx.strokeStyle = '#90caf9';
      ctx.setLineDash([5 / scale, 3 / scale]);
      ctx.strokeRect(
        Math.min(state.dragStart.x, state.dragNow.x),
        Math.min(state.dragStart.y, state.dragNow.y),
        Math.abs(state.dragNow.x - state.dragStart.x),
        Math.abs(state.dragNow.y - state.dragStart.y)
      );
      ctx.setLineDash([]);
    }

    const r = 5 / scale;
    state.pos.forEach((p) => marker(p, r, '#66bb6a'));
    state.neg.forEach((p) => marker(p, r, '#ef5350'));

    ctx.restore();
  }

  function marker(p, r, colour) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = r * 0.35;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
  }

  /* ── SAM ────────────────────────────────────────────────────────────── */

  async function runSegment() {
    // Guard here rather than only on the pointer handlers: the button, Space
    // and the auto-run checkbox all reach this, and overlapping requests would
    // interleave on the single sam3_serve stream.
    if (state.busy) return;
    if (state.server !== SERVER.ENCODED) {
      setStatus(state.server === SERVER.READY
        ? '画像のエンコードが終わっていません' : '先にモデルを読み込んでください', 'warn');
      return;
    }
    if (!state.pos.length && !state.box) {
      setStatus(state.neg.length ? '前景の点か矩形が必要です' : 'プロンプトがありません', 'warn');
      return;
    }

    const operation = ++state.operation, promptRevision = state.promptRevision;
    const prompt = window.AnnotationProject.snapshotPrompt(state.pos,state.neg,state.box);
    const backend = state.activeBackend && JSON.parse(JSON.stringify(state.activeBackend));
    state.candidates = []; state.candidateIndex = 0; renderCandidates();
    setBusy(true, 'セグメンテーション中…');
    const t0 = performance.now();
    try {
      const res = await window.Bridge.samSegment(
        prompt.pos, prompt.neg, prompt.box, $('multimask').checked
      );
      if (operation !== state.operation || promptRevision !== state.promptRevision) return;
      const dets = res.detections || [];
      const previewColour = colourFor(nextColourIndex());
      state.candidates = dets.map((d) => {
        if (!d || !Array.isArray(d.mask_size) || d.mask_size.length !== 2) return null;
        const r = rasterise(d.counts, d.mask_size[0], d.mask_size[1], previewColour, 130);
        return r && Number.isFinite(d.score) && Array.isArray(d.box) && d.box.length === 4 && d.box.every(Number.isFinite) && d.mask_size[0] === state.width && d.mask_size[1] === state.height ? { ...d, layer: r.canvas, mask: r.mask, prompt, promptRevision, backend } : null;
      }).filter(Boolean);
      // The decoder rejects an RLE that does not describe mask_size exactly, so
      // a malformed detection is dropped rather than drawn at the wrong offset.
      const dropped = dets.length - state.candidates.length;
      // Highest score first: multimask candidates arrive in decoder-token order,
      // and the status line and the default selection both mean "the best one".
      state.candidates.sort((a, b) => b.score - a.score);
      state.candidateIndex = 0;

      const ms = Math.round(performance.now() - t0);
      if (!state.candidates.length) {
        setStatus(dropped ? `マスクを復号できませんでした (${ms}ms)` : `検出なし (${ms}ms)`,
                  dropped ? 'err' : 'warn');
      } else {
        const note = dropped ? ` / ${dropped} 件は復号失敗` : '';
        setStatus(`候補 ${state.candidates.length} 件 / score=${state.candidates[0].score.toFixed(3)}${note} (${ms}ms)`, 'ok');
      }
    } catch (err) {
      if (operation !== state.operation) return;
      const msg = String(err);
      if (/exited unexpectedly|not loaded yet|spawn|pipe|timed out|model stopped|cancelled/i.test(msg)) serverDied(msg);
      else setStatus('失敗: ' + msg, 'err');
    } finally {
      if (operation !== state.operation) return;
      setBusy(false);
      renderCandidates();
      draw();
    }
  }

  function acceptCandidate() {
    const cand = state.candidates[state.candidateIndex];
    if (!window.AnnotationProject.candidateCurrent(cand,state.promptRevision) || state.busy) return;
    const w = cand.mask_size[0];
    const h = cand.mask_size[1];
    const enteredId = $('fruitId').value.trim();
    const fruitId = enteredId ? Number(enteredId) : state.project.nextFruitId;
    if (!Number.isSafeInteger(fruitId) || fruitId < 1 || fruitId > 2147483647) { setStatus('Fruit ID は正の整数で指定してください', 'err'); return; }
    const previous = state.instances.find(i => i.fruitId === fruitId);
    if (previous && !window.confirm(`Fruit #${fruitId} のこのフレームのマスクを置き換えますか?`)) return;
    const colourIndex = (fruitId - 1) % PALETTE.length;
    const baked = rasterise(cand.counts, w, h, colourFor(colourIndex), 110);
    if (!baked) { setStatus('マスクを復号できませんでした', 'err'); return; }
    const inst = {
      id: fruitId, fruitId, segmentId: previous ? previous.segmentId : `${fruitId}:frame-${state.project.frames[state.frameIndex].sourceFrameIndex}`,
      reviewStatus: 'draft', identityStatus: 'unverified',
      prompt: window.AnnotationProject.snapshotPrompt(cand.prompt.pos,cand.prompt.neg,cand.prompt.box),
      label: $('label').value.trim() || 'object',
      counts: cand.counts,
      // Carried through to COCO's segmentation.size: the grid these counts
      // were produced on, which is not necessarily the displayed image size.
      maskWidth: w,
      maskHeight: h,
      score: cand.score,
      source: 'sam',
      provenance: {model: cand.backend, method:'interactive-prompt'},
      visible: true,
      colourIndex,
      area: window.RLE.area(cand.mask),
      bbox: window.RLE.bbox(cand.mask, w, h),
      layer: baked.canvas,
    };
    if (previous) state.instances.splice(state.instances.indexOf(previous), 1);
    state.instances.push(inst);
    window.AnnotationProject.recordModel(state.project, state.activeBackend);
    state.project.nextFruitId = Math.max(state.project.nextFruitId, fruitId + 1);
    $('fruitId').value = '';
    markDirty('accept-mask',fruitId);
    state.selectedId = inst.id;
    clearPrompt();
    renderInstances();
    draw();
    setStatus(`Fruit #${inst.id} 「${inst.label}」を下書き保存 (${inst.area} px)。マスクと同一性を確認してください。`, 'ok');
  }

  function clearPrompt() {
    state.promptRevision++;
    state.pos = [];
    state.neg = [];
    state.box = null;
    state.promptLog = [];
    state.dragStart = null;
    state.dragNow = null;
    state.candidates = [];
    state.candidateIndex = 0;
    renderCandidates();
    syncControls();
  }

  // Undo the last prompt edit, or — once the prompt is empty — restore the
  // most recently deleted instance. Deleting was previously irreversible.
  function undo() {
    if (state.busy) return;
    if (state.promptLog.length) {
      const last = state.promptLog.pop();
      if (last.kind === 'pos') state.pos.pop();
      else if (last.kind === 'neg') state.neg.pop();
      else if (last.kind === 'box') state.box = null;
      state.promptRevision++;
      state.candidates = [];
      state.candidateIndex = 0;
      renderCandidates();
      draw();
      syncControls();
      setStatus('プロンプトを 1 つ戻しました');
      return;
    }
    if (state.deleted.length) {
      const entry = state.deleted[state.deleted.length - 1];
      if (!window.AnnotationProject.restoreDeleted(state.instances, entry)) { setStatus(`Fruit #${entry.inst.fruitId} の新しいマスクがあるため、重複する復元を中止しました。`, 'warn'); return; }
      state.deleted.pop();
      const {inst} = entry;
      markDirty('restore-mask',inst.fruitId);
      renderInstances();
      draw();
      setStatus(`#${inst.id} を復元しました`, 'ok');
    }
  }

  function pushPrompt(kind) {
    state.promptRevision++; state.candidates = []; state.candidateIndex = 0; renderCandidates();
    state.promptLog.push({ kind });
    setStatus('プロンプトを変更しました。再実行してください。');
    syncControls();
  }

  /* ── lists ──────────────────────────────────────────────────────────── */

  function renderCandidates() {
    const box = $('candidates');
    box.innerHTML = '';
    state.candidates.forEach((c, i) => {
      const b = document.createElement('button');
      b.className = 'chip' + (i === state.candidateIndex ? ' on' : '');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(i === state.candidateIndex));
      b.textContent = `#${i + 1} ${c.score.toFixed(3)}`;
      b.onclick = () => { state.candidateIndex = i; renderCandidates(); draw(); };
      box.appendChild(b);
    });
    syncControls();
  }

  function renderInstances() {
    const list = $('instances');
    // Rebuilding the list steals focus from whatever the user was typing in;
    // remember where it was and put it back.
    const active = document.activeElement;
    const focusedId = active && active.dataset ? active.dataset.instanceId : null;
    const selStart = active && active.selectionStart;

    list.innerHTML = '';
    $('count').textContent = String(state.instances.length);

    state.instances.forEach((inst, i) => {
      const row = document.createElement('li');
      row.className = 'inst' + (state.selectedId === inst.id ? ' sel' : '');
      row.dataset.id = String(inst.id);
      const rgb = colourFor(inst.colourIndex);

      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.background = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      swatch.setAttribute('aria-hidden', 'true');

      const name = document.createElement('input');
      name.value = inst.label;
      name.className = 'name';
      name.dataset.instanceId = String(inst.id);
      name.setAttribute('aria-label', `インスタンス ${inst.id} のラベル`);
      name.onchange = () => { inst.label = name.value.trim() || 'object'; markDirty('relabel',inst.fruitId); };
      name.onfocus = () => { state.selectedId = inst.id; list.querySelectorAll('.inst').forEach(el => el.classList.toggle('sel', Number(el.dataset.id) === inst.id)); draw(); };

      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = `#${inst.fruitId} ${inst.area}px`;
      meta.title = `score=${inst.score.toFixed(3)} / bbox=${inst.bbox.join(', ')}`;

      const eye = document.createElement('button');
      eye.className = 'icon';
      eye.type = 'button';
      eye.textContent = inst.visible ? '👁' : '🚫';
      eye.title = inst.visible ? 'マスクを隠す' : 'マスクを表示';
      eye.setAttribute('aria-label', `インスタンス ${inst.id} の表示切替`);
      eye.onclick = () => { if (state.busy) return; inst.visible = !inst.visible; markDirty('visibility',inst.fruitId); renderInstances(); draw(); };

      const del = document.createElement('button');
      del.className = 'icon danger';
      del.type = 'button';
      del.textContent = '✕';
      del.title = '削除 (元に戻せます)';
      del.setAttribute('aria-label', `インスタンス ${inst.id} を削除`);
      del.onclick = () => {
        if (state.busy) return;
        state.deleted.push({ inst, index: i });
        state.instances.splice(i, 1);
        markDirty('delete-mask',inst.fruitId);
        if (state.selectedId === inst.id) state.selectedId = null;
        renderInstances();
        draw();
        setStatus(`#${inst.id} を削除しました — 元に戻す: Ctrl+Z`);
      };

      const review = document.createElement('button');
      review.className = 'review'; review.type = 'button';
      review.textContent = inst.reviewStatus === 'reviewed' ? '確認済' : '下書き';
      review.title = 'マスクと動画内の果実 ID を確認した後に切り替えてください';
      review.onclick = () => { if (state.busy) return; inst.reviewStatus = inst.reviewStatus === 'reviewed' ? 'draft' : 'reviewed'; inst.identityStatus = inst.reviewStatus === 'reviewed' ? 'confirmed' : 'unverified'; markDirty('review-status',inst.fruitId); renderInstances(); };
      row.onclick = (ev) => { if (ev.target === row || ev.target === meta) select(inst.id); };
      row.append(swatch, name, meta, review, eye, del);
      list.appendChild(row);
    });

    if (focusedId) {
      const back = list.querySelector(`.name[data-instance-id="${focusedId}"]`);
      if (back) {
        back.focus();
        if (typeof selStart === 'number') { try { back.setSelectionRange(selStart, selStart); } catch (e) { /* not selectable */ } }
      }
    }
    syncControls();
  }

  function select(id) {
    state.selectedId = state.selectedId === id ? null : id;
    renderInstances();
    draw();
  }

  /* ── export ─────────────────────────────────────────────────────────── */

  function markDirty(action,fruitId) { if (action && state.project) window.AnnotationProject.recordManualEdit(state.project,action,state.project.frames[state.frameIndex]?.key,fruitId); state.revision++; state.dirty = true; syncControls(); }

  function downloadJson(value, name) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function exportCoco() {
    if (!state.project || state.busy) return;
    let checked;
    try { checked = window.AnnotationProject.validate(window.AnnotationProject.serialise(state.project)); }
    catch (e) { setStatus('書き出し前の検証に失敗: ' + e, 'err'); return; }
    downloadJson(window.COCO.build(checked), 'annotations.coco.json');
    setStatus('全フレームの COCO を書き出しました。fruit_id・segment_id・確認状態を保持します。', 'ok');
  }
  async function saveProject() {
    if (!state.project || state.busy || state.saving) return;
    const project = state.project, revision = state.revision;
    let doc;
    try { doc = window.AnnotationProject.serialise(window.AnnotationProject.validate(window.AnnotationProject.serialise(project))); }
    catch (e) { setStatus('保存前の検証に失敗: ' + e, 'err'); return; }
    state.saving = true; syncControls();
    try {
      if (window.Bridge.saveProject) await window.Bridge.saveProject(doc);
      downloadJson(doc, 'annotation-project.json');
      if (state.project === project && state.revision === revision) state.dirty = false;
      setStatus(state.dirty ? '保存後に変更があります。最新の内容をもう一度保存してください。' : 'プロジェクトを保存しました。元画像は含まれません。', state.dirty ? 'warn' : 'ok');
    } catch (e) { setStatus('保存に失敗: ' + e, 'err'); }
    finally { state.saving = false; syncControls(); }
  }
  function updateFrameSelect() {
    const select = $('frameSelect'); select.innerHTML = '';
    if (!state.sequence) return;
    state.sequence.frames.forEach((f, i) => {
      const o = document.createElement('option'); o.value = i; o.textContent = `${i + 1}: ${f.fileName}`; select.append(o);
    });
    select.value = state.frameIndex;
  }
  function bakeInstances(frame) {
    frame.instances.forEach(i => { const r = rasterise(i.counts, i.maskWidth, i.maskHeight, colourFor(i.colourIndex), 110); i.layer = r && r.canvas; });
  }
  async function displayFrame(index, encode = true, ownOperation = null) {
    const operation = ownOperation === null ? ++state.operation : ownOperation;
    const source = state.sequence.frames[index], frame = state.project.frames[index];
    state.server = state.server === SERVER.ENCODED ? SERVER.READY : state.server;
    const info = await window.Bridge.readImage(source.path);
    const img = new Image();
    await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error('画像をデコードできません')); img.src = info.data_url; });
    if (operation !== state.operation) return;
    if (frame.instances.length && (frame.width !== img.naturalWidth || frame.height !== img.naturalHeight)) throw new Error('保存済みマスクと元画像のサイズが一致しません');
    if (state.frameIndex !== index) state.instances.forEach(i => { delete i.layer; });
    state.frameIndex = index; state.img = img; state.width = img.naturalWidth; state.height = img.naturalHeight;
    frame.width = state.width; frame.height = state.height;
    state.imagePath = source.path; state.fileName = source.fileName;
    bakeInstances(frame); state.instances = frame.instances; state.deleted = []; state.selectedId = null;
    clearPrompt(); fitView(); renderInstances(); draw(); $('frameSelect').value = index;
    $('imageName').textContent = `${source.fileName} (${index + 1}/${state.sequence.frames.length})`;
    if (encode && state.server === SERVER.READY) {
      setStatus('画像をエンコード中…', 'work'); await window.Bridge.samLoadImage(source.path);
      if (operation !== state.operation) return;
      state.server = SERVER.ENCODED;
      await updateCacheStatus(operation);
    }
  }
  async function navigateFrame(index) {
    if (state.busy || !state.sequence || index < 0 || index >= state.sequence.frames.length) return;
    const operation = ++state.operation; setBusy(true, 'フレームを読み込み中…');
    try { await displayFrame(index, true, operation); if (operation === state.operation) { setStatus('フレームを読み込みました。ID と下書きを保持しています。', 'ok'); return true; } return false; }
    catch (e) { if (operation === state.operation) setStatus('フレーム読み込み失敗: ' + e, 'err'); return false; }
    finally { if (operation === state.operation) setBusy(false); }
  }
  async function loadImage(path) {
    if (state.busy) return;
    const seq = window.Bridge.sequenceFor(path);
    if (state.project && state.sequence && state.sequence.id === seq.id) return navigateFrame(seq.frames.findIndex(f => f.path === path));
    if (state.dirty && !window.confirm('未保存のプロジェクトを破棄して別のシーケンスを開きますか?')) return;
    const previous = {sequence:state.sequence,project:state.project,frameIndex:state.frameIndex,dirty:state.dirty};
    clearPrompt();
    state.sequence = seq; state.project = window.AnnotationProject.newProject(seq.id, seq.frames); state.frameIndex = -1;
    if (state.activeBackend) state.project.model = JSON.parse(JSON.stringify(state.activeBackend));
    updateFrameSelect(); state.dirty = false;
    if (!await navigateFrame(seq.frames.findIndex(f => f.path === path))) {
      Object.assign(state,previous); updateFrameSelect();
      if (previous.sequence && previous.frameIndex >= 0) await navigateFrame(previous.frameIndex);
      setStatus('新しい画像列を開けません。以前のプロジェクトを保持しました。', 'err');
      syncControls();
    }
  }
  async function importProject(file) {
    if (!file || state.busy) return;
    if (file.size > 256 * 1024 * 1024) { setStatus('プロジェクトは 256 MiB 以下にしてください', 'err'); return; }
    try {
      const raw = JSON.parse(await file.text());
      const project = raw.schema ? window.AnnotationProject.validate(raw) : window.AnnotationProject.fromCoco(raw);
      const seq = window.Bridge.matchSequence(project);
      if (!seq) throw new Error('登録された画像列と frame_key が一致しません。起動時の --image-root を確認してください。');
      if (state.dirty && !window.confirm('未保存のプロジェクトを破棄して読み込みますか?')) return;
      const previous = {sequence:state.sequence,project:state.project,frameIndex:state.frameIndex,dirty:state.dirty};
      state.sequence = seq; state.project = project; state.frameIndex = -1; state.dirty = false; updateFrameSelect();
      if (!await navigateFrame(0)) {
        Object.assign(state,previous); updateFrameSelect();
        if (previous.sequence && previous.frameIndex >= 0) await navigateFrame(previous.frameIndex);
        throw new Error('元画像を開けないため、以前のプロジェクトを保持しました');
      }
      setStatus('プロジェクトを復元しました。マスク・果実 ID・確認状態を保持しています。', 'ok');
    } catch (e) { setStatus('読み込み失敗: ' + e, 'err'); }
  }
  async function cancel() {
    if (!state.busy || !window.Bridge.canCancel) return;
    ++state.operation; state.server = SERVER.DEAD; state.capabilities = {}; state.activeBackend = null; clearPrompt();
    $('cacheStatus').textContent = 'キャッシュ統計: モデル停止';
    try { await window.Bridge.samStop(); setStatus('停止しました。完了済みフレームは保持。再開にはモデルの再読み込みと再シードが必要です。', 'warn'); }
    catch (e) { setStatus('停止を確認できません: ' + e, 'err'); }
    finally { setBusy(false); }
  }
  async function resetTracking() {
    if (state.busy) return;
    setBusy(true, '追跡メモリをリセット中…');
    try { await window.Bridge.samTrackReset(); state.server = SERVER.READY; clearPrompt(); setStatus('追跡メモリをリセットしました。保存済み注釈と ID は保持しています。', 'ok'); }
    catch (e) { setStatus('リセット失敗: ' + e, 'err'); }
    finally { setBusy(false); }
  }
  async function propagate() {
    if (state.busy || !state.capabilities.video_tracking || !state.project) return;
    const seeds = state.instances.filter(i => i.prompt);
    if (!seeds.length) { setStatus('現在フレームに点または矩形シードが必要です', 'warn'); return; }
    const start = state.frameIndex;
    if (state.project.frames.slice(start + 1).some(f => f.instances.length) && !window.confirm('以降の既存マスクは保持し、未注釈の果実だけ追跡候補を追加しますか?')) return;
    const operation = ++state.operation; setBusy(true, '動画追跡を初期化中…');
    try {
      await window.AnnotationTracking.run({
        project: state.project, sequence: state.sequence, seeds, startFrame: start,
        bridge: window.Bridge, backend: state.activeBackend, isCancelled: () => operation !== state.operation,
        onProgress: (id, index) => setStatus(`Fruit #${id} / フレーム ${index + 1}/${state.sequence.frames.length} — 全結果は下書き`, 'work'),
        onFrame: async (index) => { window.AnnotationProject.recordModel(state.project,state.activeBackend); markDirty(); await displayFrame(index, false, operation); },
      });
      if (operation === state.operation) { state.server = SERVER.READY; await updateCacheStatus(operation); }
      if (operation === state.operation) setStatus('伝播完了。果実 ID は維持されていますが、各フレームの同一性とマスクは未確認です。', 'ok');
    } catch (e) { if (operation === state.operation) { state.server = SERVER.READY; setStatus('追跡失敗。完了済みフレームを保持: ' + e, 'err'); } }
    finally { if (operation === state.operation) setBusy(false); }
  }

  /* ── wiring ─────────────────────────────────────────────────────────── */

  async function startModel() {
    if (state.busy) return;
    const modelPath = $('modelPath').value.trim();
    if (!modelPath) { setStatus('モデルのパスを指定してください', 'warn'); return; }
    const operation = ++state.operation;
    state.server = SERVER.NONE; state.capabilities = {}; state.activeBackend = null; clearPrompt();
    $('cacheStatus').textContent = 'キャッシュ統計: モデル読み込み中';
    setBusy(true, 'モデルを読み込み中…');
    try {
      const ready = await window.Bridge.samStart({
        modelPath,
        cpu: $('cpu').checked,
        nThreads: parseInt($('threads').value, 10) || 4,
      });
      if (operation !== state.operation) return;
      state.capabilities = ready || {};
      const info = `${ready.compute || 'native'} / ${ready.image_mode || 'image'}${ready.preprocessing ? ' / ' + ready.preprocessing : ''} / 動画追跡: ${ready.experimental_video ? '実験モード・品質未認証' : ready.video_tracking ? '対応' : '未対応・検証待ち'}`;
      $('modelCapabilities').textContent = info + (ready.compute === 'experimental_ggml_q8_0_cpu' ? ' / Q8 計算は実験的・品質未認証' : '') + '。出力は確認前の下書きです。';
      state.activeBackend = { name: ready.model_name || modelPath.split(/[\\/]/).pop(), sha256: ready.model_sha256 || null, native_executable_sha256: ready.native_executable_sha256 || null, compute: ready.compute, imageMode: ready.image_mode, preprocessing: ready.preprocessing || null, build_identity: ready.build_identity || null, runtime_provenance: ready.runtime_provenance ? JSON.parse(JSON.stringify(ready.runtime_provenance)) : null, cache_config: ready.cache_config ? {...ready.cache_config} : null, modelType: ready.model_type, videoTracking: !!ready.video_tracking, experimentalVideo: !!ready.experimental_video };
      // Loading a new backend must never rewrite the origins of saved masks.
      state.server = SERVER.READY;
      $('startModel').textContent = 'モデル再読み込み';
      if (state.imagePath) {
        setStatus('画像をエンコード中… (数秒)', 'work');
        await window.Bridge.samLoadImage(state.imagePath);
        if (operation !== state.operation) return;
        state.server = SERVER.ENCODED;
      }
      await updateCacheStatus(operation);
      if (operation !== state.operation) return;
      setStatus(state.imagePath ? '準備完了 — クリックでセグメンテーション'
                                : 'モデル読み込み完了 — 画像を開いてください', 'ok');
    } catch (err) {
      if (operation !== state.operation) return;
      state.server = SERVER.NONE;
      $('cacheStatus').textContent = 'キャッシュ統計: モデル読み込み失敗';
      $('modelCapabilities').textContent = 'モデル読み込みに失敗しました。別モデルへの自動切替はありません。';
      setStatus('モデル読み込み失敗: ' + err, 'err');
    } finally {
      if (operation === state.operation) setBusy(false);
    }
  }

  function setMode(mode) {
    if (state.busy) return;
    if (state.mode !== mode) { state.promptRevision++; state.candidates = []; state.candidateIndex = 0; renderCandidates(); }
    state.mode = mode;
    // Switching prompt kind without clearing the other one leaves an invisible
    // half of the prompt attached to the next run.
    if (mode === 'point' && state.box) { state.box = null; state.promptLog = state.promptLog.filter((e) => e.kind !== 'box'); }
    if (mode === 'box') { state.pos = []; state.neg = []; state.promptLog = state.promptLog.filter((e) => e.kind === 'box'); }
    document.querySelectorAll('[data-mode]').forEach((o) => {
      const on = o.dataset.mode === mode;
      o.classList.toggle('on', on);
      o.setAttribute('aria-pressed', String(on));
    });
    canvas.style.cursor = mode === 'pan' ? 'grab' : 'crosshair';
    draw();
    syncControls();
  }

  function bind() {
    $('openImage').onclick = async () => {
      const p = await window.Bridge.openImageDialog();
      if (p) await loadImage(p);
    };
    $('browseModel').onclick = async () => {
      const p = await window.Bridge.openModelDialog();
      if (p) $('modelPath').value = p;
    };
    $('startModel').onclick = startModel;
    $('refreshCacheStats').onclick = refreshCacheStats;
    $('segment').onclick = runSegment;
    $('accept').onclick = acceptCandidate;
    $('undo').onclick = undo;
    $('clear').onclick = () => { clearPrompt(); draw(); setStatus('プロンプトを消去しました'); };
    $('export').onclick = exportCoco;
    $('saveProject').onclick = saveProject;
    $('importProject').onclick = () => $('projectFile').click();
    $('restoreSaved').onclick = async () => {
      if (state.busy || !window.Bridge.loadSavedProject) return;
      setBusy(true, '保存済みプロジェクトを読み込み中…');
      try { const raw = await window.Bridge.loadSavedProject(); setBusy(false); await importProject({size:0,text:async()=>JSON.stringify(raw)}); }
      catch (e) { setStatus('保存済みプロジェクト読み込み失敗: ' + e, 'err'); }
      finally { setBusy(false); }
    };
    $('projectFile').onchange = async () => { await importProject($('projectFile').files[0]); $('projectFile').value = ''; };
    $('previousFrame').onclick = () => navigateFrame(state.frameIndex - 1);
    $('nextFrame').onclick = () => navigateFrame(state.frameIndex + 1);
    $('frameSelect').onchange = () => navigateFrame(Number($('frameSelect').value));
    $('propagate').onclick = propagate; $('cancel').onclick = cancel; $('resetTracking').onclick = resetTracking;
    $('fit').onclick = () => { fitView(); draw(); };

    document.querySelectorAll('[data-mode]').forEach((b) => {
      b.onclick = () => setMode(b.dataset.mode);
    });

    canvas.addEventListener('pointerdown', (ev) => {
      if (!state.img || state.busy) return;
      // Middle button and Alt pan from any mode, so panning never costs a mode
      // switch mid-annotation.
      if (state.mode === 'pan' || ev.button === 1 || ev.altKey) {
        state.panFrom = { x: ev.clientX, y: ev.clientY, vx: state.view.x, vy: state.view.y };
        canvas.setPointerCapture(ev.pointerId);
        canvas.style.cursor = 'grabbing';
        ev.preventDefault();
        return;
      }
      if (state.mode === 'box') {
        state.dragStart = toImage(ev);
        state.dragNow = state.dragStart;
        canvas.setPointerCapture(ev.pointerId);
      }
    });

    canvas.addEventListener('pointermove', (ev) => {
      if (state.panFrom) {
        state.view.x = state.panFrom.vx + (ev.clientX - state.panFrom.x);
        state.view.y = state.panFrom.vy + (ev.clientY - state.panFrom.y);
        draw();
        return;
      }
      if (state.dragStart) { state.dragNow = toImage(ev); draw(); }
    });

    canvas.addEventListener('pointerup', (ev) => {
      if (state.panFrom) {
        state.panFrom = null;
        canvas.style.cursor = state.mode === 'pan' ? 'grab' : 'crosshair';
        return;
      }
      if (!state.img) return;
      if (state.busy) {
        // Drop any drag that started before the request; leaving dragStart set
        // would keep painting a rubber-band box until the next pointerdown.
        state.dragStart = null;
        state.dragNow = null;
        draw();
        return;
      }

      if (state.dragStart) {
        const a = state.dragStart;
        const b = toImage(ev);
        state.dragStart = null;
        state.dragNow = null;
        if (Math.abs(a.x - b.x) > 3 && Math.abs(a.y - b.y) > 3) {
          state.box = [Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x, b.x), Math.max(a.y, b.y)];
          pushPrompt('box');
          draw();
          if ($('auto').checked) runSegment();
        } else {
          draw();
        }
        return;
      }

      if (state.mode !== 'point') return;
      const p = toImage(ev);
      if (p.x < 0 || p.y < 0 || p.x >= state.width || p.y >= state.height) return;
      // Shift (or the right button) marks background — the same gesture SAM
      // demos use, so it needs no explaining.
      const negative = ev.shiftKey || ev.button === 2;
      (negative ? state.neg : state.pos).push([p.x, p.y]);
      pushPrompt(negative ? 'neg' : 'pos');
      draw();
      if ($('auto').checked) runSegment();
    });

    canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

    canvas.addEventListener('wheel', (ev) => {
      if (!state.img) return;
      ev.preventDefault();
      const before = toImage(ev);
      const factor = ev.deltaY < 0 ? 1.1 : 1 / 1.1;
      state.view.scale = Math.min(16, Math.max(0.05, state.view.scale * factor));
      const rect = canvas.getBoundingClientRect();
      state.view.x = ev.clientX - rect.left - before.x * state.view.scale;
      state.view.y = ev.clientY - rect.top - before.y * state.view.scale;
      draw();
    }, { passive: false });

    // Enter in the label field accepts — the ↵ badge sits next to that field,
    // so swallowing it there would make the badge a lie. Space is left alone
    // inside inputs so it can still type a space.
    $('label').addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); acceptCandidate(); }
    });

    window.addEventListener('keydown', (ev) => {
      const t = ev.target;
      const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'z') { ev.preventDefault(); undo(); return; }
      if (ev.key === 'Escape') { if (state.busy) return; clearPrompt(); draw(); return; }
      if (typing) return;
      // Space and Enter activate a focused button; only claim them otherwise.
      if (t && t.tagName === 'BUTTON') return;
      if (ev.key === 'Enter') { ev.preventDefault(); acceptCandidate(); }
      if (ev.key === ' ') { ev.preventDefault(); runSegment(); }
    });

    // Resizing used to call fitView(), throwing away whatever the user had
    // zoomed to. Keep the view and only re-place it the first time.
    window.addEventListener('resize', () => {
      resizeCanvas();
      if (!state.viewPlaced) fitView();
      draw();
    });

    window.addEventListener('beforeunload', (ev) => {
      if (state.dirty || state.busy || state.saving) { ev.preventDefault(); ev.returnValue = ''; }
    });
  }

  function init() {
    // Without this a throw anywhere below leaves the header stuck on 起動中…
    // and the whole UI dead with nothing said.
    window.addEventListener('error', (ev) => {
      setStatus('内部エラー: ' + (ev.message || ev.error), 'err');
    });
    window.addEventListener('unhandledrejection', (ev) => {
      setStatus('内部エラー: ' + (ev.reason && ev.reason.message ? ev.reason.message : ev.reason), 'err');
    });

    resizeCanvas();
    watchDpr();
    bind();
    setMode('point');
    renderInstances();
    renderCandidates();

    if (!window.Bridge.native) {
      document.body.classList.add('browser');
      setStatus('ブラウザで開いています — SAM 機能はデスクトップ版 (Tauri) が必要です', 'warn');
    } else {
      setStatus('モデルを選んで開始してください');
    }
    syncControls();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
