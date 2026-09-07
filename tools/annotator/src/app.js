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
    promptLog: [],          // ordered record of prompt edits, for undo
    dragStart: null,
    dragNow: null,
    panFrom: null,
    candidates: [],
    candidateIndex: 0,
    instances: [],
    nextId: 1,
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
  const NATIVE_CONTROLS = ['openImage', 'browseModel', 'startModel', 'segment', 'accept', 'clear', 'export', 'undo'];

  function syncControls() {
    const native = window.Bridge.native;
    const busy = state.busy;
    const hasImage = !!state.img;
    const encoded = state.server === SERVER.ENCODED;

    $('startModel').disabled = busy || !native;
    $('browseModel').disabled = busy || !native;
    $('openImage').disabled = busy || !native;
    $('segment').disabled = busy || !native || !encoded || !hasPrompt();
    $('accept').disabled = busy || !state.candidates.length;
    $('clear').disabled = busy || (!hasPrompt() && !state.candidates.length);
    $('undo').disabled = busy || (!state.promptLog.length && !state.deleted.length);
    $('export').disabled = busy || state.instances.length === 0;
    $('fit').disabled = !hasImage;
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
    setStatus('セグメンテーションサーバが停止しました: ' + err + ' — モデルを開始し直してください', 'err');
    $('startModel').textContent = 'モデル開始';
    syncControls();
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

    setBusy(true, 'セグメンテーション中…');
    const t0 = performance.now();
    try {
      const res = await window.Bridge.samSegment(
        state.pos, state.neg, state.box, $('multimask').checked
      );
      const dets = res.detections || [];
      const previewColour = colourFor(nextColourIndex());
      state.candidates = dets.map((d) => {
        const r = rasterise(d.counts, d.mask_size[0], d.mask_size[1], previewColour, 130);
        return r ? { ...d, layer: r.canvas, mask: r.mask } : null;
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
      const msg = String(err);
      if (/exited unexpectedly|not loaded yet|spawn|Broken pipe/i.test(msg)) serverDied(msg);
      else setStatus('失敗: ' + msg, 'err');
    } finally {
      setBusy(false);
      renderCandidates();
      draw();
    }
  }

  function acceptCandidate() {
    const cand = state.candidates[state.candidateIndex];
    if (!cand || state.busy) return;
    const w = cand.mask_size[0];
    const h = cand.mask_size[1];
    const colourIndex = nextColourIndex();
    const baked = rasterise(cand.counts, w, h, colourFor(colourIndex), 110);
    if (!baked) { setStatus('マスクを復号できませんでした', 'err'); return; }
    const inst = {
      id: state.nextId++,
      label: $('label').value.trim() || 'object',
      counts: cand.counts,
      // Carried through to COCO's segmentation.size: the grid these counts
      // were produced on, which is not necessarily the displayed image size.
      maskWidth: w,
      maskHeight: h,
      score: cand.score,
      source: 'sam',
      visible: true,
      colourIndex,
      area: window.RLE.area(cand.mask),
      bbox: window.RLE.bbox(cand.mask, w, h),
      layer: baked.canvas,
    };
    state.instances.push(inst);
    state.selectedId = inst.id;
    clearPrompt();
    renderInstances();
    draw();
    setStatus(`#${inst.id} 「${inst.label}」を確定 (${inst.area} px)`, 'ok');
  }

  function clearPrompt() {
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
      state.candidates = [];
      state.candidateIndex = 0;
      renderCandidates();
      draw();
      syncControls();
      setStatus('プロンプトを 1 つ戻しました');
      return;
    }
    if (state.deleted.length) {
      const { inst, index } = state.deleted.pop();
      state.instances.splice(Math.min(index, state.instances.length), 0, inst);
      renderInstances();
      draw();
      setStatus(`#${inst.id} を復元しました`, 'ok');
    }
  }

  function pushPrompt(kind) {
    state.promptLog.push({ kind });
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
      name.onchange = () => { inst.label = name.value.trim() || 'object'; };
      name.onfocus = () => { select(inst.id); };

      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = `${inst.area}px`;
      meta.title = `score=${inst.score.toFixed(3)} / bbox=${inst.bbox.join(', ')}`;

      const eye = document.createElement('button');
      eye.className = 'icon';
      eye.type = 'button';
      eye.textContent = inst.visible ? '👁' : '🚫';
      eye.title = inst.visible ? 'マスクを隠す' : 'マスクを表示';
      eye.setAttribute('aria-label', `インスタンス ${inst.id} の表示切替`);
      eye.onclick = () => { inst.visible = !inst.visible; renderInstances(); draw(); };

      const del = document.createElement('button');
      del.className = 'icon danger';
      del.type = 'button';
      del.textContent = '✕';
      del.title = '削除 (元に戻せます)';
      del.setAttribute('aria-label', `インスタンス ${inst.id} を削除`);
      del.onclick = () => {
        state.deleted.push({ inst, index: i });
        state.instances.splice(i, 1);
        if (state.selectedId === inst.id) state.selectedId = null;
        renderInstances();
        draw();
        setStatus(`#${inst.id} を削除しました — 元に戻す: Ctrl+Z`);
      };

      row.onclick = (ev) => { if (ev.target === row || ev.target === meta) select(inst.id); };
      row.append(swatch, name, meta, eye, del);
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

  function exportCoco() {
    const doc = window.COCO.build({
      width: state.width,
      height: state.height,
      fileName: state.fileName,
      instances: state.instances,
    });
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (state.fileName || 'annotations').replace(/\.[^.]+$/, '') + '.coco.json';
    // Some engines ignore a click on a detached anchor, and revoking the URL in
    // the same task can cancel a download that has not started reading yet.
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 0);
    state.dirty = false;
    setStatus(`COCO を書き出しました (${state.instances.length} 件)`, 'ok');
  }

  /* ── wiring ─────────────────────────────────────────────────────────── */

  async function loadImage(path) {
    if (state.busy) return;
    // Loading discards every accepted instance, and there is no project save,
    // so unexported work would vanish without a word.
    if (state.instances.length && !window.confirm(
      `${state.instances.length} 件のインスタンスが未書き出しです。破棄して別の画像を開きますか?`)) {
      return;
    }

    setBusy(true, '画像を読み込み中…');
    try {
      const info = await window.Bridge.readImage(path);
      const img = new Image();
      await new Promise((res, rej) => {
        img.onload = res;
        img.onerror = () => rej(new Error('画像をデコードできません'));
        img.src = info.data_url;
      });
      state.img = img;
      state.width = img.naturalWidth;
      state.height = img.naturalHeight;
      state.imagePath = path;
      state.fileName = path.split(/[\\/]/).pop();
      state.instances = [];
      state.deleted = [];
      state.selectedId = null;
      state.nextId = 1;
      clearPrompt();
      fitView();
      renderInstances();
      draw();
      $('imageName').textContent = state.fileName;

      if (state.server === SERVER.READY || state.server === SERVER.ENCODED) {
        // Demote first: until the encode succeeds the server still holds the
        // previous image, and segmenting then would answer for the wrong one.
        state.server = SERVER.READY;
        setStatus('画像をエンコード中… (数秒)', 'work');
        await window.Bridge.samLoadImage(path);
        state.server = SERVER.ENCODED;
        setStatus('準備完了 — クリックでセグメンテーション', 'ok');
      } else {
        setStatus('画像を読み込みました。モデルを開始してください', 'warn');
      }
    } catch (err) {
      const msg = String(err);
      if (/exited unexpectedly|not loaded yet|spawn/i.test(msg)) serverDied(msg);
      else setStatus('失敗: ' + msg, 'err');
    } finally {
      setBusy(false);
    }
  }

  async function startModel() {
    if (state.busy) return;
    const modelPath = $('modelPath').value.trim();
    if (!modelPath) { setStatus('モデルのパスを指定してください', 'warn'); return; }
    setBusy(true, 'モデルを読み込み中…');
    try {
      await window.Bridge.samStart({
        modelPath,
        cpu: $('cpu').checked,
        nThreads: parseInt($('threads').value, 10) || 4,
      });
      state.server = SERVER.READY;
      $('startModel').textContent = 'モデル再読み込み';
      if (state.imagePath) {
        setStatus('画像をエンコード中… (数秒)', 'work');
        await window.Bridge.samLoadImage(state.imagePath);
        state.server = SERVER.ENCODED;
      }
      setStatus(state.imagePath ? '準備完了 — クリックでセグメンテーション'
                                : 'モデル読み込み完了 — 画像を開いてください', 'ok');
    } catch (err) {
      state.server = SERVER.NONE;
      setStatus('モデル読み込み失敗: ' + err, 'err');
    } finally {
      setBusy(false);
    }
  }

  function setMode(mode) {
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
    $('segment').onclick = runSegment;
    $('accept').onclick = acceptCandidate;
    $('undo').onclick = undo;
    $('clear').onclick = () => { clearPrompt(); draw(); setStatus('プロンプトを消去しました'); };
    $('export').onclick = exportCoco;
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
      if (ev.key === 'Escape') { clearPrompt(); draw(); return; }
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
      if (state.instances.length) { ev.preventDefault(); ev.returnValue = ''; }
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
