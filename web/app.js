/* 鼓谱 前端：曲库 + 乐谱显示（alphaTab）+ 与原音频同步播放 */
(() => {
  const $ = (s) => document.querySelector(s);
  const el = {
    list: $('#list'), drop: $('#drop'), file: $('#file'),
    empty: $('#empty'), working: $('#working'), viewer: $('#viewer'),
    wTitle: $('#w-title'), wBar: $('#w-bar'), wMsg: $('#w-msg'), wRetry: $('#w-retry'),
    title: $('#title'), chips: $('#chips'), tracks: $('#tracks'),
    scroll: $('#scroll'), at: $('#at'), loading: $('#loading'),
    legend: $('#legend'), legendAt: $('#legend-at'),
    audio: $('#audio'), play: $('#play'), playIco: $('#play-ico'),
    tCur: $('#t-cur'), tDur: $('#t-dur'), seek: $('#seek'), barno: $('#barno'),
    speed: $('#speed'), listen: $('#listen'), follow: $('#follow'),
    zoomVal: $('#zoom-val'), toast: $('#toast'),
    menuExport: $('#menu-export'), panelFix: $('#panel-fix'),
  };
  const LISTEN = { mix: '原曲', nodrums: '去鼓伴奏', drums: '仅鼓', bass: '仅贝斯', guitar: '仅吉他', piano: '仅键盘', vocals: '仅人声', other: '仅其他乐器' };
  const KIND = { kick: '底鼓', snare: '军鼓', hh_closed: '闭镲', hh_open: '开镲', crash: '吊镲', ride: '叮叮镲', tom_hi: '高通', tom_mid: '中通', tom_floor: '落地通' };

  const S = { projects: [], cur: null, curId: null, api: null, score: null, sel: null, zoom: 1, seeking: false, timer: 0, shift: 0 };

  // ---------------------------------------------------------------- 小工具
  const fmt = (t) => { t = Math.max(0, t || 0); return Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0'); };
  let toastTimer = 0;
  function toast(msg) {
    el.toast.textContent = msg; el.toast.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => (el.toast.hidden = true), 3200);
  }
  async function api(path, opt) {
    const r = await fetch(path, opt);
    if (!r.ok) {
      let m = r.statusText;
      try { m = (await r.json()).detail || m; } catch (e) { /* 非 JSON */ }
      throw new Error(m);
    }
    const ct = r.headers.get('content-type') || '';
    return ct.includes('json') ? r.json() : r.text();
  }

  // ---------------------------------------------------------------- 曲库
  async function refreshList() {
    try { S.projects = await api('/api/projects'); } catch (e) { return; }
    renderList();
    const cur = S.projects.find((p) => p.id === S.curId);
    if (S.curId && !cur) { S.curId = null; S.cur = null; show('empty'); }
    if (cur && (!S.cur || S.cur.status !== 'done')) {
      if (cur.status === 'done') open(cur.id); else showWorking(cur);
    }
  }
  function renderList() {
    el.list.replaceChildren(...S.projects.map((p) => {
      const li = document.createElement('li');
      li.className = p.id === S.curId ? 'on' : '';
      li.tabIndex = 0;
      const nm = document.createElement('div'); nm.className = 'nm'; nm.textContent = p.title; nm.title = p.filename;
      const st = document.createElement('div'); st.className = 'st' + (p.status === 'error' ? ' err' : '');
      st.textContent = p.status === 'done' ? new Date(p.created * 1000).toLocaleDateString('zh-CN')
        : p.status === 'error' ? '处理失败' : p.status === 'queued' ? '排队中' : `${p.message} ${Math.round(p.progress * 100)}%`;
      const del = document.createElement('button'); del.className = 'del'; del.textContent = '✕'; del.title = '删除'; del.setAttribute('aria-label', '删除 ' + p.title);
      del.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm(`删除「${p.title}」及其全部分轨和乐谱？`)) return;
        try { await api('/api/projects/' + p.id, { method: 'DELETE' }); } catch (err) { toast(err.message); }
        if (S.curId === p.id) { stopAudio(); S.curId = null; S.cur = null; show('empty'); }
        refreshList();
      };
      li.append(nm, del, st);
      if (p.status === 'processing') {
        const mb = document.createElement('div'); mb.className = 'mini-bar';
        const i = document.createElement('i'); i.style.width = Math.round(p.progress * 100) + '%'; mb.append(i); li.append(mb);
      }
      li.onclick = () => open(p.id);
      li.onkeydown = (e) => { if (e.key === 'Enter') open(p.id); };
      return li;
    }));
  }
  async function upload(files) {
    let first = null;
    for (const f of files) {
      const fd = new FormData(); fd.append('file', f);
      try {
        const p = await api('/api/projects', { method: 'POST', body: fd });
        first = first || p;
      } catch (e) { toast(`${f.name}：${e.message}`); }
    }
    await refreshList();
    if (first) open(first.id);
  }
  el.file.onchange = () => { if (el.file.files.length) upload([...el.file.files]); el.file.value = ''; };
  el.drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') el.file.click(); };
  ['dragenter', 'dragover'].forEach((t) => window.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => window.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove('over'); }));
  window.addEventListener('drop', (e) => { if (e.dataTransfer && e.dataTransfer.files.length) upload([...e.dataTransfer.files]); });

  function show(which) {
    el.empty.hidden = which !== 'empty';
    el.working.hidden = which !== 'working';
    el.viewer.hidden = which !== 'viewer';
  }
  function showWorking(p) {
    stopAudio();
    S.cur = p; show('working');
    el.wTitle.textContent = p.title;
    el.wBar.style.width = Math.round((p.progress || 0) * 100) + '%';
    el.wMsg.textContent = p.status === 'error' ? '处理失败：' + (p.error || '') : p.status === 'queued' ? '排队中…' : `${p.message}… ${Math.round(p.progress * 100)}%`;
    el.wRetry.hidden = p.status !== 'error';
  }
  el.wRetry.onclick = async () => { try { await api(`/api/projects/${S.curId}/retry`, { method: 'POST' }); refreshList(); } catch (e) { toast(e.message); } };

  async function open(id) {
    S.curId = id; renderList();
    let p;
    try { p = await api('/api/projects/' + id); } catch (e) { toast(e.message); return; }
    if (S.curId !== id) return;
    if (p.status !== 'done') { showWorking(p); return; }
    stopAudio();
    S.cur = p; show('viewer');
    fillHeader(p);
    fillListen(p);
    await loadScore(p, true);
  }

  // ---------------------------------------------------------------- 头部信息
  function fillHeader(p) {
    const sc = p.score;
    el.title.textContent = p.title;
    const chips = [`♩ = ${Math.round(sc.tempo)}`, `${sc.numerator}/4 拍`, `${sc.bars} 小节`, fmt(sc.duration)];
    el.chips.replaceChildren(...chips.map((t) => { const s = document.createElement('span'); s.textContent = t; return s; }));
    const dc = sc.drum_counts || {};
    el.chips.title = Object.keys(dc).map((k) => `${KIND[k] || k} ${dc[k]}`).join('　');
  }
  el.title.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); el.title.blur(); } if (e.key === 'Escape') { el.title.textContent = S.cur.title; el.title.blur(); } };
  el.title.onblur = async () => {
    const t = el.title.textContent.trim();
    if (!S.cur || !t || t === S.cur.title) { if (S.cur) el.title.textContent = S.cur.title; return; }
    await rerender({}, t);
  };

  // ---------------------------------------------------------------- alphaTab
  function ensureApi() {
    if (S.api) return S.api;
    const at = new alphaTab.AlphaTabApi(el.at, {
      core: { fontDirectory: '/alphatab/font/', logLevel: 'warning' },
      display: { layoutMode: 'page', scale: S.zoom, stretchForce: 0.9 },
      notation: {
        rhythmMode: 'showwithbars',
        elements: new Map([[alphaTab.NotationElement.EffectDynamics, false], [alphaTab.NotationElement.GuitarTuning, false]]),
      },
      player: {
        playerMode: alphaTab.PlayerMode.EnabledExternalMedia,
        enableCursor: true, enableUserInteraction: true, enableElementHighlighting: true,
        scrollElement: el.scroll, scrollMode: alphaTab.ScrollMode.Continuous, scrollOffsetY: -30,
      },
    });
    S.api = at;
    at.error.on((e) => { console.error(e); toast('乐谱解析出错：' + (e && e.message ? e.message : e)); el.loading.hidden = true; });
    at.renderStarted.on(() => { el.loading.hidden = false; });
    at.renderFinished.on(() => { el.loading.hidden = true; });
    at.scoreLoaded.on((score) => { S.score = score; buildTracks(); attachMedia(); });
    return at;
  }

  // 让 alphaTab 把 <audio> 当作时间轴（官方 external media 接法）
  function attachMedia() {
    const at = S.api, a = el.audio;
    if (!at.player || !at.player.output) return;
    at.player.output.handler = {
      get backingTrackDuration() { return Number.isFinite(a.duration) ? a.duration * 1000 : (S.cur ? S.cur.score.duration * 1000 : 0); },
      get playbackRate() { return a.playbackRate; },
      set playbackRate(v) { a.playbackRate = v; },
      get masterVolume() { return a.volume; },
      set masterVolume(v) { a.volume = v; },
      // 同步点记录的就是音频文件自身的时间，变速不改变它，所以这里不做速度换算
      seekTo(ms) { a.currentTime = Math.max(0, ms / 1000); },
      play() { a.play(); },
      pause() { a.pause(); },
    };
  }
  function pushPos() {
    const at = S.api, a = el.audio;
    if (at && at.player && at.player.output && at.player.output.updatePosition) {
      at.player.output.updatePosition(a.currentTime * 1000);
    }
    if (!S.seeking) {
      const d = a.duration || (S.cur && S.cur.score.duration) || 1;
      el.seek.value = Math.round((a.currentTime / d) * 1000);
    }
    el.tCur.textContent = fmt(a.currentTime);
    if (S.cur && S.cur.score) {
      const bt = S.cur.score.bar_times; let lo = 0, hi = bt.length - 1;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (bt[m] <= a.currentTime + 0.01) lo = m; else hi = m - 1; }
      el.barno.textContent = `第 ${lo + 1} / ${bt.length} 小节`;
    }
  }
  const a = el.audio;
  a.addEventListener('timeupdate', pushPos);
  a.addEventListener('seeked', pushPos);
  a.addEventListener('loadedmetadata', () => { el.tDur.textContent = fmt(a.duration); });
  a.addEventListener('play', () => {
    clearInterval(S.timer);
    if (S.api) S.api.play();
    S.timer = setInterval(pushPos, 40);
    el.playIco.setAttribute('d', 'M6 4h4v16H6zM14 4h4v16h-4z'); el.play.setAttribute('aria-label', '暂停');
  });
  a.addEventListener('pause', () => {
    if (S.api) S.api.pause();
    clearInterval(S.timer);
    el.playIco.setAttribute('d', 'M7 4l14 8-14 8z'); el.play.setAttribute('aria-label', '播放');
  });
  a.addEventListener('ratechange', () => { if (S.api) S.api.playbackSpeed = a.playbackRate; });
  a.addEventListener('error', () => { if (a.src) toast('音频加载失败'); });
  function stopAudio() { a.pause(); a.removeAttribute('src'); a.load(); clearInterval(S.timer); }
  function togglePlay() { if (!S.cur || S.cur.status !== 'done') return; if (a.paused) a.play(); else a.pause(); }
  el.play.onclick = togglePlay;
  el.seek.oninput = () => { S.seeking = true; const d = a.duration || 1; el.tCur.textContent = fmt((el.seek.value / 1000) * d); };
  el.seek.onchange = () => { const d = a.duration || 1; a.currentTime = (el.seek.value / 1000) * d; S.seeking = false; };
  el.speed.onchange = () => { a.playbackRate = parseFloat(el.speed.value); a.preservesPitch = true; };
  el.follow.onchange = () => {
    if (!S.api) return;
    S.api.settings.player.scrollMode = el.follow.checked ? alphaTab.ScrollMode.Continuous : alphaTab.ScrollMode.Off;
    S.api.updateSettings();
  };
  function fillListen(p) {
    const keys = Object.keys(LISTEN).filter((k) => p.media.includes(k));
    el.listen.replaceChildren(...keys.map((k) => { const o = document.createElement('option'); o.value = k; o.textContent = LISTEN[k]; return o; }));
    el.listen.value = 'mix';
    setSource('mix', 0, false);
  }
  function setSource(key, at, resume) {
    const rate = parseFloat(el.speed.value);
    a.src = `/media/${S.cur.id}/${key}.flac`;
    a.addEventListener('loadedmetadata', function once() {
      a.removeEventListener('loadedmetadata', once);
      a.playbackRate = rate; a.currentTime = at;
      if (resume) a.play();
    });
    a.load();
  }
  el.listen.onchange = () => setSource(el.listen.value, a.currentTime, !a.paused);

  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || e.target.isContentEditable) return;
    if (el.viewer.hidden) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      const bt = S.cur.score.bar_times; let i = 0;
      while (i + 1 < bt.length && bt[i + 1] <= a.currentTime + 0.05) i++;
      i = Math.min(bt.length - 1, Math.max(0, i + (e.key === 'ArrowRight' ? 1 : (a.currentTime - bt[i] > 0.6 ? 0 : -1))));
      a.currentTime = bt[i]; e.preventDefault();
    }
  });

  // ---------------------------------------------------------------- 乐谱与音轨
  async function loadScore(p, resetSel) {
    const at = ensureApi();
    el.loading.hidden = false;
    let tex;
    try { tex = await api(`/api/projects/${p.id}/score`); } catch (e) { toast(e.message); return; }
    S.tex = tex;
    const keys = p.score.tracks.map((t) => t.key);
    if (resetSel || (S.sel !== 'all' && !keys.includes(S.sel))) S.sel = keys.includes('drums') ? 'drums' : 'all';
    const idx = S.sel === 'all' ? 'all' : [keys.indexOf(S.sel)];
    at.tex(tex, idx);
  }
  function buildTracks() {
    const p = S.cur; if (!p) return;
    const mk = (key, label, n) => {
      const b = document.createElement('button');
      b.textContent = label; b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', S.sel === key);
      if (n != null) { const s = document.createElement('small'); s.textContent = n; b.append(s); }
      b.className = S.sel === key ? 'on' : '';
      b.onclick = () => selectTrack(key);
      return b;
    };
    const tabs = p.score.tracks.map((t) => mk(t.key, t.name, t.notes));
    if (p.score.tracks.length > 1) tabs.unshift(mk('all', '总谱'));
    el.tracks.replaceChildren(...tabs);
    el.legend.hidden = !(S.sel === 'drums' || S.sel === 'all') || !p.score.tracks.some((t) => t.key === 'drums');
    if (!el.legend.hidden) renderLegend();
  }
  function selectTrack(key) {
    if (!S.score || S.sel === key) return;
    S.sel = key;
    const keys = S.cur.score.tracks.map((t) => t.key);
    const tracks = key === 'all' ? S.score.tracks : [S.score.tracks[keys.indexOf(key)]];
    S.api.renderTracks(tracks);
    buildTracks();
    el.scroll.scrollTop = 0;
  }

  let legendApi = null;
  function renderLegend() {
    if (legendApi) return;
    const items = [['Kick', 36, '底鼓'], ['Snare', 38, '军鼓'], ['Ghost', 38, '鬼音'], ['HH', 42, '闭镲'], ['HHOpen', 46, '开镲'], ['Crash', 49, '吊镲'], ['Ride', 51, '叮叮镲'], ['TomHi', 48, '高通'], ['TomMid', 47, '中通'], ['TomFloor', 43, '落地通']];
    let tex = '\\track "图例"\n\\staff {score} \\instrument percussion \\clef neutral\n';
    tex += items.filter((i) => i[0] !== 'Ghost').map((i) => `\\articulation ("${i[0]}" ${i[1]})`).join('\n') + '\n';
    tex += `\\ts (${items.length} 4) ` + items.map((i) => (i[0] === 'Ghost' ? `(Snare{g}).4 {txt "${i[2]}"}` : `(${i[0]}).4 {txt "${i[2]}"}`)).join(' ');
    legendApi = new alphaTab.AlphaTabApi(el.legendAt, {
      core: { fontDirectory: '/alphatab/font/', tex: false, logLevel: 'warning' },
      display: { layoutMode: 'horizontal', scale: 0.9, padding: [8, 4] },
      notation: { elements: new Map([[alphaTab.NotationElement.EffectDynamics, false], [alphaTab.NotationElement.TrackNames, false]]) },
      player: { playerMode: alphaTab.PlayerMode.Disabled },
    });
    legendApi.tex(tex);
  }

  // ---------------------------------------------------------------- 缩放 / 导出 / 打印
  function setZoom(z) {
    S.zoom = Math.min(1.6, Math.max(0.6, Math.round(z * 10) / 10));
    el.zoomVal.textContent = Math.round(S.zoom * 100) + '%';
    if (S.api) { S.api.settings.display.scale = S.zoom; S.api.updateSettings(); S.api.render(); }
  }
  $('#zoom-in').onclick = () => setZoom(S.zoom + 0.1);
  $('#zoom-out').onclick = () => setZoom(S.zoom - 0.1);

  $('#btn-print').onclick = () => {
    if (!S.api || !S.score) return;
    // alphaTab 会打开一个按 A4 排好版的窗口并调起打印；目标打印机选“另存为 PDF”即可导出
    S.api.print();
    toast('已打开 A4 排版窗口。打印机选“另存为 PDF”即可导出。');
  };
  function download(data, name, type) {
    const url = URL.createObjectURL(new Blob([data], { type }));
    const l = document.createElement('a'); l.href = url; l.download = name; document.body.append(l); l.click(); l.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  function popToggle(pop, anchor) {
    const open = pop.hidden;
    el.menuExport.hidden = true; el.panelFix.hidden = true;
    if (open) {
      pop.hidden = false;
      const r = anchor.getBoundingClientRect(), v = el.viewer.getBoundingClientRect();
      pop.style.top = r.bottom - v.top + 6 + 'px';
      pop.style.right = Math.max(8, v.right - r.right) + 'px';
    }
  }
  $('#btn-export').onclick = (e) => { e.stopPropagation(); popToggle(el.menuExport, e.currentTarget); };
  el.menuExport.onclick = (e) => {
    const x = e.target.dataset.x; if (!x || !S.score) return;
    el.menuExport.hidden = true;
    const name = S.cur.title;
    try {
      if (x === 'gp') download(new alphaTab.exporter.Gp7Exporter().export(S.score, S.api.settings), name + '.gp', 'application/octet-stream');
      if (x === 'midi') S.api.downloadMidi();
      if (x === 'atex') download(S.tex, name + '.atex', 'text/plain;charset=utf-8');
    } catch (err) { console.error(err); toast('导出失败：' + err.message); }
  };
  document.addEventListener('click', (e) => {
    if (!el.menuExport.hidden && !el.menuExport.contains(e.target)) el.menuExport.hidden = true;
    if (!el.panelFix.hidden && !el.panelFix.contains(e.target) && e.target.id !== 'btn-fix') el.panelFix.hidden = true;
  });

  // ---------------------------------------------------------------- 校正
  const fx = { bpb: $('#opt-bpb'), ts: $('#opt-ts'), sens: $('#opt-sens'), sensVal: $('#sens-val'), ghost: $('#opt-ghost'), open: $('#opt-open'), shiftVal: $('#shift-val') };
  const sensText = (v) => (v > 0.05 ? `多识别 +${Math.round(v * 10)}` : v < -0.05 ? `少识别 ${Math.round(v * 10)}` : '标准');
  $('#btn-fix').onclick = (e) => {
    e.stopPropagation();
    const o = S.cur.score.options;
    S.shift = o.beat_shift; fx.shiftVal.textContent = (S.shift > 0 ? '+' : '') + S.shift + ' 拍';
    fx.bpb.value = String(o.beats_per_bar); fx.ts.value = String(o.tempo_scale);
    fx.sens.value = o.drum_sensitivity; fx.sensVal.textContent = sensText(o.drum_sensitivity);
    fx.ghost.checked = !!o.ghost_notes; fx.open.checked = !!o.open_hihat;
    popToggle(el.panelFix, e.currentTarget);
  };
  fx.sens.oninput = () => { fx.sensVal.textContent = sensText(parseFloat(fx.sens.value)); };
  el.panelFix.querySelectorAll('[data-shift]').forEach((b) => (b.onclick = () => {
    S.shift += parseInt(b.dataset.shift, 10); fx.shiftVal.textContent = (S.shift > 0 ? '+' : '') + S.shift + ' 拍';
  }));
  $('#fix-cancel').onclick = () => (el.panelFix.hidden = true);
  $('#fix-apply').onclick = async () => {
    el.panelFix.hidden = true;
    await rerender({
      beat_shift: S.shift, beats_per_bar: parseInt(fx.bpb.value, 10), tempo_scale: parseFloat(fx.ts.value),
      drum_sensitivity: parseFloat(fx.sens.value), ghost_notes: fx.ghost.checked, open_hihat: fx.open.checked,
    });
  };
  async function rerender(options, title) {
    el.loading.hidden = false;
    const t = a.currentTime, playing = !a.paused;
    a.pause();
    try {
      const p = await api(`/api/projects/${S.cur.id}/render`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ options, title }) });
      S.cur = p; fillHeader(p); refreshList();
      await loadScore(p, false);
      a.currentTime = t; if (playing) a.play();
    } catch (e) { toast('重新出谱失败：' + e.message); el.loading.hidden = true; }
  }

  // ---------------------------------------------------------------- 启动
  refreshList();
  setInterval(() => { if (S.projects.some((p) => p.status === 'queued' || p.status === 'processing')) refreshList(); }, 1500);
  window.__app = S; // 调试用
})();
