/* Vision-Nav in the browser: DINOv2 retrieval (ONNX Runtime Web), ORB + RANSAC re-ranking (OpenCV.js), A* over landmarks. */
(() => {
  "use strict";
  const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const TOP_K = 8, CONNECT = 320;
  const $ = (s) => document.querySelector(s);
  const nf = new Intl.NumberFormat("en");
  const f1 = new Intl.NumberFormat("en", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

  let cv, session, tiles, tileEmb, mapImg, mapGray, orb, bf, landmarks, evalData;
  let W = 1498, H = 704;
  const state = { frame: 0, query: null, truth: null, result: null, start: null, target: null, route: null, colour: false, busy: false };
  const baseCanvases = {};
  const ink = {};

  /* ---------- helpers ---------- */
  function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function rgbOf(color) {
    const c = document.createElement("canvas"); c.width = c.height = 1;
    const x = c.getContext("2d"); x.fillStyle = color; x.fillRect(0, 0, 1, 1);
    return [...x.getImageData(0, 0, 1, 1).data.slice(0, 3)];
  }
  function toGps(x, y) {
    const g = tiles.gps;
    return [g.tl_lat + (g.br_lat - g.tl_lat) * (y / H), g.tl_lon + (g.br_lon - g.tl_lon) * (x / W)];
  }
  function metres(a, b) {
    const R = 6371000, r = Math.PI / 180, p1 = a[0] * r, p2 = b[0] * r, dp = p2 - p1, dl = (b[1] - a[1]) * r;
    return 2 * R * Math.asin(Math.sqrt(Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2));
  }
  const pxMetres = (a, b) => metres(toGps(a[0], a[1]), toGps(b[0], b[1]));
  function loadImage(src) {
    return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
  }
  async function fetchWithProgress(url, onBytes) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    if (!r.body || !r.body.getReader) { const b = await r.arrayBuffer(); onBytes(b.byteLength); return b; }
    const reader = r.body.getReader(); const parts = []; let n = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); n += value.length; onBytes(value.length); }
    const out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
    return out.buffer;
  }
  /* cyanotype print: grey value mapped from the sheet's darkest blue to its white line */
  function cyanotype(imgData) {
    const d = imgData.data, lo = ink.lo, hi = ink.hi;
    for (let i = 0; i < d.length; i += 4) {
      const g = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      const t = Math.min(1, Math.max(0, (g - 0.1) / 0.78)) ** 1.15;
      d[i] = lo[0] + (hi[0] - lo[0]) * t; d[i + 1] = lo[1] + (hi[1] - lo[1]) * t; d[i + 2] = lo[2] + (hi[2] - lo[2]) * t;
    }
    return imgData;
  }

  /* ---------- loading ---------- */
  async function boot() {
    ink.line = token("--color-ink"); ink.muted = token("--color-muted"); ink.sheet = token("--color-paper"); ink.rule2 = token("--color-rule-2");
    ink.lo = rgbOf(token("--color-paper-3")); ink.hi = rgbOf(token("--color-ink"));
    const [tj, tb, lj, ej, img] = await Promise.all([
      fetch("data/tiles.json").then((r) => r.json()), fetch("data/tiles.bin").then((r) => r.arrayBuffer()),
      fetch("data/landmarks.json").then((r) => r.json()), fetch("data/eval.json").then((r) => r.json()), loadImage("data/iitr_campus.png"),
    ]);
    tiles = tj; landmarks = lj; evalData = ej; mapImg = img; [W, H] = tiles.map;
    const q = new Int8Array(tb), D = tiles.dim; tileEmb = new Float32Array(q.length);
    for (let i = 0; i < tiles.count; i++) {
      let s = 0; const sc = tiles.rowScale[i];
      for (let j = 0; j < D; j++) { const v = q[i * D + j] * sc; tileEmb[i * D + j] = v; s += v * v; }
      s = Math.sqrt(s) || 1; for (let j = 0; j < D; j++) tileEmb[i * D + j] /= s;
    }
    for (const mode of ["cyan", "true"]) {
      const c = document.createElement("canvas"); c.width = W; c.height = H;
      const x = c.getContext("2d"); x.drawImage(mapImg, 0, 0);
      if (mode === "cyan") x.putImageData(cyanotype(x.getImageData(0, 0, W, H)), 0, 0);
      baseCanvases[mode] = c;
    }
    $("#dim-label").textContent = `${nf.format(Math.round(metres([tiles.gps.tl_lat, tiles.gps.tl_lon], [tiles.gps.tl_lat, tiles.gps.br_lon])))} m`;
    buildThumbs(); buildEval(); bindUI();
    const q0 = new URLSearchParams(location.search).get("frame");
    await selectFrame(q0 !== null && +q0 >= 0 && +q0 < evalData.rows.length ? +q0 : 0, false);
    drawMap();
    loadModels();
  }

  async function loadModels() {
    $("#error").hidden = true; $("#progress").hidden = false;
    const run = $("#run"); run.disabled = true; run.textContent = "Loading the model";
    const total = 25.0e6 + 13.3e6; let got = 0;
    const tick = (n) => { got += n; $("#meter").style.transform = `scaleX(${Math.min(1, got / total)})`; $("#load-note").textContent = `Fetching DINOv2 and OpenCV: ${f1.format(got / 1e6)} of 38.3 MB. Cached after the first visit.`; };
    try {
      if (!window.ort) throw new Error("ONNX Runtime did not load");
      ort.env.wasm.wasmPaths = ORT_DIST; ort.env.wasm.numThreads = 1;
      const [modelBuf, cvBuf] = await Promise.all([fetchWithProgress("model/dinov2_vits14_int8.onnx", tick), fetchWithProgress("vendor/opencv.js", tick)]);
      $("#load-note").textContent = "Starting the network and OpenCV";
      session = await ort.InferenceSession.create(modelBuf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      const url = URL.createObjectURL(new Blob([cvBuf], { type: "text/javascript" }));
      await new Promise((res, rej) => { const s = document.createElement("script"); s.src = url; s.onload = res; s.onerror = () => rej(new Error("OpenCV did not start")); document.head.append(s); });
      let mod = window.cv;
      if (mod && typeof mod.then === "function" && !mod.Mat) mod = await mod;
      if (!mod.Mat) await new Promise((res) => { mod.onRuntimeInitialized = res; });
      cv = mod;
      const rgba = cv.matFromImageData(baseCanvases.true.getContext("2d").getImageData(0, 0, W, H));
      mapGray = new cv.Mat(); cv.cvtColor(rgba, mapGray, cv.COLOR_RGBA2GRAY); rgba.delete();
      orb = new cv.ORB(2000); bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
      $("#progress").hidden = true;
      run.disabled = false; run.textContent = "Locate this frame"; $("#run-all").disabled = false;
      locate();
    } catch (e) {
      console.error(e);
      $("#progress").hidden = true; $("#error").hidden = false;
      $("#error-text").textContent = `The model did not load (${e.message}). A network filter may block the CDN, or this browser lacks WebAssembly.`;
      run.textContent = "Model unavailable";
    }
  }

  /* ---------- UI ---------- */
  function buildThumbs() {
    const box = $("#thumbs");
    box.innerHTML = evalData.rows.map((r, i) => `<button class="thumb" type="button" role="option" aria-selected="false" data-i="${i}" aria-label="Test frame ${i + 1}, ${r.file}"><img src="frames/${r.file}" alt="" width="180" height="180" loading="lazy"></button>`).join("");
    box.addEventListener("click", (e) => { const b = e.target.closest(".thumb"); if (b) selectFrame(+b.dataset.i, true); });
  }
  function buildEval() {
    const s = evalData.summary;
    $("#d-register").textContent = `Error in metres. Retrieval alone has a median of ${f1.format(s.retrieval_median_m)} m; re-ranking brings it to ${f1.format(s.refined_median_m)} m, with ${s.refined_under_25m} of ${s.frames} frames inside 25 m.`;
    $("#t-eval tbody").innerHTML = evalData.rows.map((r, i) => `<tr data-frame="${i}" tabindex="0"><td><span class="num">${r.file}</span></td><td class="n">${f1.format(r.retrieval_m)}</td><td class="n">${f1.format(r.refined_m)}</td><td class="n" id="ev-${i}"><span class="muted">not run</span></td><td class="n">${r.inliers}</td></tr>`).join("");
    $("#t-eval tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-frame]"); if (tr) { selectFrame(+tr.dataset.frame, true).then(() => $("#locate").scrollIntoView({ behavior: "smooth" })); } });
    $("#t-eval tbody").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-frame]"); if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); tr.click(); } });
  }
  function bindUI() {
    $("#run").addEventListener("click", () => locate());
    $("#retry").addEventListener("click", loadModels);
    $("#run-all").addEventListener("click", runAll);
    $("#upload").addEventListener("change", async (e) => {
      const f = e.target.files && e.target.files[0]; if (!f) return;
      const url = URL.createObjectURL(f);
      try { const img = await loadImage(url); setQuery(img, f.name, null, "Your image. It locates only if it shows this campus from above."); if (session) locate(); }
      catch { $("#frame-note").textContent = "That file is not an image this browser can read."; }
    });
    for (const id of ["l-cand", "l-land", "l-route", "l-colour"]) $(`#${id}`).addEventListener("change", () => { state.colour = $("#l-colour").checked; drawMap(); drawMatches(); });
    const map = $("#map");
    map.addEventListener("click", (e) => {
      const p = mapPoint(e);
      const mode = document.querySelector('input[name="mode"]:checked').value;
      if (mode === "start") { state.start = p; state.startManual = true; } else state.target = p;
      $("#l-route").checked = true; plan(); drawMap();
    });
    map.addEventListener("pointermove", (e) => { const p = mapPoint(e); const g = toGps(p[0], p[1]); $("#cursor").textContent = `${p[0]}, ${p[1]} px  ${g[0].toFixed(5)}, ${g[1].toFixed(5)}`; });
    map.addEventListener("pointerleave", () => { $("#cursor").textContent = ""; });
    window.addEventListener("resize", () => drawMap());
  }
  function mapPoint(e) {
    const r = $("#map").getBoundingClientRect();
    return [Math.round(((e.clientX - r.left) / r.width) * W), Math.round(((e.clientY - r.top) / r.height) * H)];
  }
  async function selectFrame(i, run) {
    state.frame = i;
    const r = evalData.rows[i];
    const img = await loadImage(`frames/${r.file}`);
    setQuery(img, r.file, r.truth, "Simulated drone frame, ground truth known");
    document.querySelectorAll(".thumb").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.i === i)));
    history.replaceState(null, "", `?frame=${i}`);
    if (run && session) await locate();
  }
  function setQuery(img, name, truth, note) {
    state.query = img; state.truth = truth; state.result = null;
    $("#frame-name").textContent = name; $("#frame-note").textContent = note;
    const c = $("#query"), x = c.getContext("2d");
    x.clearRect(0, 0, c.width, c.height); x.drawImage(img, 0, 0, c.width, c.height);
    for (const id of ["r-px", "r-gps", "r-err", "r-inl", "r-ms"]) $(`#${id}`).textContent = session ? "running" : "waiting";
    drawMap(); drawMatches();
  }

  /* ---------- the pipeline ---------- */
  function embed(img) {
    const S = tiles.size, c = document.createElement("canvas"); c.width = c.height = S;
    const x = c.getContext("2d"); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = "high"; x.drawImage(img, 0, 0, S, S);
    const px = x.getImageData(0, 0, S, S).data, data = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) for (let ch = 0; ch < 3; ch++) data[ch * S * S + i] = (px[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
    return session.run({ image: new ort.Tensor("float32", data, [1, 3, S, S]) }).then((o) => o.embedding.data);
  }
  function search(e) {
    const D = tiles.dim, sims = new Float32Array(tiles.count);
    for (let i = 0; i < tiles.count; i++) { let s = 0; for (let j = 0; j < D; j++) s += tileEmb[i * D + j] * e[j]; sims[i] = s; }
    return [...sims.keys()].sort((a, b) => sims[b] - sims[a]).slice(0, TOP_K).map((i) => ({ i, sim: sims[i], x: tiles.positions[i][0], y: tiles.positions[i][1], tile: tiles.scales[i] }));
  }
  function queryGray(img) {
    const c = document.createElement("canvas"); c.width = img.naturalWidth || img.width; c.height = img.naturalHeight || img.height;
    const x = c.getContext("2d"); x.drawImage(img, 0, 0);
    const rgba = cv.matFromImageData(x.getImageData(0, 0, c.width, c.height)), g = new cv.Mat();
    cv.cvtColor(rgba, g, cv.COLOR_RGBA2GRAY); rgba.delete(); return g;
  }
  /* port of Localizer._refine with the ORB matcher from core/models.py */
  function refine(qGray, cand) {
    const { x: cx, y: cy, tile } = cand, half = tile >> 1;
    const x1 = Math.max(0, cx - half), x2 = Math.min(W, cx + half), y1 = Math.max(0, cy - half), y2 = Math.min(H, cy + half);
    const crop = mapGray.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
    const uav = new cv.Mat(); cv.resize(qGray, uav, new cv.Size(x2 - x1, y2 - y1), 0, 0, cv.INTER_LINEAR);
    const k0 = new cv.KeyPointVector(), k1 = new cv.KeyPointVector(), d0 = new cv.Mat(), d1 = new cv.Mat(), none = new cv.Mat();
    const out = { x: cx, y: cy, inliers: 0, pairs: [], box: [x1, y1, x2 - x1, y2 - y1] };
    const trash = [crop, uav, k0, k1, d0, d1, none];
    try {
      orb.detectAndCompute(uav, none, k0, d0); orb.detectAndCompute(crop, none, k1, d1);
      if (d0.rows < 4 || d1.rows < 4) return out;
      const mv = new cv.DMatchVector(); trash.push(mv); bf.match(d0, d1, mv);
      const ms = []; for (let i = 0; i < mv.size(); i++) ms.push(mv.get(i));
      ms.sort((a, b) => a.distance - b.distance); const top = ms.slice(0, 200);
      if (top.length < 4) { out.inliers = top.length; return out; }
      const a0 = [], a1 = [];
      for (const m of top) { const p = k0.get(m.queryIdx).pt, q = k1.get(m.trainIdx).pt; a0.push(p.x, p.y); a1.push(q.x, q.y); }
      const p0 = cv.matFromArray(top.length, 1, cv.CV_32FC2, a0), p1 = cv.matFromArray(top.length, 1, cv.CV_32FC2, a1), mask = new cv.Mat();
      trash.push(p0, p1, mask);
      const Hm = cv.findHomography(p0, p1, cv.RANSAC, 5.0, mask); trash.push(Hm);
      if (Hm.empty()) { out.inliers = top.length; return out; }
      const h = Hm.data64F, ux = uav.cols / 2, uy = uav.rows / 2, w = h[6] * ux + h[7] * uy + h[8];
      const rx = Math.trunc(x1 + (h[0] * ux + h[1] * uy + h[2]) / w), ry = Math.trunc(y1 + (h[3] * ux + h[4] * uy + h[5]) / w);
      let inl = 0; const pairs = [];
      for (let i = 0; i < top.length; i++) if (mask.data[i]) { inl++; pairs.push([a0[2 * i], a0[2 * i + 1], a1[2 * i], a1[2 * i + 1]]); }
      out.inliers = inl; out.pairs = pairs;
      if (Math.hypot(rx - cx, ry - cy) <= tile) { out.x = rx; out.y = ry; }
      return out;
    } finally { for (const t of trash) t.delete(); }
  }
  async function locateImage(img) {
    const t0 = performance.now();
    const e = await embed(img); const t1 = performance.now();
    const cands = search(e); const t2 = performance.now();
    const g = queryGray(img); let best = null;
    try { for (const c of cands) { const r = refine(g, c); c.inliers = r.inliers; if (!best || r.inliers > best.inliers) best = { ...r, cand: c }; } }
    finally { g.delete(); }
    const t3 = performance.now();
    return { cands, best, ms: { embed: t1 - t0, search: t2 - t1, match: t3 - t2, total: t3 - t0 } };
  }
  async function locate() {
    if (!session || !state.query || state.busy) return;
    state.busy = true; const run = $("#run"); run.setAttribute("aria-busy", "true"); run.textContent = "Locating";
    try {
      const res = await locateImage(state.query); state.result = res;
      const b = res.best, g = toGps(b.x, b.y);
      $("#r-px").textContent = `${b.x}, ${b.y} px`;
      $("#r-gps").textContent = `${g[0].toFixed(5)}, ${g[1].toFixed(5)}`;
      $("#r-err").textContent = state.truth ? `${f1.format(pxMetres([b.x, b.y], state.truth))} m` : "no ground truth";
      $("#r-inl").textContent = `${b.inliers}`;
      $("#r-ms").textContent = `${nf.format(Math.round(res.ms.total))} ms`;
      if (state.truth) { const cell = $(`#ev-${state.frame}`); if (cell) cell.innerHTML = `<strong>${f1.format(pxMetres([b.x, b.y], state.truth))}</strong>`; }
      if (!state.startManual) state.start = [b.x, b.y];
      if (!state.target) state.target = defaultTarget([b.x, b.y]);
      plan(); drawMap(); drawMatches();
    } catch (e) {
      console.error(e); $("#r-err").textContent = "the match failed";
    } finally { state.busy = false; run.removeAttribute("aria-busy"); run.textContent = "Locate this frame"; }
  }
  function defaultTarget(from) { // the landmark furthest away, so the first route crosses campus
    let best = landmarks[0], d = -1;
    for (const l of landmarks) { const dd = Math.hypot(l.pixel[0] - from[0], l.pixel[1] - from[1]); if (dd > d) { d = dd; best = l; } }
    return best.pixel.slice();
  }
  async function runAll() {
    const btn = $("#run-all"); btn.disabled = true;
    const errs = [];
    for (let i = 0; i < evalData.rows.length; i++) {
      btn.textContent = `Running frame ${i + 1} of ${evalData.rows.length}`;
      const r = evalData.rows[i], img = await loadImage(`frames/${r.file}`);
      const res = await locateImage(img), b = res.best, err = pxMetres([b.x, b.y], r.truth); errs.push(err);
      $(`#ev-${i}`).innerHTML = `<strong>${f1.format(err)}</strong>`;
      await new Promise((ok) => setTimeout(ok, 0));
    }
    errs.sort((a, b) => a - b);
    const med = (errs[9] + errs[10]) / 2, under = errs.filter((e) => e < 25).length;
    btn.textContent = `Done here: median ${f1.format(med)} m, ${under} of 20 inside 25 m`; btn.disabled = false;
  }

  /* ---------- A* over the landmark graph (core/route_planner.py) ---------- */
  function plan() {
    if (!state.start || !state.target) { state.route = null; renderRoute(); return; }
    const nodes = landmarks.map((l) => l.pixel), n = nodes.length, S = n, G = n + 1;
    const pos = (k) => (k === S ? state.start : k === G ? state.target : nodes[k]);
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const edge = (a, b) => { const d = dist(a, b); return d > CONNECT ? null : d; };
    const directOK = edge(state.start, state.target) !== null && dist(state.start, state.target) <= CONNECT * 0.8;
    function* nbrs(k) {
      if (k === G) return;
      if (k === S) { for (let i = 0; i < n; i++) { const c = edge(state.start, nodes[i]); if (c !== null) yield [i, c]; } if (directOK) yield [G, dist(state.start, state.target)]; return; }
      for (let j = 0; j < n; j++) if (j !== k) { const c = edge(nodes[k], nodes[j]); if (c !== null) yield [j, c]; }
      const cg = edge(nodes[k], state.target); if (cg !== null) yield [G, cg];
    }
    const h = (k) => dist(pos(k), state.target);
    const open = [[h(S), 0, S, [S]]], best = new Map([[S, 0]]);
    let path = null;
    while (open.length) {
      open.sort((a, b) => a[0] - b[0]);
      const [, g, k, p] = open.shift();
      if (k === G) { path = p; break; }
      if (g > (best.get(k) ?? Infinity)) continue;
      for (const [j, c] of nbrs(k)) { const ng = g + c; if (ng < (best.get(j) ?? Infinity)) { best.set(j, ng); open.push([ng + h(j), ng, j, [...p, j]]); } }
    }
    state.route = (path || [S, G]).map((k) => ({ k, p: pos(k), lm: k < n ? landmarks[k] : null, kind: k === S ? "start" : k === G ? "target" : "landmark" }));
    state.routeFound = !!path;
    renderRoute();
  }
  function renderRoute() {
    const tb = $("#t-route tbody");
    if (!state.route) { tb.innerHTML = `<tr><td colspan="4" class="muted">Locate a frame, then click the map to set a target.</td></tr>`; return; }
    let total = 0;
    const kinds = { JUNCTION: "Junction", BUILDING: "Building", COURT: "Court", COMPLEX: "Complex" };
    const rows = state.route.map((w, i) => {
      const leg = i ? pxMetres(state.route[i - 1].p, w.p) : 0; total += leg;
      const name = w.kind === "start" ? (state.startManual ? "Start, set by click" : "Start, the drone's fix") : w.kind === "target" ? "Target" : w.lm.label;
      return `<tr><td class="num">${i}</td><td><strong>${name}</strong></td><td>${w.lm ? kinds[w.lm.type] || w.lm.type : "Point"}</td><td class="n">${i ? `${nf.format(Math.round(leg))} m` : ""}</td></tr>`;
    });
    rows.push(`<tr><td></td><td><strong>${state.routeFound ? "Route length" : "No landmark route; flying direct"}</strong></td><td></td><td class="n"><strong>${nf.format(Math.round(total))} m</strong></td></tr>`);
    tb.innerHTML = rows.join("");
  }

  /* ---------- drawing ---------- */
  function drawMap() {
    const c = $("#map"); if (!baseCanvases.cyan) return;
    const x = c.getContext("2d"), k = W / Math.max(1, c.clientWidth);
    x.drawImage(state.colour ? baseCanvases.true : baseCanvases.cyan, 0, 0);
    x.lineJoin = "miter"; x.font = `500 ${Math.round(12 * k)}px "IBM Plex Mono", monospace`;
    const line = state.colour ? "#ffffff" : ink.line;
    x.strokeStyle = line; x.fillStyle = line;
    if ($("#l-land").checked) {
      for (const l of landmarks) {
        const [lx, ly] = l.pixel; x.lineWidth = 1.5 * k;
        x.beginPath(); x.moveTo(lx - 6 * k, ly - 6 * k); x.lineTo(lx + 6 * k, ly + 6 * k); x.moveTo(lx + 6 * k, ly - 6 * k); x.lineTo(lx - 6 * k, ly + 6 * k); x.stroke();
        x.fillText(l.label, lx + 8 * k, ly - 6 * k);
      }
    }
    const res = state.result;
    if (res && $("#l-cand").checked) {
      res.cands.forEach((cd, i) => {
        const s = cd.tile; x.lineWidth = (cd === res.best.cand ? 2 : 1) * k; x.globalAlpha = cd === res.best.cand ? 1 : 0.7;
        x.strokeRect(cd.x - s / 2, cd.y - s / 2, s, s);
        x.fillText(String(i + 1), cd.x - s / 2 + 5 * k, cd.y - s / 2 + 15 * k);
      });
      x.globalAlpha = 1;
    }
    if (state.route && $("#l-route").checked) {
      x.lineWidth = 2.5 * k; x.setLineDash([]); x.beginPath();
      state.route.forEach((w, i) => (i ? x.lineTo(w.p[0], w.p[1]) : x.moveTo(w.p[0], w.p[1]))); x.stroke();
      for (const w of state.route.slice(1, -1)) { x.beginPath(); x.arc(w.p[0], w.p[1], 5 * k, 0, 7); x.fill(); x.fillText(w.lm.label, w.p[0] + 8 * k, w.p[1] + 16 * k); }
      if (state.target) { const [tx, ty] = state.target; x.lineWidth = 2 * k; x.strokeRect(tx - 9 * k, ty - 9 * k, 18 * k, 18 * k); x.fillText("target", tx + 13 * k, ty + 4 * k); }
    }
    if (state.truth && res) { const [tx, ty] = state.truth; x.lineWidth = 1.5 * k; x.beginPath(); x.moveTo(tx, ty - 12 * k); x.lineTo(tx + 12 * k, ty); x.lineTo(tx, ty + 12 * k); x.lineTo(tx - 12 * k, ty); x.closePath(); x.stroke(); x.fillText("truth", tx + 15 * k, ty - 10 * k); }
    if (res) {
      const { x: bx, y: by } = res.best; x.lineWidth = 2.5 * k;
      x.beginPath(); x.arc(bx, by, 16 * k, 0, 7); x.moveTo(bx - 28 * k, by); x.lineTo(bx + 28 * k, by); x.moveTo(bx, by - 28 * k); x.lineTo(bx, by + 28 * k); x.stroke();
      x.fillText("fix", bx + 20 * k, by + 22 * k);
    } else if (state.start && state.startManual) { const [sx, sy] = state.start; x.lineWidth = 2 * k; x.beginPath(); x.arc(sx, sy, 9 * k, 0, 7); x.stroke(); }
  }
  function drawMatches() {
    const c = $("#matches"), x = c.getContext("2d"), res = state.result;
    x.fillStyle = token("--color-paper-3"); x.fillRect(0, 0, c.width, c.height);
    x.fillStyle = ink.muted; x.font = `500 16px "IBM Plex Sans", sans-serif`;
    if (!res || !state.query) { x.fillText(session ? "Locating the frame" : "The matches appear once the model has loaded", 24, 40); return; }
    const b = res.best, [bx1, by1, bw, bh] = b.box, pad = 24, gap = 80;
    const side = Math.min((c.width - 2 * pad - gap) / 2, c.height - 2 * pad), s0 = side / bw, s1 = side / bh;
    const ox0 = pad, oy = (c.height - side) / 2, ox1 = pad + side + gap;
    const tmp = document.createElement("canvas"); tmp.width = bw; tmp.height = bh; const tx = tmp.getContext("2d");
    tx.drawImage(state.query, 0, 0, bw, bh); let d = tx.getImageData(0, 0, bw, bh); if (!state.colour) d = cyanotype(d); tx.putImageData(d, 0, 0);
    x.imageSmoothingEnabled = true; x.drawImage(tmp, ox0, oy, side, side);
    tx.drawImage(mapImg, bx1, by1, bw, bh, 0, 0, bw, bh); d = tx.getImageData(0, 0, bw, bh); if (!state.colour) d = cyanotype(d); tx.putImageData(d, 0, 0);
    x.drawImage(tmp, ox1, oy, side, side);
    x.strokeStyle = state.colour ? "#ffffff" : ink.line; x.lineWidth = 1; x.globalAlpha = 0.75;
    for (const [ax, ay, cx2, cy2] of b.pairs) { x.beginPath(); x.moveTo(ox0 + ax * s0, oy + ay * s1); x.lineTo(ox1 + cx2 * s0, oy + cy2 * s1); x.stroke(); }
    x.globalAlpha = 1; x.lineWidth = 1; x.strokeRect(ox0, oy, side, side); x.strokeRect(ox1, oy, side, side);
    x.fillStyle = ink.line; x.font = `500 14px "IBM Plex Sans", sans-serif`;
    x.fillText("Camera frame", ox0, oy - 8); x.fillText(`Map tile at ${bx1 + (bw >> 1)}, ${by1 + (bh >> 1)} px, ${bw} px wide`, ox1, oy - 8);
    x.fillText(`${b.inliers} inliers`, ox0 + side + 8, oy + side / 2);
  }

  boot().catch((e) => { console.error(e); $("#error").hidden = false; $("#error-text").textContent = "The map data did not load. Reload the page to try again."; $("#progress").hidden = true; });
})();
