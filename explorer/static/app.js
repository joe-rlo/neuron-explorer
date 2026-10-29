import { createMap } from "./map3d.js?v=79";
import { createScatter } from "./scatter.js?v=39";
import { t, tp, initI18n, applyTranslations, getLang } from "./i18n.js?v=1";

const $ = s => document.querySelector(s);
const api = async (p, body) => {
  const r = await fetch(p, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error(`${p.split("?")[0]} ${r.status}: ${(await r.text()).slice(0, 160)}`);
  return r.json();
};
initI18n($("#lang"));
addEventListener("unhandledrejection", e => { const el = document.querySelector("#status"); if (el) el.textContent = String(e.reason?.message || e.reason); });
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtDose = d => (d > 0 ? "+" : d < 0 ? "−" : "") + Math.abs(d);

let agentSelecting = false, agentMarked = false;  // Claude's selections and marks (see agentUi); declared early, openNeuron reads them on load
const state = { layer: null, units: {}, unitMap: {}, focus: null, unit: null, view: "map", chat: { mode: "add", dir: "writes", dose: { add: 0.8, mul: 2 } } };
const ADD_STEPS = [...Array.from({ length: 30 }, (_, i) => +((i + 1) * 0.05).toFixed(2)), 1.75, 2, 2.5, 3, 4, 5];
const MUL_STEPS = [...Array.from({ length: 9 }, (_, i) => i * 0.5), 5, 6, 8, 12, 16];

const index = await api("/api/index");
const arch = index.arch;
const tip = $("#tip");

// ---------- map ----------
const map = createMap($("#map"), $("#layer-labels"), arch, {
  onPick: (layer, n) => openNeuron(layer, n, "neuron", true, false),
  onPickHead: (layer, head) => openHead(layer, head),
  onPickLayer: l => { if (l !== state.layer) { layerSel.value = l; setLayer(l); } },
  onLayerMode: l => {
    $("#layer-back").classList.toggle("on", l !== null);
    $("#layer-back").textContent = l !== null ? t("map.allLayers", { l }) : "";
    if (l !== null && l !== state.layer) setLayer(l);
  },
  onHover: (p, x, y) => {
    if (!p) return hideTip();
    if (p.type === "head") {
      if (hoverKey !== `h/${p.layer}/${p.head}`) map.setHoverConnections(null);
      const key = `h/${p.layer}/${p.head}`, head = `L${p.layer} h${p.head}`;
      const body = () => {
        const h = hoverCache.get(key);
        const side = (sg, list) => list.slice(0, 3).map(c => `<div class="tl"><span class="sgn">${sg}</span><span class="faint">L${c.layer} N${c.n}</span>  ${esc(state.unitMap[c.layer]?.get(c.n)?.name || (c.lens || []).slice(0, 4).map(t => t.trim() || t).join("  "))}</div>`).join("");
        return h ? side("←", h.reads) + side("→", h.writes) : "";
      };
      showTipHTML(x, y, head, body());
      clearTimeout(hoverTimer);
      hoverKey = key;
      hoverTimer = setTimeout(async () => {
        if (!hoverCache.has(key)) hoverCache.set(key, await api(`/api/head?layer=${p.layer}&head=${p.head}`));
        if (hoverKey !== key) return;
        const h = hoverCache.get(key);
        if (!(state.focus && (state.focus.kind === "neuron" || state.focus.kind === "head")))
          map.setHoverHead(p.layer, p.head, headRoleLinks(h));
        showTipHTML(x, y, head, body());
      }, 120);
      return;
    }
    if (p.type === "block") {
      if (hoverKey) { hoverKey = null; map.setHoverConnections(null); }
      const full = p.full;
      return showTip(x, y, `L${p.layer}`, full ? t("map.fullAttention", { q: arch.attn_heads, kv: arch.kv_heads }) : t("map.linearAttention", { k: arch.linear_k_heads, v: arch.linear_v_heads }));
    }
    const u = state.unitMap[p.layer]?.get(p.n), key = `${p.layer}/${p.n}`;
    const head = `L${p.layer} N${p.n}${u ? `  ${t("map.mass", { z: u.z.toFixed(1) })}` : ""}`;
    const body = () => {
      const h = hoverCache.get(key), rows = [];
      if (u?.name) rows.push(`<div class="tname">${esc(u.name)}</div>`);
      if (u) rows.push(`<div class="twords">${esc(u.words.join("  "))}</div>`);
      if (h && h.steer) rows.push(`<div class="tl"><span class="sgn up">+</span>${esc(h.steer.join("  "))}</div><div class="tl"><span class="sgn down">−</span>${esc(h.steer_neg.join("  "))}</div>`);
      if (h) rows.push(`<div class="tl faintline"><span class="sgn up">+</span>${esc(h.lens.join("  "))}</div><div class="tl faintline"><span class="sgn down">−</span>${esc(h.lens_neg.join("  "))}</div>`);
      const nh = hoverCache.get(`nh/${key}`);
      if (nh) {
        const top = roleHeads(nh), into = top.filter(k => k.role === "o").slice(0, 3), from = top.filter(k => k.role !== "o").slice(0, 4);
        if (into.length) rows.push(`<div class="tl faintline"><span class="sgn">←</span>${into.map(k => `L${k.layer} h${k.head}`).join("  ")}</div>`);
        if (from.length) rows.push(`<div class="tl faintline"><span class="sgn">→</span>${from.map(k => `${k.role} L${k.layer} h${k.head}`).join("  ")}</div>`);
      }
      return rows.join("");
    };
    showTipHTML(x, y, head, body());
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(async () => {
      const [hv, nh] = await Promise.all([hoverCache.get(key) || api(`/api/hover?layer=${p.layer}&n=${p.n}`),
        hoverCache.get(`nh/${key}`) || api(`/api/neuron/heads?layer=${p.layer}&n=${p.n}`)]);
      hoverCache.set(key, hv); hoverCache.set(`nh/${key}`, nh);
      if (hoverKey !== key) return;
      // with a neuron or head open, hover only shows the tooltip; its links and web stay on screen
      if (!(state.focus && (state.focus.kind === "neuron" || state.focus.kind === "head"))) map.setHoverConnections({ layer: p.layer, n: p.n }, hoverCache.get(key).con);
      showTipHTML(x, y, head, body());
    }, 120);
    hoverKey = key;
  },
});
// a neuron's heads over all four roles (/api/neuron/heads roles), strongest first, each head once
const roleHeads = r => {
  const seen = new Set();
  return ["q", "k", "v", "o"].flatMap(role => (r.roles?.[role] || []).map(x => ({ ...x, role }))).filter(x => x.z >= 3)
    .sort((a, b) => b.z - a.z).filter(x => !seen.has(`${x.layer}/${x.head}`) && seen.add(`${x.layer}/${x.head}`));
};
function showTip(x, y, head, body) { showTipHTML(x, y, head, esc(body)); }
function showTipHTML(x, y, head, html) {
  tip.innerHTML = `<div class="lab">${esc(head)}</div>${html}`;
  tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(x + 14, innerWidth - r.width - 8)}px`;
  tip.style.top = `${Math.min(y + 14, innerHeight - r.height - 8)}px`;
}
function hideTip() {
  tip.style.display = "none";
  if (hoverKey) { hoverKey = null; clearTimeout(hoverTimer); map.setHoverConnections(null); }
}
let hoverTimer, hoverKey = null;
const hoverCache = new Map();
// tip lives in the embed view section; lift it to body so the map can use it too
document.body.appendChild(tip);

// ---------- sheet layout ----------
const layouts = {};
let wantLayout = null;
async function setLayout(kind) {
  wantLayout = kind;
  document.querySelectorAll("#layout-mode button").forEach(b => b.classList.toggle("on", b.dataset.k === kind));
  if (kind === "index") { map.setLayout(null); return; }
  if (!layouts[kind]) {
    const r = await api(`/api/layout?kind=${kind}`);
    if (wantLayout !== kind) return;
    if (!r.available) { document.querySelectorAll("#layout-mode button").forEach(b => b.classList.toggle("on", b.dataset.k === "index")); map.setLayout(null); return; }
    const bin = atob(r.cell), a = new Int16Array(bin.length / 2);
    for (let i = 0; i < a.length; i++) a[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    layouts[kind] = a;
  }
  map.setLayout(layouts[kind]);
}
document.querySelectorAll("#layout-mode button").forEach(b => b.onclick = () => setLayout(b.dataset.k));
setLayout("writes");

// connection web of the focused neuron: downstream cone forward, upstream cone backward
let webOn = true, webKey = null;
async function refreshWeb() {
  document.querySelectorAll("#web-mode button").forEach(b => b.classList.toggle("on", (b.dataset.w === "on") === webOn));
  const f = state.focus, first = Math.max(1, Math.min(24, Math.round(Number($("#web-z").value) || 5)));
  const k = Number(new URLSearchParams(location.search).get("webk")) || 2;
  const hops = Number(new URLSearchParams(location.search).get("webhops")) || 0;  // hops out each way; 0 = all the way
  if (!webOn || !f || f.kind !== "neuron") { webKey = null; state.webSum = null; map.setWeb(null); renderList(); return; }
  const key = `${f.layer}/${f.n}/${first}/${k}/${hops}`;
  if (key === webKey) return;
  webKey = key;
  const r = await fetch(`/api/web?layer=${f.layer}&n=${f.n}&first=${first}&k=${k}&hops=${hops}`);
  if (!r.ok || webKey !== key) return;
  state.webTitle = { k, n: r.headers.get("X-Web-Total") };
  $("#web-z").title = t("web.title", state.webTitle);
  map.setWeb(new Int16Array(await r.arrayBuffer()));
  const sum = await api(`/api/web/summary?layer=${f.layer}&n=${f.n}&first=${first}&k=${k}&hops=${hops}&top=24`);
  if (webKey !== key) return;
  state.webSum = { ...sum, layer: f.layer, n: f.n, links: r.headers.get("X-Web-Total") };
  renderList();
}
document.querySelectorAll("#web-mode button").forEach(b => b.onclick = () => { webOn = b.dataset.w === "on"; webKey = null; refreshWeb(); });
$("#web-z").onchange = () => refreshWeb();

// ---------- scatter ----------
const scatter = createScatter($("#scatter"), {
  onPick: i => { const d = scatter.data; if (d.kind[i] === "neuron") openNeuron(d.layer, d.neuron[i]); },
  onHover: async (i, x, y) => {
    if (i < 0) { scatterHover = null; return hideTip(); }
    const d = scatter.data, layer = d.layer, key = `${layer}/${d.row[i]}`;
    const head = `${d.kind[i] === "baseline" ? t("scatter.baseline") : (d.kind[i] === "random" ? t("scatter.random") + " " : "N") + d.neuron[i]} · ${d.prompt[i]} · ${fmtDose(d.dose[i])}`;
    scatterHover = key;
    if (rolloutCache.has(key)) return showTip(x, y, head, rolloutCache.get(key));
    showTip(x, y, head, "…");
    const r = await api(`/api/rollout?layer=${layer}&i=${d.row[i]}`);
    rolloutCache.set(key, r.text.slice(0, 500));
    if (scatterHover === key && state.view === "embed") showTip(x, y, head, rolloutCache.get(key));
  },
});
let scatterHover = null;
const rolloutCache = new Map();
document.querySelectorAll("#color-by button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#color-by button").forEach(x => x.classList.toggle("on", x === b));
  scatter.setColorBy(b.dataset.c);
});

// ---------- views / layer ----------
document.querySelectorAll("#views button").forEach(b => b.onclick = () => setView(b.dataset.view));
function setView(v) {
  state.view = v;
  document.querySelectorAll("#views button").forEach(x => x.classList.toggle("on", x.dataset.view === v));
  $("#map-view").classList.toggle("on", v === "map");
  $("#embed-view").classList.toggle("on", v === "embed");
  if (v === "embed") loadEmbeddings();
}
const layerSel = $("#layer");
for (let l = 0; l < arch.n_layers; l++) {
  const o = document.createElement("option"); o.value = l;
  o.textContent = `L${l}${index.layers[l] ? " ●" : ""}`;  // ● = has sweep rollouts
  layerSel.appendChild(o);
}
layerSel.onchange = () => { setLayer(+layerSel.value); map.enterLayer(+layerSel.value); };

async function setLayer(l) {
  state.layer = l; layerSel.value = l;
  if (scatter.data && scatter.data.layer !== l) { scatter.setData(null); hideTip(); }
  try { localStorage.setItem("ne.layer", l); } catch {}
  if (!state.units[l]) {
    if (!lensResults) $("#rows").innerHTML = `<li class="head">…</li>`;
    const us = index.layers[l] ? await api(`/api/units?layer=${l}`) : [];
    state.units[l] = us;
    state.unitMap[l] = new Map(us.filter(u => u.kind === "neuron").map(u => [u.n, u]));
    if (state.layer !== l) return;
  }
  if (state.units[l].length) map.setUnits(l, state.units[l]);
  renderList();
  if (state.view === "embed") loadEmbeddings();
}
async function loadEmbeddings() {
  const l = state.layer;
  $("#embed-empty").textContent = "…";
  const d = index.layers[l] ? await api(`/api/embeddings?layer=${l}`) : {};
  if (state.layer !== l) return;
  $("#embed-empty").textContent = d.x ? "" : "—";
  scatter.setData(d.x ? { ...d, layer: l } : null);
  scatter.select(state.layer, state.focus?.layer === state.layer ? state.focus.n : null);
}

// ---------- left panel: net (the open neuron's web, then the list) | run (text through the model) ----------
let leftTab = "net";
document.querySelectorAll("#ltabs button[data-lt]").forEach(b => b.onclick = () => showLeft(b.dataset.lt));
function showLeft(t) {
  if (t === leftTab) return;
  leftTab = t;
  document.querySelectorAll("#ltabs button[data-lt]").forEach(b => b.classList.toggle("on", b.dataset.lt === t));
  $("#lt-net").classList.toggle("on", t === "net"); $("#lt-run").classList.toggle("on", t === "run");
  $(`#ltabs button[data-lt="${t}"]`).classList.remove("fresh");
  syncLeftMap();
}
// a tab whose contents changed while hidden (Claude ran text, or searched) gets a dot until opened
function markLeftFresh(t) { if (t !== leftTab) $(`#ltabs button[data-lt="${t}"]`).classList.add("fresh"); }
// the map shows what the open tab shows: lens hits and the open neuron's web on net, what fired on run
function syncLeftMap() {
  lightHeads(state.inf.pos);
  if (leftTab === "run" && state.inf.text) { inferView(); showTokenPaths(state.inf.pos); return; }
  map.setMarks(leftTab === "net" && lensResults?.results ? lensResults.results : []);
  if (String(webKey).startsWith("tok/")) { webKey = null; refreshWeb(); }
}

// ---------- lens search ----------
let lensTimer, lensResults = null, fireResults = null;
$("#lens-q").addEventListener("input", e => {
  if (e.isTrusted) showLeft(state.qMode === "fires" ? "run" : "net");
  clearTimeout(lensTimer);
  lensTimer = setTimeout(runLensSearch, state.qMode === "fires" ? 700 : 250);
});
state.qMode = "lens";
document.querySelectorAll("#q-mode button").forEach(b => b.onclick = e => {
  state.qMode = b.dataset.q;
  document.querySelectorAll("#q-mode button").forEach(x => x.classList.toggle("on", x === b));
  syncPlaceholders();
  if (e.isTrusted) {
    showLeft(state.qMode === "fires" ? "run" : "net");
    if (state.qMode === "fires" && document.activeElement !== $("#inf-text")) $("#inf-text").focus();
  }
  runLensSearch();
});
// the paste box in the list drives the same search as the header box
$("#inf-text").addEventListener("input", () => {
  if (state.qMode !== "fires") $('#q-mode button[data-q="fires"]').click();  // typing here means fires mode
  $("#lens-q").value = $("#inf-text").value; $("#lens-q").dispatchEvent(new Event("input"));
});
$("#inf-minus").addEventListener("input", () => {
  if (state.qMode !== "fires" || $("#lens-q").value.trim().length < 2) return;
  clearTimeout(lensTimer); lensTimer = setTimeout(runLensSearch, 700);
});
$("#lens-q").addEventListener("input", () => { if (state.qMode === "fires" && document.activeElement === $("#lens-q")) $("#inf-text").value = $("#lens-q").value; });
function syncPlaceholders() {
  $("#lens-q").placeholder = state.qMode === "fires" ? t("header.searchText") : t("header.searchLens");
  $("#c-input").placeholder = state.chat.dir === "reads" ? t("chat.textToRead") : t("chat.who");
}
// fires: run text through the model once, then filter what fired without re-running
state.inf = { text: null, pos: -1, sign: "both" };
let qGen = 0;
const stillWanted = (g, mode, q) => g === qGen && state.qMode === mode && $("#lens-q").value.trim() === q;
const minusTexts = () => $("#inf-minus").value.split("\n").map(t => t.trim()).filter(Boolean);
async function runInfer(text, g) {
  markLeftFresh("run");
  const minus = minusTexts(), key = JSON.stringify([text, minus]);
  if (key !== state.inf.key) {
    $("#run-rows").innerHTML = `<li class="head">${minus.length ? `running with ${minus.length} minus text${minus.length > 1 ? "s" : ""}…` : t("results.running")}</li>`;
    const r = await api("/api/infer", { text, minus });
    if (!stillWanted(g, "fires", text) || JSON.stringify(minusTexts()) !== JSON.stringify(minus)) return;
    if (!r.ready) { $("#run-rows").innerHTML = `<li class="head">${esc(localizeStatus(state.statusText) || t("results.measuring"))}</li>`; return; }
    state.inf = { ...state.inf, text, key, pos: -1, tokens: r.tokens };
    $("#inf-toks").innerHTML = r.tokens.map((t, i) => `<button data-i="${r.pos[i]}">${esc(t)}</button>`).join("");
  }
  await inferView(g);
}
// hovering a token previews it: its strongest neurons marked on the map, and the direct links among them as the web;
// clicking pins it; leaving the tokens goes back to the pinned token (or none) and the open neuron's web
let tokHover = null, tokTimer;
async function showTokenPaths(pos) {
  if (pos < 0) { webKey = null; return refreshWeb(); }
  if (leftTab !== "run") return;
  const r = await fetch(`/api/infer/paths?pos=${pos}&min=${$("#inf-min").value || 1}`);
  if (!r.ok || (tokHover ?? state.inf.pos) !== pos) return;
  webKey = `tok/${pos}`;
  map.setWeb(new Int16Array(await r.arrayBuffer()));
}
function previewToken(pos) {
  const keep = state.inf.pos;
  state.inf.pos = pos; inferView(); state.inf.pos = keep;
  showTokenPaths(pos); lightHeads(pos);
}
// on run, the head lines light by how much each head writes at the token (pinned or hovered), and with a head open
// the token chips shade by where that head reads from there
let headsGen = 0;
async function lightHeads(pos) {
  const g = ++headsGen, f = state.focus;
  const on = leftTab === "run" && state.inf.text && pos >= 0;
  const r = on ? await api(`/api/infer/heads?pos=${pos}`) : null;
  if (g !== headsGen) return;
  if (r?.layers) map.setHeadLevels(r.layers.map(row => row.map(v => Math.max(0, (v - 0.15) / 0.6))));
  else if (f?.kind === "neuron" && state.nheads?.layer === f.layer && state.nheads.n === f.n) {
    const lv = [];
    for (const [l, zs] of Object.entries(state.nheads.layers)) lv[+l] = zs.map(z => Math.max(0, Math.min(1, (z - 2.5) / 5)));
    map.setHeadLevels(lv);
  } else map.setHeadLevels(null);
  shadeTokens(on ? pos : -1);
}
let shadeGen = 0;
async function shadeTokens(pos) {
  const g = ++shadeGen, f = state.focus, btns = [...document.querySelectorAll("#inf-toks button")];
  const r = pos >= 0 && f?.kind === "head" ? await api(`/api/head/row?layer=${f.layer}&head=${f.n}&pos=${pos}`) : null;
  if (g !== shadeGen) return;
  btns.forEach((b, i) => {
    const w = r?.units?.[i];
    b.style.background = w == null || b.classList.contains("on") ? "" :
      `rgba(${w >= 0 ? "255,181,71" : "255,79,154"},${Math.min(0.85, Math.abs(w) * 1.6).toFixed(3)})`;
    b.style.color = w != null && Math.abs(w) > 0.3 && !b.classList.contains("on") ? "#000" : "";
  });
}
$("#inf-toks").addEventListener("pointerover", e => {
  const b = e.target.closest("button"); if (!b || tokHover === +b.dataset.i) return;
  tokHover = +b.dataset.i; clearTimeout(tokTimer);
  tokTimer = setTimeout(() => previewToken(tokHover), 90);
});
$("#inf-toks").addEventListener("pointerleave", () => {
  clearTimeout(tokTimer); tokHover = null;
  inferView(); showTokenPaths(state.inf.pos); lightHeads(state.inf.pos);
});
$("#inf-toks").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  state.inf.pos = state.inf.pos === +b.dataset.i ? -1 : +b.dataset.i;
  document.querySelectorAll("#inf-toks button").forEach(x => x.classList.toggle("on", +x.dataset.i === state.inf.pos));
  inferView(); showTokenPaths(state.inf.pos); lightHeads(state.inf.pos);
});
document.querySelectorAll("#inf-sign button").forEach(b => b.onclick = () => {
  state.inf.sign = b.dataset.s;
  document.querySelectorAll("#inf-sign button").forEach(x => x.classList.toggle("on", x === b));
  inferView();
});
let infTimer;
for (const id of ["#inf-lo", "#inf-hi", "#inf-min", "#inf-q"]) $(id).addEventListener("input", () => { clearTimeout(infTimer); infTimer = setTimeout(inferView, 200); });
let viewGen = 0;
async function inferView(g = qGen) {
  const f = state.inf, mine = ++viewGen;
  if (!f.text) return;
  const qs = new URLSearchParams({ pos: f.pos, sign: f.sign, lo: $("#inf-lo").value || 0, hi: $("#inf-hi").value || 99,
    min: $("#inf-min").value || 0, q: $("#inf-q").value, limit: 300 });
  const r = await api(`/api/infer/view?${qs}`);
  if (mine !== viewGen || g !== qGen || state.qMode !== "fires") return;
  const marks = [];
  if (r.map) {
    const bin = atob(r.map), u16 = new Uint16Array(bin.length / 2);
    for (let i = 0; i < u16.length; i++) u16[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const N = r.shape[1];
    for (let i = 0; i < u16.length; i++) {
      if (!u16[i]) continue;
      const v = halfToFloat(u16[i]);
      if (v) marks.push({ layer: Math.floor(i / N), n: i % N, sign: v < 0 ? "-" : "+", w: Math.min(1, 0.25 + Math.abs(v) / 4) });
    }
  }
  if (leftTab === "run") map.setMarks(marks.length > 40000 ? marks.sort((a, b) => b.w - a.w).slice(0, 40000) : marks);
  fireResults = { ...r, fires: true, ready: true };
  renderResults(fireResults, $("#run-rows"));
}
function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

async function runLensSearch() {
  const q = $("#lens-q").value.trim(), g = ++qGen;
  if (q.length < 2) {
    // an emptied box clears both: lens hits, and the last run with its token chips, pinned token and token web
    lensResults = null; fireResults = null;
    state.inf = { ...state.inf, text: null, key: null, pos: -1, tokens: null };
    $("#inf-toks").innerHTML = ""; $("#run-rows").innerHTML = "";
    syncLeftMap(); renderList(); return;
  }
  if (state.qMode === "fires") return runInfer(q, g);
  const r = await api(`/api/search?q=${encodeURIComponent(q)}&limit=400`);
  if (!stillWanted(g, "lens", q)) return;
  lensResults = r;
  if (leftTab === "net") map.setMarks(r.results || []); else markLeftFresh("net");
  if ($("#list").classList.contains("hidden")) setListHidden(false);
  renderList();
}

// ---------- list ----------
$("#filter").oninput = renderList;
$("#show-random").onchange = renderList;
$("#sort").onchange = renderList;
// what the open neuron's web says: lens tokens pooled over its cone (see /api/web/summary); click a token to search it
function webSummaryHTML() {
  const w = state.webSum, f = state.focus;
  if (!w || !f || f.kind !== "neuron" || w.layer !== f.layer || w.n !== f.n) return "";
  const chips = list => list.length ? list.map(x => `<span class="chip" data-q="${esc(x.tok)}" title="${esc(t("web.chipTitle", { n: x.n, w: x.w }))}">${esc(x.tok)}<i>${x.n}</i></span>`).join("")
    : `<span class="faint">${t("web.nothing")}</span>`;
  const src = w.src === "steer" ? t("web.srcSteer") : t("web.srcLogit");
  return `<div class="head">${t("web.head", { layer: w.layer, n: w.n, links: w.links, src })}</div>
    <div class="wsum"><div class="lab"><span class="sw up"></span>${t("web.fedBy", { n: w.up.neurons })}</div><div class="chips">${chips(w.up.fed_by)}</div></div>
    <div class="wsum"><div class="lab"><span class="sw down"></span>${t("web.heldBy")}</div><div class="chips">${chips(w.up.held_by || [])}</div></div>
    <div class="wsum"><div class="lab"><span class="sw up"></span>${t("web.toward", { n: w.down.neurons })}</div><div class="chips">${chips(w.down.toward)}</div></div>
    <div class="wsum"><div class="lab"><span class="sw down"></span>${t("web.away")}</div><div class="chips">${chips(w.down.away)}</div></div>`;
}
// the open neuron's heads by role: later heads whose query, key or value reads it, and heads whose output it reads
function nheadsHTML() {
  const H = state.nheads, f = state.focus;
  if (!H?.roles || !f || f.kind !== "neuron" || H.layer !== f.layer || H.n !== f.n) return "";
  const name = x => `L${x.layer} h${x.heads.length > 1 ? `${x.heads[0]}–${x.heads.at(-1)}` : x.head}`;
  const row = r => `<div class="lrow"><span class="sgn">${r}</span><div class="chips">${(H.roles[r] || []).filter(x => x.z >= 2).map(x =>
    `<span class="chip" data-l="${x.layer}" data-h="${x.head}">${name(x)}<i>${x.z}</i></span>`).join("")}</div></div>`;
  return `<div class="wsum"><div class="lab">${t("lens.heads")}</div>${["q", "k", "v", "o"].map(row).join("")}</div>`;
}
// the open neuron's own lenses (what it writes), above its web (what its cone does)
function lensHTML() {
  const L = state.lens, f = state.focus;
  if (!L || !f || f.kind !== "neuron" || L.layer !== f.layer || L.n !== f.n) return "";
  const chips = ts => ts.slice(0, 12).map(t => `<span class="chip" data-q="${esc(t.trim() || t)}">${esc(t)}</span>`).join("");
  const pair = (up, down) => `<div class="lrow"><span class="sgn up">+</span><div class="chips">${chips(up)}</div></div>
    <div class="lrow"><span class="sgn down">−</span><div class="chips">${chips(down)}</div></div>`;
  return (L.steer_up ? `<div class="wsum"><div class="lab">${t("lens.steer", { up: L.kl_up, down: L.kl_down })}</div>${pair(L.steer_up, L.steer_down)}</div>` : "")
    + `<div class="wsum"><div class="lab">${t("lens.logit")}</div>${pair(L.up, L.down)}</div>`;
}
// a head's strongest neurons in each role, as map links
const ROLE_COL = { q: "#73ff8c", k: "#b399ff", v: "#66d9ff", o: "#66d9ff" };
const headRoleLinks = h => [["q", h.q], ["k", h.k], ["v", h.reads], ["o", h.writes]]
  .flatMap(([role, list]) => (list || []).slice(0, 5).map(c => ({ ...c, role, side: role === "o" ? "down" : "up" })));
// the open head by role: the neurons feeding its query, key and value (earlier) and reading its output (later), each
// with its own lens tokens
function headHTML() {
  const H = state.headInfo, f = state.focus;
  if (!H || !f || f.kind !== "head" || H.layer !== f.layer || H.head !== f.n) return "";
  const line = c => { const nm = state.unitMap[c.layer]?.get(c.n)?.name;
    return `<div class="nline" data-nl="${c.layer}" data-nn="${c.n}"><span class="id">L${c.layer} N${c.n}</span>`
      + `<span class="w">${nm ? `<b>${esc(nm)}</b> ` : ""}${esc((c.lens || []).slice(0, 4).map(t => t.trim() || t).join("  "))}</span><span class="z">${c.z.toFixed(0)}</span></div>`; };
  const role = (r, list) => `<div class="wsum"><div class="lab"><span class="sw" style="background:${ROLE_COL[r]}"></span>${r}</div>${(list || []).slice(0, 10).map(line).join("")}</div>`;
  return role("q", H.q) + role("k", H.k) + role("v", H.reads) + role("o", H.writes);
}
function renderList() {
  $("#net").innerHTML = headHTML() || lensHTML() + webSummaryHTML() + nheadsHTML();
  if (lensResults) return renderResults(lensResults, $("#rows"));
  const q = $("#filter").value.trim().toLowerCase(), showR = $("#show-random").checked;
  let units = (state.units[state.layer] || []).filter(u => (showR || u.kind === "neuron") &&
    (!q || u.words.some(w => w.includes(q)) || u.name.toLowerCase().includes(q) || String(u.n) === q));
  // window: neurons with a therapeutic index for the chat's measure first, highest index first
  const mName = currentMeasure()?.name, byWin = $("#sort").value === "window";
  const wi = u => u.kind === "neuron" && mName ? windowOf(state.layer, u.n, mName)?.index ?? null : null;
  if (byWin) units = units.map(u => [u, wi(u)]).sort((a, b) => (b[1] ?? -Infinity) - (a[1] ?? -Infinity)).map(x => x[0]);
  // item selectivity / category d' from the chosen stimulus set (fetched per layer, then the list redraws)
  const tk = $("#sort").value === "si" ? "si" : $("#sort").value === "dprime" ? "d" : null, ts = tk && tuneScores(state.layer);
  const tv = u => u.kind === "neuron" && ts ? ts[tk][u.n] : null;
  // ranked over every neuron of the layer, not only the swept ones: the swept unit's words where there is one, else
  // the neuron's lens tokens
  if (tk && ts) units = ts[tk === "si" ? "top_si" : "top_d"].map(r => state.unitMap[state.layer]?.get(r.n) || { kind: "neuron", n: r.n, name: r.name, words: r.lens, z: 0 })
    .filter(u => !q || u.words.some(w => w.toLowerCase().includes(q)) || u.name.toLowerCase().includes(q) || String(u.n) === q);
  else if (tk) units = [];
  const rnd = (state.units[state.layer] || []).filter(u => u.kind === "random").map(u => u.z);
  const floor = rnd.length ? Math.max(...rnd) : -Infinity;
  const ol = $("#rows");
  if (tk && !tuneSet()?.ready) { ol.innerHTML = `<li class="head">run a stimulus set on a neuron's Tuning tab to sort by ${tk === "si" ? "item selectivity" : "category d'"}</li>`; return; }
  ol.innerHTML = (tk && ts ? `<li class="head">${tk === "si" ? "item selectivity" : `d' for ${esc(ts.category)}`} · ${esc(tuneSet().name)} · all neurons of L${state.layer}</li>` : "") + units.slice(0, 600).map(u => `<li data-n="${u.n}" data-kind="${u.kind}" class="${u.kind}${u.z <= floor ? " below" : ""}${state.focus && state.focus.n === u.n && state.focus.layer === state.layer && u.kind === "neuron" ? " sel" : ""}">
    <span class="n">${u.kind === "random" ? "r" : "N"}${u.n}</span><span class="z">${byWin ? (wi(u) == null ? "—" : num(wi(u))) : tk ? (tv(u) == null ? "—" : num(tv(u))) : u.z.toFixed(1)}</span>
    <span class="w">${u.name ? `<b>${esc(u.name)}</b> ` : ""}${esc(u.words.join(" "))}</span></li>`).join("");
}
// lens hits (net tab) or what fired (run tab)
function renderResults(r, ol) {
  if (!r.ready) { ol.innerHTML = `<li class="head">${esc(localizeStatus(state.statusText) || t("results.indexing"))}</li>`; return; }
  const rows = (r.results || []).map(x => `<li class="res${x.swept === false ? " unswept" : ""}" data-l="${x.layer}" data-n="${x.n}" data-kind="neuron">
    <span class="n">L${x.layer} N${x.n}</span><span class="sg ${x.sign === "+" ? "up" : "down"}">${r.fires ? `${x.sign === "+" ? "" : "−"}${x.score.toFixed(1)}×` : `${x.sign === "+" ? "+" : "−"}${x.rank + 1}`}</span>
    <span class="w">${x.main != null ? `<span class="raw" title="score on the text · mean score on the minus texts">${x.main.toFixed(1)} − ${x.minus.toFixed(1)}</span>` : ""}${r.fires ? `<span class="tk">"${esc(x.token)}"</span>` : ""}${x.name ? `<b>${esc(x.name)}</b> ` : ""}${esc(x.lens.join(" "))}</span></li>`).join("");
  const head = r.contrast ? `${r.total ?? 0} neurons pass · contrast = score on the text − mean score on ${r.contrast} minus text${r.contrast > 1 ? "s" : ""}`
    : r.fires ? t("results.pass", { total: r.total ?? 0 }) : t("results.summary", { total: r.total ?? 0, tokens: r.tokens ?? 0 });
  // both signs are strong responses; the sign is only the direction
  const why = r.fires ? "−2× is as strong as +2×: the neuron fires negatively instead. The sign shows direction, not strength."
    : "+n: the token ranks nth among what this neuron pushes toward; −n: nth among what it pushes away from. Both mean a strong link; the sign shows direction.";
  ol.innerHTML = `<li class="head">${head}<div class="why">${why}</div></li>` + rows;
}
// a head chip marks its head on the map while hovered
$("#net").addEventListener("mouseover", e => { const c = e.target.closest(".chip[data-h]"); map.hoverHead(c ? +c.dataset.l : null, c ? +c.dataset.h : null); });
$("#net").addEventListener("mouseleave", () => map.hoverHead(null));
// a web-summary token searches the lens for it
$("#net").addEventListener("click", e => {
  const nl = e.target.closest(".nline"); if (nl) return goToNeuron(+nl.dataset.nl, +nl.dataset.nn, "neuron");
  const chip = e.target.closest(".chip"); if (!chip) return;
  if (chip.dataset.h) return openHead(+chip.dataset.l, +chip.dataset.h);

  if (state.qMode !== "lens") $('#q-mode button[data-q="lens"]').click();
  $("#lens-q").value = chip.dataset.q; $("#lens-q").dispatchEvent(new Event("input"));
});
for (const ol of [$("#rows"), $("#run-rows")]) {
  ol.addEventListener("click", e => {
    const li = e.target.closest("li"); if (!li || !li.dataset.n) return;
    goToNeuron(li.dataset.l ? +li.dataset.l : state.layer, +li.dataset.n, li.dataset.kind);
  });
  ol.addEventListener("pointerover", e => {
    const li = e.target.closest("li"); if (li && li.dataset.kind === "neuron") map.hover(li.dataset.l ? +li.dataset.l : state.layer, +li.dataset.n);
  });
  ol.addEventListener("pointerleave", () => map.hover(null));
}

// ---------- jump / routing ----------
$("#jump").addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const m = e.target.value.match(/L?\s*(\d+)\D+N?\s*(\d+)/i);
  if (m) openNeuron(+m[1], +m[2]);
});
addEventListener("hashchange", fromHash);
// what's open is kept in this browser (a reload reopens it) and never in the address bar, where every neuron would
// become a browser-history entry; a link with #L19/N7176 (a neuron) or #L19/h2 (an attention head) opens that
function saveFocus() {
  const f = state.focus;
  try { localStorage.setItem("explorer.focus", JSON.stringify(f && (f.kind === "neuron" || f.kind === "head") ? f : null)); } catch {}
}
function restoreFocus() {
  let f = null; try { f = JSON.parse(localStorage.getItem("explorer.focus") || "null"); } catch {}
  if (!f) return;
  if (f.kind === "head") openHead(f.layer, f.n); else openNeuron(f.layer, f.n, "neuron", false);
}
function fromHash() {
  const m = location.hash.match(/^#L(\d+)\/(N|h)(\d+)/);
  if (!m) return;
  const kind = m[2] === "h" ? "head" : "neuron";
  history.replaceState(null, "", location.pathname + location.search);  // the link is used up; later picks are remembered
  if (isFocus(+m[1], +m[3], kind)) return;
  if (kind === "head") openHead(+m[1], +m[3]); else openNeuron(+m[1], +m[3], "neuron", true);
}

// ---------- panel collapse ----------
function updateInsets() {
  const listShown = !$("#list").classList.contains("hidden");
  const focusShown = focusEl.classList.contains("on") && !focusEl.classList.contains("collapsed");
  map.setInsets(listShown ? $("#list").getBoundingClientRect().width + 24 : 0, focusShown ? focusEl.getBoundingClientRect().width + 24 : 0);
}
$("#layer-back").onclick = () => map.exitLayer();
// panel widths follow the window (vw-based), so window resizes and browser zoom must re-measure them
addEventListener("resize", () => updateInsets());
function setListHidden(h) {
  document.body.classList.toggle("list-hidden", h);
  $("#list").classList.toggle("hidden", h); $("#list-tab").classList.toggle("on", h); updateInsets();
}
function setFocusCollapsed(c) {
  focusEl.classList.toggle("collapsed", c);
  syncEdge(); updateInsets();
}
// the right edge handle: brings back a collapsed panel, or opens Claude when no panel is up
function syncEdge() {
  const on = focusEl.classList.contains("on"), col = focusEl.classList.contains("collapsed"), f = state.focus;
  const t = $("#focus-tab");
  t.classList.toggle("on", !on || col);
  t.textContent = on && col && f && f.kind !== "head" ? `‹ ${f.kind === "random" ? "r" : "N"}${f.n}` : "‹ claude";
}
$("#list-hide").onclick = () => setListHidden(true);
$("#list-tab").onclick = () => setListHidden(false);
$("#f-collapse").onclick = () => setFocusCollapsed(true);
$("#focus-tab").onclick = () => focusEl.classList.contains("on") ? setFocusCollapsed(false) : openClaude();

// ---------- focus panel ----------
const focusEl = $("#focus");
// closing the neuron keeps the panel up when Claude, Lab or Learn is the open tab
$("#f-close").onclick = () => { flushName(); if (running) running.abort(); state.focus = null; $("#list").classList.remove("has-focus");
  map.selectHead(null); state.headInfo = null; state.nheads = null; $("#f-name").style.display = ""; lightHeads(state.inf.pos);
  if (["claude", "lab", "learn"].includes(curTab())) syncTabs(); else focusEl.classList.remove("on");
  setFocusCollapsed(false); map.select(null); refreshWeb(); renderAgentActions(); scatter.select(null, null); saveFocus(); renderList(); };
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));
const tabBtn = t => $(`#tabs button[data-tab="${t}"]`);
const curTab = () => $("#tabs button.on")?.dataset.tab;
function setTab(t) {
  document.querySelectorAll("#tabs button[data-tab]").forEach(x => x.classList.toggle("on", x.dataset.tab === t));
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x.id === `tab-${t}`));
  focusEl.classList.toggle("on-claude", t === "claude");
  tabBtn(t).classList.remove("fresh");
  if (t === "claude") { const th = $("#a-thread"); th.scrollTop = th.scrollHeight; }
  if (t === "lab") renderLab();
  if (t === "learn") renderLearn();
}
// a tab that changed while another was showing gets a dot until opened
function markFresh(t) { if (curTab() !== t) tabBtn(t).classList.add("fresh"); }
// neuron tabs show while a neuron is open (Chat for real neurons only); Learn, Lab and Claude always
function syncTabs() {
  const f = state.focus?.kind === "head" ? null : state.focus;
  for (const t of ["examples", "details"]) tabBtn(t).style.display = f ? "" : "none";
  tabBtn("tuning").style.display = f && f.kind === "neuron" ? "" : "none";
  tabBtn("chat").style.display = f && f.kind === "neuron" ? "" : "none";
  if (tabBtn(curTab()).style.display === "none") setTab(f ? "examples" : "claude");
}
function openClaude() {
  focusEl.classList.add("on"); syncTabs(); setTab("claude"); setFocusCollapsed(false); $("#a-input").focus();
}

// opening a neuron on another layer moves the layer menu with it; a brief notice under the menu says so
let jumpTimer;
function jumpNotice(l) {
  const el = $("#jump-note"), r = layerSel.getBoundingClientRect();
  el.textContent = `Jumped to L${l}`; el.style.left = `${r.left}px`; el.style.top = `${r.bottom + 6}px`;
  el.classList.add("on"); clearTimeout(jumpTimer); jumpTimer = setTimeout(() => el.classList.remove("on"), 2400);
}
async function openNeuron(layer, n, kind = "neuron", push = true, fly = true) {
  if (!agentSelecting) clearAgentMarks();
  flushName();
  if (running) running.abort();
  state.focus = { layer, n, kind }; state.unit = null;
  map.selectHead(null); state.headInfo = null; state.nheads = null; $("#f-name").style.display = ""; lightHeads(state.inf.pos);
  if (map.layerMode !== null && map.layerMode !== layer) map.enterLayer(layer);
  if (push && kind === "neuron") saveFocus();
  focusEl.classList.add("on"); $("#list").classList.add("has-focus");
  setFocusCollapsed(false);
  $("#f-id").textContent = kind === "random" ? t("focus.idRandom", { layer, n }) : `L${layer} N${n}`;
  $("#f-rank").textContent = "";
  setNameField(state.unitMap[layer]?.get(n)?.name || "");
  $("#f-name").disabled = kind !== "neuron";
  $("#tab-examples").innerHTML = ""; $("#tab-details").innerHTML = "";
  state.thread = []; renderThread(); $("#c-reads").innerHTML = ""; $("#c-lens").innerHTML = "";
  syncTabs(); renderSet();
  if (layer !== state.layer) { if (state.layer != null) jumpNotice(layer); await setLayer(layer); }
  if (!isFocus(layer, n, kind)) return;
  setNameField(state.unitMap[layer]?.get(n)?.name || "");
  renderList();
  if (kind === "neuron") { map.select(layer, n, fly); scatter.select(layer, n); loadNeuronHeads(layer, n); }
  refreshWeb(); renderAgentActions();
  const u = await api(`/api/unit?layer=${layer}&n=${n}&kind=${kind}`);
  if (!isFocus(layer, n, kind)) return;
  state.unit = u;
  setNameField(u.name || "");
  if (u.stats) $("#f-rank").textContent = t("focus.rank", { rank: u.rank ?? "—", n: u.n_ranked, mass: (u.stats.mass ?? u.stats.z).toFixed(1), nullPct: (u.stats.null_pct * 100).toFixed(0) });
  renderExamples(u);
  renderDetails(u);
  if (kind === "neuron") renderTuning();
  const hasRollouts = !!(u.rollouts && u.rollouts.length);
  document.querySelector('#tabs button[data-tab="examples"]').style.display = hasRollouts ? "" : "none";
  if (!hasRollouts && document.querySelector("#tabs button.on")?.dataset.tab === "examples") setTab("details");
  if (kind === "neuron") loadLive(layer, n);
}
// the heads linked to the open neuron light up on their slabs (see /api/neuron/heads)
async function loadNeuronHeads(layer, n) {
  state.nheads = null; lightHeads(tokHover ?? state.inf.pos);
  const r = await api(`/api/neuron/heads?layer=${layer}&n=${n}`);
  if (!isFocus(layer, n, "neuron")) return;
  state.nheads = { ...r, layer, n }; lightHeads(tokHover ?? state.inf.pos); renderList(); renderConnections();
  const want = Object.values(r.roles || {}).flat().filter(x => x.z >= 2 && !hoverCache.has(`h/${x.layer}/${x.head}`));
  await Promise.all(want.map(async x => hoverCache.set(`h/${x.layer}/${x.head}`, await api(`/api/head?layer=${x.layer}&head=${x.head}`))));
  if (isFocus(layer, n, "neuron")) renderConnections();
  map.setNeuronHeads(roleHeads(r).slice(0, 8));
}

// a head opens in the list panel: its id up top, what it reads and writes in net, and on run the text shaded by
// where it reads from; the neuron panel steps aside unless Claude is showing
async function openHead(layer, head) {
  if (!agentSelecting) clearAgentMarks();
  flushName();
  if (running) running.abort();
  state.focus = { layer, n: head, kind: "head" }; state.unit = null;
  saveFocus();
  map.select(null); scatter.select(null, null); map.selectHead(layer, head);
  $("#list").classList.add("has-focus");
  if ($("#list").classList.contains("hidden")) setListHidden(false);
  $("#f-id").textContent = `L${layer} h${head}`; $("#f-rank").textContent = "";
  $("#f-name").style.display = "none";
  if (curTab() === "claude") syncTabs(); else { focusEl.classList.remove("on"); syncTabs(); }
  renderSet();
  syncEdge(); updateInsets();
  refreshWeb(); renderAgentActions();
  shadeTokens(tokHover ?? state.inf.pos);
  lightHeads(tokHover ?? state.inf.pos);
  const r = await api(`/api/head?layer=${layer}&head=${head}`);
  if (!isFocus(layer, head, "head")) return;
  state.headInfo = r; hoverCache.set(`h/${layer}/${head}`, r);
  map.setHeadLinks(layer, head, headRoleLinks(r));
  renderList();
}

const isFocus = (layer, n, kind) => state.focus && state.focus.layer === layer && state.focus.n === n && state.focus.kind === kind;

function renderExamples(u) {
  if (!u.rollouts || !u.rollouts.length) { $("#tab-examples").innerHTML = `<div class="slot">—</div>`; return; }
  const prompts = Object.keys(u.prompts).filter(p => u.rollouts.some(r => r.prompt === p));
  const count = {};
  for (const r of u.rollouts) count[r.dose] = (count[r.dose] || 0) + 1;
  // doses shared by several prompts get columns; a prompt's one-off doses span its empty cells
  const doses = Object.keys(count).map(Number).filter(d => count[d] > 1).sort((a, b) => a - b);
  const relOf = d => u.rollouts.find(r => r.dose === d).rel_push.toFixed(2);
  const cell = r => `<div class="cell${r.finish === "length" ? " capped" : ""}">${esc(r.text)}</div>`;
  let h = `<div class="grid" style="grid-template-columns: 66px repeat(${1 + doses.length}, minmax(0,1fr))"><div class="h"></div><div class="h">0</div>`;
  for (const d of doses) h += `<div class="h ${d > 0 ? "up" : "down"}">${fmtDose(d)}<span class="dim"> · ${relOf(d)}</span></div>`;
  for (const p of prompts) {
    const b = u.baselines.find(r => r.prompt === p);
    h += `<div class="h">${p}</div><div class="cell base${b?.finish === "length" ? " capped" : ""}">${esc(b?.text ?? "")}</div>`;
    const mine = u.rollouts.filter(r => r.prompt === p);
    const shared = doses.map(d => mine.find(r => r.dose === d));
    const extra = mine.filter(r => !doses.includes(r.dose));
    if (shared.some(Boolean)) h += shared.map(r => r ? cell(r) : "<div></div>").join("");
    else if (extra.length) {
      const span = Math.max(1, Math.floor(doses.length / extra.length));
      h += extra.map((r, i) => `<div class="cell${r.finish === "length" ? " capped" : ""}" style="grid-column: span ${i === extra.length - 1 ? doses.length - span * (extra.length - 1) : span}"><span class="h ${r.dose > 0 ? "up" : "down"}">${fmtDose(r.dose)} · ${r.rel_push.toFixed(2)}</span>  ${esc(r.text)}</div>`).join("");
    } else h += doses.map(() => "<div></div>").join("");
  }
  $("#tab-examples").innerHTML = h + "</div>";
}
$("#tab-examples").addEventListener("click", e => { const c = e.target.closest(".cell"); if (c) c.classList.toggle("open"); });

function renderDetails(u) {
  const s = u.stats;
  let h = "";
  if (s) {
    h += `<div class="lab">${t("details.words")}</div><div class="crumbs">` + s.words.map(([w, c, z]) =>
      `<span class="tok" title="${esc(t("details.rollouts", { n: c, z: z.toFixed(1), prompts: (s.per_prompt[w] || []).join(" ") }))}">${esc(w)}</span>`).join("") + `</div>`;
    h += `<table class="dtable"><tr><th class="num">${t("details.dose")}</th><th class="num">${t("details.push")}</th><th class="num">${t("details.distinct")}</th><th class="num">${t("details.capped")}</th><th class="num">${t("details.logprob")}</th></tr>` +
      Object.entries(s.dose).map(([d, v]) => `<tr><td class="num">${d}</td><td class="num">${v.rel_push.toFixed(2)}</td><td class="num">${v.distinct4.toFixed(2)}</td><td class="num">${(v.capped * 100).toFixed(0)}%</td><td class="num">${v.logprob.toFixed(2)}</td></tr>`).join("") + `</table>`;
  }
  if (u.kind === "neuron") h += `<div class="fg-link"><button type="button" id="d-guide" class="dim" title="this neuron's field guide card: your hypothesis and notes with everything run on it">field guide card ›</button></div>`;
  h += `<div id="d-window"></div><div id="d-results"></div><div id="d-lens" class="slot">—</div><div class="lab" style="margin-top:18px">${t("details.connections")}</div><div id="d-con" class="slot">—</div>`;
  $("#tab-details").innerHTML = h;
  renderWindow();
}

async function loadLive(layer, n) {
  $("#d-lens") && ($("#d-lens").textContent = "…");
  $("#d-con") && ($("#d-con").textContent = "…");
  const conP = api(`/api/connections?layer=${layer}&n=${n}&k=24`);
  let lens;
  try { lens = await api(`/api/lens?layer=${layer}&n=${n}`); }
  catch (err) { if (isFocus(layer, n, "neuron") && $("#d-lens")) $("#d-lens").textContent = err.message; }
  if (!isFocus(layer, n, "neuron")) return;
  conP.then(con => {
    if (!isFocus(layer, n, "neuron")) return;
    map.setConnections(con); state.con = con; renderConnections();
  }, err => { if (isFocus(layer, n, "neuron") && $("#d-con")) $("#d-con").textContent = err.message; });
  if (!lens) return;
  state.lens = { ...lens, layer, n }; renderList(); renderLensDetails(); renderMeasures();
}

// the open neuron's lenses in Details and above the chat; rendered from state so a language change can redraw it
function renderLensDetails() {
  const L = state.lens, f = state.focus;
  if (!L || !f || f.kind !== "neuron" || L.layer !== f.layer || L.n !== f.n) return;
  const tok = x => `<span class="tok">${esc(x)}</span>`;
  const steer = L.steer_up ? `<div class="lab">${t("lens.steerShort")}</div><div class="lens"><div class="up"><div class="lab">${t("lens.push", { kl: L.kl_up })}</div>${L.steer_up.map(tok).join("")}</div><div class="down"><div class="lab">${t("lens.pull", { kl: L.kl_down })}</div>${L.steer_down.map(tok).join("")}</div></div><div class="lab" style="margin-top:14px">${t("lens.logit")}</div>` : "";
  const el = $("#d-lens");
  if (el) {
    el.classList.remove("slot");
    el.innerHTML = `${steer}<div class="lens"><div class="up"><div class="lab">+</div>${L.up.map(tok).join("")}</div><div class="down"><div class="lab">−</div>${L.down.map(tok).join("")}</div></div>`;
  }
  const words = state.unitMap[L.layer]?.get(L.n)?.words || [];
  // the four lines above the chat: what each pushed column should lean toward, by steer lens and by logit lens
  const line = (cls, k, tip, toks) => `<div class="${cls}" title="${tip}"><span class="ll ${k.endsWith("▲") ? "up" : "down"}">${k}</span>${esc(toks.slice(0, 10).join("  "))}</div>`;
  $("#c-lens").innerHTML = (words.length ? `<div class="words">${esc(words.join("  "))}</div>` : "") +
    (L.steer_up ? line("", "steer ▲", "steer lens, ▲ column: tokens whose next-token odds rise most when this neuron is pushed up", L.steer_up)
      + line("", "steer ▼", "steer lens, ▼ column: tokens whose next-token odds rise most when this neuron is pushed down", L.steer_down) : "") +
    line("faintline", "logit ▲", "logit lens, ▲ column: tokens the neuron's output direction points at in the vocabulary", L.up)
    + line("faintline", "logit ▼", "logit lens, ▼ column: tokens the neuron's output direction points away from", L.down);
}

// sign = sign(g * u): the sign of the reading neuron's activation that the writing neuron's direction produces on its
// own. Reversing the writer reverses g and u together, so the sign holds for either direction of the writer.
function renderConnections() {
  const el = $("#d-con") || $("#d-con-wrap"); if (!el || !state.con) return;
  const tok = t => `<span class="tok">${esc(t)}</span>`;
  const row = c => {
    const pole = c.sign > 0 ? 1 : -1, toks = (pole > 0 ? c.lens : c.lens_neg) || [];
    const logit = (pole > 0 ? c.logit : c.logit_neg) || [];
    const sweep = c.name || (c.words || []).join(" ");
    return `<tr data-l="${c.layer}" data-n="${c.n}" class="con"><td class="pole ${pole > 0 ? "up" : "down"}">${pole > 0 ? "↑" : "↓"}</td>
      <td>${toks.slice(0, 4).map(tok).join("")}${logit.length ? `<div class="logit">${logit.slice(0, 4).map(tok).join("")}</div>` : ""}${sweep ? `<div class="dim">${esc(sweep)}</div>` : ""}</td>
      <td class="num dim">${c.z.toFixed(0)}</td><td class="mono faint">L${c.layer}·${c.n}</td></tr>`;
  };
  el.outerHTML = `<div id="d-con-wrap">
    <div class="lens"><div><div class="lab">${t("details.writesInto")}</div><table class="dtable">${state.con.down.slice(0, 14).map(row).join("")}</table></div>
    <div><div class="lab">${t("details.drivenBy")}</div><table class="dtable">${state.con.up.slice(0, 14).map(row).join("")}</table></div></div>
    ${headTablesHTML()}</div>`;
}
// the open neuron's heads in Details: later heads reading it (q / k / v) and heads it reads (o), each with what the
// head writes (its o tokens) over what it carries (v tokens, faint)
function headTablesHTML() {
  const H = state.nheads, f = state.focus;
  if (!H?.roles || !f || f.kind !== "neuron" || H.layer !== f.layer || H.n !== f.n) return "";
  const tok = t => `<span class="tok">${esc(t.tok)}</span>`;
  const row = x => {
    const info = hoverCache.get(`h/${x.layer}/${x.head}`);
    const id = `L${x.layer}·h${x.heads.length > 1 ? `${x.heads[0]}–${x.heads.at(-1)}` : x.head}`;
    return `<tr class="con" data-hl="${x.layer}" data-hh="${x.head}"><td class="pole mono dim">${x.role}</td>
      <td>${info ? info.write_toks.slice(0, 4).map(tok).join("") + `<div class="logit">${info.read_toks.slice(0, 4).map(tok).join("")}</div>` : ""}</td>
      <td class="num dim">${x.z.toFixed(1)}</td><td class="mono faint">${id}</td></tr>`;
  };
  const rows = roles => roles.flatMap(r => (H.roles[r] || []).map(x => ({ ...x, role: r }))).filter(x => x.z >= 2).sort((a, b) => b.z - a.z);
  return `<div class="lens"><div><div class="lab">${t("details.headsReading")}</div><table class="dtable">${rows(["q", "k", "v"]).slice(0, 10).map(row).join("")}</table></div>
    <div><div class="lab">${t("details.headsFeeding")}</div><table class="dtable">${rows(["o"]).slice(0, 10).map(row).join("")}</table></div></div>`;
}
// opening a neuron from a list or link also centres the camera on it
function goToNeuron(layer, n, kind = "neuron") {
  openNeuron(layer, n, kind, true, false);
  if (kind === "neuron") map.focusNeuron(layer, n);
}
$("#tab-details").addEventListener("click", e => {
  if (e.target.id === "d-guide" && state.focus) { FG.open = { layer: state.focus.layer, n: state.focus.n }; showLab("guide"); return; }
  const th = e.target.closest("tr[data-hl]"); if (th) return openHead(+th.dataset.hl, +th.dataset.hh);
  const tr = e.target.closest("tr[data-l]"); if (tr) goToNeuron(+tr.dataset.l, +tr.dataset.n);
});
$("#tab-details").addEventListener("mouseover", e => { const th = e.target.closest("tr[data-hl]"); map.hoverHead(th ? +th.dataset.hl : null, th ? +th.dataset.hh : null); });
$("#tab-details").addEventListener("mouseleave", () => map.hoverHead(null));

// name autosave
let nameTimer, pendingName = null;
async function saveName(p) {
  await api("/api/name", { layer: p.layer, n: p.n, name: p.name });
  const u = state.unitMap[p.layer]?.get(p.n); if (u) u.name = p.name.trim();
  if (p.layer === state.layer && state.units[p.layer]?.length) map.setUnits(p.layer, state.units[p.layer]);
  renderList();
  if (isFocus(p.layer, p.n, "neuron")) { $("#f-name").classList.add("saved"); setTimeout(() => $("#f-name").classList.remove("saved"), 900); }
}
function flushName() {
  clearTimeout(nameTimer);
  if (pendingName) { const p = pendingName; pendingName = null; saveName(p); }
}
// the name field wraps long names: a one-line textarea that grows to fit; Enter finishes editing instead of a newline
function sizeNameField() { const t = $("#f-name"); t.style.height = "auto"; t.style.height = `${t.scrollHeight}px`; }
function setNameField(v) { $("#f-name").value = v; sizeNameField(); }
$("#f-name").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } });
$("#f-name").addEventListener("input", () => { const t = $("#f-name"); if (t.value.includes("\n")) t.value = t.value.replace(/\n+/g, " "); sizeNameField(); });
addEventListener("resize", sizeNameField);
$("#f-name").addEventListener("input", () => {
  const f = state.focus; if (!f || f.kind !== "neuron") return;
  pendingName = { layer: f.layer, n: f.n, name: $("#f-name").value };
  clearTimeout(nameTimer);
  nameTimer = setTimeout(flushName, 500);
});

// ---------- chat ----------
const doseEl = $("#c-dose");
function showDose() { doseEl.value = state.chat.mode === "mul" ? `×${state.chat.dose.mul}` : state.chat.dose.add; }
doseEl.addEventListener("change", () => {
  const v = parseFloat(doseEl.value.replace(/[×x*]/g, ""));
  if (Number.isFinite(v) && v >= 0) state.chat.dose[state.chat.mode] = v;
  showDose();
});
document.querySelectorAll("#c-mode button").forEach(b => b.onclick = () => {
  state.chat.mode = b.dataset.m;
  document.querySelectorAll("#c-mode button").forEach(x => x.classList.toggle("on", x === b)); showDose();
});
document.querySelectorAll("#c-dir button").forEach(b => b.onclick = () => {
  state.chat.dir = b.dataset.d;
  document.querySelectorAll("#c-dir button").forEach(x => x.classList.toggle("on", x === b));
  $("#tab-chat").classList.toggle("reads", b.dataset.d === "reads");
  syncPlaceholders();
});
function step(dir) {
  const steps = state.chat.mode === "mul" ? MUL_STEPS : ADD_STEPS, key = state.chat.mode;
  // step to the next value in the ladder above or below the current dose (typed values land between steps)
  const v = state.chat.dose[key];
  const next = dir > 0 ? steps.find(x => x > v + 1e-9) : [...steps].reverse().find(x => x < v - 1e-9);
  if (next !== undefined) state.chat.dose[key] = next;
  showDose();
}
$("#c-minus").onclick = () => step(-1); $("#c-plus").onclick = () => step(1);
showDose();

let running = null;
state.thread = [];
const ROWS = ["up", "zero", "down"];
function turnLabels(t) {
  return t.mode === "mul" ? [`▲ ×${t.dose}`, "×1", `▼ ×−${t.dose}`] : [`▲ +${t.dose}`, "0", `▼ −${t.dose}`];
}

// ---------- experiments: outcome measures, n paired samples per column, dose sweeps ----------
const X = { cfg: null, measures: [], windows: {}, job: null, pick: null };
const num = v => v == null ? "—" : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
const sgn = v => v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + num(Math.abs(v));
const ivl = c => c ? `[${num(c[0])}, ${num(c[1])}]` : "";
const LENS_MEASURE = "lens words";
async function loadExperiments() {
  try { X.cfg = await api("/api/experiment/config"); X.windows = await api("/api/experiment/windows"); }
  catch { return; }
  X.measures = X.cfg.measures;
  $("#x-method").value = X.cfg.ablation || "mean";
  $("#x-n").value = X.cfg.n; $("#x-temp").value = X.cfg.temperature; $("#x-max").value = X.cfg.max_new; $("#x-seed").value = X.cfg.seed;
  let saved = null; try { saved = localStorage.getItem("ne.measure"); } catch {}
  X.pick = saved ?? LENS_MEASURE; renderMeasures();
  loadTasks();
}
// the open neuron's own lens as a measure: word starts of its top steer-lens tokens (logit lens without a steer
// index), so a reply counts words like the ones this neuron pushes toward
function lensMeasure() {
  const L = state.lens, f = state.focus;
  if (!L || !f || L.layer !== f.layer || L.n !== f.n) return null;
  const clean = (L.steer_up || L.up).slice(0, 12).map(t => t.trim().toLowerCase().replace(/^[^\p{L}]+|[^\p{L}]+$/gu, ""))
    .filter(w => /^\p{L}+$/u.test(w) && !(w.length < 3 && /^[a-z]+$/.test(w))).sort((a, b) => a.length - b.length);
  const words = [];
  for (const w of clean) if (!words.some(s => w.startsWith(s))) words.push(w);
  return words.length ? { name: LENS_MEASURE, type: "words", words, stems: true, auto: true } : null;
}
const allMeasures = () => [lensMeasure(), ...X.measures].filter(Boolean);
function renderMeasures() {
  const sel = $("#x-measure"), ms = allMeasures(), lm = ms[0]?.auto ? ms[0] : null;
  // lens words stays picked while the open neuron's lens loads, rather than silently falling back to another measure
  const wait = X.pick === LENS_MEASURE && !lm ? `<option value="${LENS_MEASURE}">${LENS_MEASURE} …</option>` : "";
  sel.innerHTML = wait + ms.map(m => `<option>${esc(m.name)}</option>`).join("") + `<option value="">no measure</option>`;
  sel.value = wait || ms.some(m => m.name === X.pick) || X.pick === "" ? X.pick : (ms[0]?.name ?? "");
  sel.title = lm && sel.value === lm.name ? `lens words: replies are scored by hits per 100 words on words starting with ${lm.words.join(", ")} (from this neuron's lens)`
    : "what to count in every reply; coherence (how natural the text is to the unsteered model) is always scored too";
}
const currentMeasure = () => allMeasures().find(m => m.name === $("#x-measure").value) || null;
// a run started while lens words waits on the neuron's lens waits a few seconds for it instead of dropping the measure
async function lensReady() {
  for (let i = 0; i < 40 && X.pick === LENS_MEASURE && !lensMeasure(); i++) await new Promise(r => setTimeout(r, 250));
}
$("#x-measure").onchange = () => {
  X.pick = $("#x-measure").value;
  try { localStorage.setItem("ne.measure", X.pick); } catch {}
  renderMeasures();
  if (!$("#x-editor").hidden) renderEditor();
  renderWindow(); if ($("#sort").value === "window") renderList();
};
$("#x-more-btn").onclick = () => {
  const el = $("#x-more"); el.hidden = !el.hidden; $("#x-more-btn").setAttribute("aria-expanded", String(!el.hidden));
};
function xSettings() {
  const int = (id, lo, hi, d) => { const v = parseInt($(id).value, 10); return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; };
  const t = parseFloat($("#x-temp").value);
  return { n: int("#x-n", 1, 32, 8), temperature: Number.isFinite(t) && t >= 0 ? t : 0.7,
           max_new: int("#x-max", 1, 2000, 200), seed: int("#x-seed", 0, 2 ** 31 - 1, 0), measure: currentMeasure() };
}
const windowOf = (layer, n, name) => X.windows[`L${layer}/N${n}`]?.[name];

// the measure editor: one preset at a time; save writes the whole list to experiments/measures.json
function renderEditor(m = currentMeasure() || { name: "", type: "words", words: [] }) {
  if (m.auto) m = { ...m, name: "", auto: false };  // lens words is computed per neuron: editing it starts a new preset
  const val = m.type === "words" ? (m.words || []).join(", ") : m.type === "contains" ? m.word || "" : m.pattern || "";
  const hint = { words: "words or phrases, comma-separated: hits per 100 words", contains: "one word: 1 if the reply has it, else 0", regex: "a regular expression: match count" };
  $("#x-editor").innerHTML = `<div class="row"><input id="xe-name" placeholder="measure name" value="${esc(m.name)}" spellcheck="false">
    <span class="seg" id="xe-type">${["words", "contains", "regex"].map(k => `<button type="button" data-k="${k}" class="${k === m.type ? "on" : ""}">${k}</button>`).join("")}</span>
    ${m.type === "words" ? `<label class="xf" title="count words that start with an entry (alien matches aliens), not just whole words"><input type="checkbox" id="xe-stems"${m.stems ? " checked" : ""}> word starts</label>` : ""}</div>
    <textarea id="xe-val" rows="2" spellcheck="false" placeholder="${hint[m.type]}">${esc(val)}</textarea>
    <div class="row"><button type="button" id="xe-save">save</button><button type="button" id="xe-new" class="dim">new</button><button type="button" id="xe-del" class="dim">delete</button><span id="xe-msg" class="dim"></span><button type="button" id="xe-close" class="dim">close</button></div>`;
  $("#x-editor").dataset.orig = m.name;
  $("#xe-type").onclick = e => { const b = e.target.closest("button"); if (b) renderEditor({ ...readEditor(), type: b.dataset.k }); };
  $("#xe-new").onclick = () => renderEditor({ name: "", type: "words", words: [] });
  $("#xe-close").onclick = () => { $("#x-editor").hidden = true; };
  $("#xe-del").onclick = () => saveMeasures(X.measures.filter(x => x.name !== $("#x-editor").dataset.orig), "");
  $("#xe-save").onclick = () => {
    const m = readEditor(), orig = $("#x-editor").dataset.orig;
    if (!m.name) { $("#xe-msg").textContent = "name it first"; return; }
    if (m.name === LENS_MEASURE) { $("#xe-msg").textContent = `“${LENS_MEASURE}” is the automatic measure; pick another name`; return; }
    const list = X.measures.filter(x => x.name !== orig && x.name !== m.name);
    const at = X.measures.findIndex(x => x.name === orig);
    list.splice(at < 0 ? list.length : Math.min(at, list.length), 0, m);
    saveMeasures(list, m.name);
  };
}
function readEditor() {
  const type = $("#xe-type .on")?.dataset.k || "words", v = $("#xe-val").value.trim(), name = $("#xe-name").value.trim();
  if (type === "words") return { name, type, words: v.split(/[,\n]/).map(w => w.trim()).filter(Boolean), ...($("#xe-stems")?.checked ? { stems: true } : {}) };
  return type === "contains" ? { name, type, word: v } : { name, type, pattern: v };
}
async function saveMeasures(list, pick) {
  const r = await api("/api/experiment/measures", { measures: list }).catch(err => ({ error: err.message }));
  if (r.error) { $("#xe-msg").textContent = r.error; return; }
  X.measures = r.measures; X.pick = pick; renderMeasures(); $("#x-measure").onchange(); renderEditor();
}
$("#x-edit").onclick = () => { const ed = $("#x-editor"); ed.hidden = !ed.hidden; if (!ed.hidden) renderEditor(); };

function renderThread() {
  const el = $("#c-thread");
  el.innerHTML = state.thread.map(entryHTML).join("");
  el.scrollTop = el.scrollHeight;
}
const entryHTML = (t, i) => t.kind === "sweep" ? `<div class="turn sweep" data-i="${i}">${sweepHTML(t)}</div>`
  : t.kind === "task" ? `<div class="turn sweep task" data-i="${i}">${taskHTML(t)}</div>` : t.kind === "ablate" ? ablateHTML(t, i) : turnHTML(t, i);
const COL_TIPS = ["pushed up by +dose", "unsteered: the baseline the other two are compared with", "pushed down by −dose"];
function turnHTML(t, i) {
  const S = t.summary, labs = turnLabels(t), m = t.measure, hide = t.blind && !t.blind.picked;
  const picker = t.n > 1 ? `<div class="picker" role="group" aria-label="reply"><span class="xk" title="each column has ${t.n} replies; reply i used the same seed in all three, so they line up">reply</span>${Array.from({ length: t.n }, (_, s) =>
    `<button type="button" data-s="${s}" class="${s === t.pick ? "on" : ""}">${s + 1}</button>`).join("")}</div>` : "";
  // nothing to compare when no reply in any column scored above zero: say so instead of three "unchanged" lines
  const none = S && m && S.columns.every(c => c.measure.mean === 0 && !c.measure.sd);
  const notes = hide ? "" : !S ? (t.scoring ? `<div class="xnote dim">scoring…</div>` : "")
    : (none ? `<div class="xnote">No reply in any column had any ${esc(m.name)}, so there is nothing to compare.${m.auto ? "" : ` Try ${LENS_MEASURE}, which counts the words this neuron's lens points at.`}</div>`
      : S.columns.filter(c => c.vs0).map(c => `<div class="xline">${esc(c.vs0.sentence)}</div>`).join(""))
    + (S.warn ? `<div class="xnote warn">${S.n} replies per column (under ${X.cfg?.warn_n ?? 5}), so these intervals are rough.</div>` : "")
    + (S.reproducible === false ? `<div class="xnote dim">Running on ${esc(S.device)}: the same seed may not reproduce this exact text.</div>` : "");
  return `<div class="turn" data-i="${i}">
    <div class="user">${esc(t.user)}<span class="recipe">${t.mode === "mul" ? "×" : "add"} ${t.dose}${t.n > 1 ? ` · ${t.n} replies · temp ${t.temperature}` : ""}${m ? ` · ${esc(m.name)}${m.auto ? ` (${esc(m.words.join(", "))})` : ""}` : ""}</span></div>
    ${picker}
    <div class="cols">${(t.blind ? t.blind.order : [0, 1, 2]).map((r, j) => {
      const c = ROWS[r], C = S?.columns[r], sc = C?.scores?.[t.pick], coh = C?.coh_scores?.[t.pick];
      const one = sc != null && !hide ? `<span class="one" title="scores of the reply shown">reply ${t.pick + 1}: ${num(sc)}${coh != null ? ` · coherence ${num(coh)}` : ""}</span>` : "";
      const lab = !t.blind ? labs[r] : hide ? BLIND_NAMES[j] : `${BLIND_NAMES[j]} · ${labs[r]}`;
      return `<div class="col ${hide ? "blind" : c}${t.live && !t.done[r] ? " live" : ""}" data-r="${r}"><span class="lab" title="${hide ? "hidden until you pick" : COL_TIPS[r]}">${lab}</span>${C && !hide ? colStats(C, m) : ""}<span class="txt">${esc(t.replies[r][t.pick] || "")}</span>${one}</div>`;
    }).join("")}
    ${t.error ? `<div class="err">${esc(t.error)}</div>` : ""}</div>${t.blind ? blindHTML(t) : ""}${notes}${predHTML(t)}</div>`;
}
// a column's stats over its n replies: the measure's mean and 95% interval, the paired difference from the
// unsteered column with its effect size, and coherence
function colStats(c, m) {
  const row = (k, tip, v, sub = "") => `<div title="${tip}"><span class="k">${k}</span><span>${v}${sub ? `<span class="ci">${sub}</span>` : ""}</span></div>`;
  return `<div class="xs">`
    + (m ? row(esc(m.name), `mean ${esc(m.name)} over ${c.measure.n} replies, with its 95% interval`, num(c.measure.mean), ivl(c.measure.ci)) : "")
    + (m && c.vs0 ? row(c.label === "ablated" ? "vs intact" : "vs 0", "change from the unsteered (or intact) column, reply by reply, with its 95% interval and effect size d (0.5 is medium, 0.8 large)",
                        sgn(c.vs0.diff), [ivl(c.vs0.ci), c.vs0.d != null ? `d ${num(c.vs0.d)}` : ""].filter(Boolean).join(" · ")) : "")
    + row("coherence", "mean log-probability per token under the unsteered model: near 0 reads naturally, much lower is garbled", num(c.coherence.mean), ivl(c.coherence.ci))
    + (c.script.mean > 0.01 ? row("other script", "share of letters in a different writing system from the prompt", `${Math.round(c.script.mean * 100)}%`) : "") + `</div>`;
}
$("#c-thread").addEventListener("click", e => {
  const b = e.target.closest(".picker button");
  if (b) { const t = state.thread[+b.closest(".turn").dataset.i]; t.pick = +b.dataset.s; renderTurn(t); return; }
  if (e.target.closest(".x-cancel") && X.job) api("/api/experiment/cancel", { id: X.job.id }).catch(() => {});
  const bp = e.target.closest(".blind-q button[data-j]");
  if (bp) { pickBlind(state.thread[+bp.closest(".turn").dataset.i], +bp.dataset.j); return; }
  const v = e.target.closest(".xpred button[data-v]");
  if (v) { const t = state.thread[+v.closest(".turn").dataset.i], to = t.verdict === v.dataset.v ? null : v.dataset.v;
           setVerdict(expIdOf(t), to).then(ok => { if (ok) { t.verdict = to; renderTurn(t); } }); }
});
function renderTurn(t) {
  const i = state.thread.indexOf(t), el = $(`#c-thread .turn[data-i="${i}"]`);
  if (!el) return;
  // a re-render while the blind property is being typed keeps the cursor where it was
  const a = document.activeElement, typing = el.contains(a) && a.classList.contains("bprop") ? a.selectionStart : null;
  el.outerHTML = entryHTML(t, i);
  const box = typing != null && $(`#c-thread .turn[data-i="${i}"] .bprop`);
  if (box) { box.focus(); box.setSelectionRange(typing, typing); }
}
$("#c-clear").onclick = () => { if (running) running.abort(); if (X.job) api("/api/experiment/cancel", { id: X.job.id }).catch(() => {}); state.thread = []; renderThread(); };
let reading = false;

// reads a server-sent event stream, calling on(event) for each
async function sse(res, on) {
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    let k;
    while ((k = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, k); buf = buf.slice(k + 2);
      if (line.startsWith("data: ")) on(JSON.parse(line.slice(6)));
    }
  }
}

$("#c-form").addEventListener("submit", async e => {
  e.preventDefault();
  if (running) { running.abort(); return; }
  const f = state.focus; if (!f) return;
  const text = $("#c-input").value.trim() || $("#c-input").placeholder;
  if (state.chat.dir === "reads") return runReads(f, text);
  await lensReady(); if (running || !isFocus(f.layer, f.n, f.kind)) return;
  const dose = state.chat.mode === "mul" ? state.chat.dose.mul : state.chat.dose.add, x = xSettings();
  const history = state.thread.filter(t => !t.kind && !t.error).map(t => ({ user: t.user, replies: t.replies }));
  const blank = () => Array.from({ length: x.n }, () => "");
  const turn = { user: text, n: x.n, pick: 0, replies: [blank(), blank(), blank()], done: [false, false, false], dose,
                 mode: state.chat.mode, temperature: x.temperature, measure: x.measure, live: true, prediction: takePrediction(),
                 blind: state.chat.blind ? { order: shuffle([0, 1, 2]), property: lastProperty, target: "up", picked: null } : null };
  state.thread.push(turn); $("#c-input").value = ""; renderThread();
  const ctl = new AbortController(), thread = state.thread;
  running = ctl; $("#c-go").textContent = t("chat.stop");
  try {
    const res = await fetch("/api/steer", { method: "POST", headers: { "Content-Type": "application/json" }, signal: ctl.signal,
      body: JSON.stringify({ layer: f.layer, n: f.n, prompt: text, dose, mode: state.chat.mode, history,
                             samples: x.n, temperature: x.temperature, seed: x.seed, max_new: x.max_new, measure: x.measure,
                             prediction: turn.prediction }) });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 160)}`);
    await sse(res, ev => {
      if (ev.error) turn.error = ev.error;
      if (ev.phase === "scoring") { turn.done = [true, true, true]; turn.scoring = true; renderTurn(turn); }
      if (ev.summary) { turn.summary = ev.summary; turn.scoring = false; renderTurn(turn); renderLesionCard(); }
      if (ev.row == null) return;
      turn.replies[ev.row][ev.s] += ev.t;
      if (ev.s !== turn.pick) return;
      const txt = $(`#c-thread .turn[data-i="${state.thread.indexOf(turn)}"] .col[data-r="${ev.row}"] .txt`);
      if (txt) txt.textContent += ev.t;
      const th = $("#c-thread"); if (th.scrollHeight - th.scrollTop - th.clientHeight < 80) th.scrollTop = th.scrollHeight;
    });
  } catch (err) {
    if (err.name !== "AbortError") turn.error = String(err);
  } finally {
    turn.live = false; turn.done = [true, true, true]; turn.scoring = false;
    if (running === ctl) { running = null; $("#c-go").textContent = t("chat.run"); }
    if (state.thread === thread) renderTurn(turn);
  }
});

// ---- dose sweep: a background job on the server; points draw as each dose finishes ----
async function runSweep(quick) {
  const f = state.focus; if (!f || f.kind !== "neuron" || !X.cfg) return;
  if (X.job) { $("#status").textContent = "a dose sweep is already running"; return; }
  await lensReady(); if (X.job || !isFocus(f.layer, f.n, "neuron")) return;
  const x = xSettings();
  if (!x.measure) { $("#status").textContent = X.pick === LENS_MEASURE ? "lens words: this neuron's lens is still loading" : "pick a measure before a dose sweep"; return; }
  const prompt = $("#c-input").value.trim() || $("#c-input").placeholder;
  const body = { layer: f.layer, n: f.n, prompt, measure: x.measure, temperature: x.temperature, seed: x.seed,
                 ...(quick ? { doses: X.cfg.quick.doses, samples: X.cfg.quick.n, max_new: X.cfg.quick.max_new }
                           : { doses: X.cfg.sweep_doses, samples: x.n, max_new: x.max_new }) };
  const t = { kind: "sweep", prompt, measure: x.measure, layer: f.layer, neuron: f.n, n: body.samples, temperature: x.temperature,
              doses: [...new Set([...body.doses, 0])].sort((a, b) => a - b), points: {}, live: true, progress: null,
              prediction: body.prediction = takePrediction() };
  state.thread.push(t); renderThread();
  let r;
  try { r = await api("/api/experiment/sweep", body); } catch (err) { r = { error: err.message }; }
  if (r.error) { t.error = r.error; t.live = false; renderTurn(t); return; }
  X.job = { id: r.id, t };
  try {
    const res = await fetch(`/api/experiment/events?id=${r.id}`);
    await sse(res, ev => {
      if (ev.error) t.error = ev.error;
      if (ev.device) { t.device = ev.device; t.reproducible = ev.reproducible; }
      if (ev.point) t.points[ev.point.dose] = ev.point;
      if (ev.progress) t.progress = ev.progress;
      if (ev.end) Object.assign(t, { live: false, window: ev.window, index: ev.index, expId: ev.id, cancelled: ev.cancelled });
      const th = $("#c-thread"), near = th.scrollHeight - th.scrollTop - th.clientHeight < 80;
      renderTurn(t); if (near) th.scrollTop = th.scrollHeight;
    });
  } catch (err) { t.error = String(err); }
  finally {
    X.job = null; t.live = false; renderTurn(t);
    try { X.windows = await api("/api/experiment/windows"); } catch {}
    renderWindow(); renderList();
  }
}
$("#x-sweep").onclick = () => runSweep(false);
$("#x-quick").onclick = () => runSweep(true);

function sweepHTML(t) {
  const p = t.progress;
  const head = `<div class="user sweep-head">dose sweep · ${esc(t.measure.name)} · ${t.n} replies per dose · temp ${t.temperature}<span class="recipe">L${t.layer} N${t.neuron} · “${esc(t.prompt)}”${t.measure.auto ? ` · ${esc(t.measure.words.join(", "))}` : ""}</span></div>`;
  const prog = t.live ? `<div class="xnote dim">${p ? `${p.done}/${p.total} replies · about ${Math.round(p.eta)} s left` : "starting…"} · <button type="button" class="x-cancel">cancel</button></div>`
    : t.cancelled ? `<div class="xnote dim">cancelled${t.expId ? " · partial results saved" : ""}</div>` : "";
  const win = t.window ? windowHTML(t.window, t.measure.name) : "";
  const notes = Object.values(t.points).filter(q => q.vs0).sort((a, b) => a.dose - b.dose).map(q => `<div class="xline">${esc(q.vs0.sentence)}</div>`).join("");
  const repro = t.reproducible === false ? `<div class="xnote dim">Running on ${esc(t.device)}: the same seed may not reproduce this exact text.</div>` : "";
  return head + prog + sweepPlot(t) + win + (t.error ? `<div class="err">${esc(t.error)}</div>` : "")
    + (notes ? `<details class="xsent"><summary class="dim">each dose vs 0</summary>${notes}</details>` : "") + predHTML(t) + repro;
}
// the therapeutic window per direction, in words: where the measure starts to move (ED) and where coherence breaks (TD)
const WINDOW_TIP = "ED (effective dose): the smallest dose whose change in the measure has a 95% interval that excludes zero. "
  + "TD (toxic dose): the smallest dose whose coherence falls below the dashed line, or whose replies switch writing system. "
  + "window = TD − ED; index = TD / ED: higher means the neuron moves the measure well before it breaks the text.";
// task windows pass a title (task · readout) and say "answers" where a sweep says "text"
function windowHTML(w, name, title = name, what = "text") {
  return `<div class="xwin"><div class="lab" title="${WINDOW_TIP}">therapeutic window · ${esc(title)} <span class="q">?</span></div>${["positive", "negative"].map(k => windowLine(k, w[k], name, what)).join("")}</div>`;
}
function windowLine(k, r, name, what = "text") {
  const s = k === "positive" ? "+" : "−", cls = k === "positive" ? "up" : "down";
  if (!r) return "";
  if (r.ed == null) return `<div><span class="sgn ${cls}">${s}</span><span class="dim">no dose up to ${s}${r.max_dose} changed ${esc(name)}</span></div>`;
  const brk = r.td == null ? `${what} still ${what === "text" ? "coherent" : "valid"} at ${s}${r.max_dose}` : `${what} break${what === "text" ? "s" : ""} at ${s}${r.td} (TD)`;
  return `<div><span class="sgn ${cls}">${s}</span>moves from ${s}${r.ed} (ED) · ${brk}${r.index != null ? ` · index ${num(r.index)}` : ""}</div>`;
}
// measure (top) and coherence (bottom) against dose: mean with its 95% interval at each dose; the dashed line is
// the toxic threshold (dose 0's coherence minus the configured drop)
function sweepPlot(t, lab = {}) {
  const W = 560, L = 46, R = 12, ds = t.doses, x0 = Math.min(...ds), x1 = Math.max(...ds);
  const X_ = d => L + (d - x0) / ((x1 - x0) || 1) * (W - L - R);
  const pts = ds.map(d => t.points[d]).filter(Boolean);
  const base = t.points[0]?.coherence.mean, thr = base != null && X.cfg ? base - X.cfg.toxic.coherence_drop : null;
  const panel = (key, y0, h, label, extra = []) => {
    const vals = pts.flatMap(q => [q[key].mean, ...(q[key].ci || [])]).concat(extra).filter(v => v != null);
    let lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const Y = v => y0 + h - (v - lo) / (hi - lo) * h;
    let g = `<line class="ax" x1="${L}" x2="${W - R}" y1="${y0 + h}" y2="${y0 + h}"/><text class="tk" x="${L - 6}" y="${Y(hi - pad) + 4}" text-anchor="end">${num(hi - pad)}</text><text class="tk" x="${L - 6}" y="${Y(lo + pad) + 4}" text-anchor="end">${num(lo + pad)}</text><text class="pl" x="${L}" y="${y0 - 4}">${esc(label)}</text>`;
    if (key === "coherence" && thr != null) g += `<line class="thr" x1="${L}" x2="${W - R}" y1="${Y(thr)}" y2="${Y(thr)}"/>`;
    const line = pts.filter(q => q[key].mean != null).map(q => `${X_(q.dose)},${Y(q[key].mean)}`).join(" ");
    if (line) g += `<polyline class="mean" points="${line}"/>`;
    for (const q of pts) {
      const m = q[key], c = q.dose > 0 ? "up" : q.dose < 0 ? "down" : "zero";
      if (m.ci) g += `<line class="ci ${c}" x1="${X_(q.dose)}" x2="${X_(q.dose)}" y1="${Y(m.ci[0])}" y2="${Y(m.ci[1])}"/>`;
      if (m.mean != null) g += `<circle class="pt ${c}" cx="${X_(q.dose)}" cy="${Y(m.mean)}" r="3.5"><title>${fmtDose(q.dose)}: ${num(m.mean)} ${ivl(m.ci)}</title></circle>`;
    }
    return g;
  };
  const top = 18, h1 = 120, gap = 34, h2 = 70, H = top + h1 + gap + h2 + 36;
  const ticks = ds.map(d => `<text class="tk" x="${X_(d)}" y="${H - 20}" text-anchor="middle">${fmtDose(d)}</text>`).join("")
    + `<text class="pl" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">dose (− pushes down · 0 unsteered · + pushes up)</text>`;
  const flat = pts.length > 1 && pts.every(q => q.measure.mean === 0 && !q.measure.sd);
  return `<svg class="xplot" viewBox="0 0 ${W} ${H}" role="img" aria-label="dose-response for ${esc(t.measure.name)} and coherence">`
    + panel("measure", top, h1, lab.top || `${t.measure.name} · ${({ words: "hits per 100 words", contains: "share of replies", regex: "matches per reply" })[t.measure.type] || ""}${flat ? " · zero at every dose so far" : ""}`)
    + panel("coherence", top + h1 + gap, h2, lab.bottom || "coherence · higher reads more naturally", thr != null ? [thr] : [])
    + ticks + `</svg><div class="xnote dim xlegend">${lab.dots || "dots: mean of the replies at each dose · bars: 95% interval"}${thr != null ? ` · <span class="thr-key">dashed</span>: ${lab.thr || `coherence ${X.cfg.toxic.coherence_drop} below the unsteered column, where text counts as broken`}` : ""}</div>`;
}

// the open neuron's therapeutic window for the chosen measure, on its Details tab
function renderWindow() {
  const el = $("#d-window"), f = state.focus; if (!el || !f) return;
  const m = currentMeasure(), w = m && windowOf(f.layer, f.n, m.name);
  const tasks = Object.entries(X.windows[`L${f.layer}/N${f.n}`] || {}).filter(([k]) => k.startsWith("task: "));
  el.innerHTML = (w ? `${windowHTML(w.window, m.name)}<div class="xnote dim">from a dose sweep on “${esc(w.prompt)}”</div>` : "")
    + tasks.map(([k, v]) => `${windowHTML(v.window, k.split(" · ").slice(1).join(" · "), k.slice(6), "answers")}<div class="xnote dim">from the ${esc(v.prompt)} task</div>`).join("");
  renderLesionCard();
}
// steering and ablation side by side on the neuron card: steering tests sufficiency, ablation necessity
async function renderLesionCard() {
  const f = state.focus, m = currentMeasure(); if (!$("#d-results") || !f || f.kind !== "neuron") return;
  const r = await api(`/api/experiment/results?layer=${f.layer}&n=${f.n}`).catch(() => ({}));
  const el = $("#d-results"); if (!el || !isFocus(f.layer, f.n, "neuron")) return;
  const x = m ? r[m.name] || {} : {};
  if (!x.steer && !x.ablate) { el.innerHTML = ""; return; }
  const cell = (k, title, tip, v, how) => `<div class="lcell" title="${tip}"><div class="lab">${title}</div>${v ? `<div class="xline">${esc(v.sentence)}</div><div class="xnote dim">${how(v)} · “${esc(v.prompt)}” · n ${v.n}</div>` : `<div class="xnote dim">not run yet for ${esc(m.name)}</div>`}</div>`;
  el.innerHTML = `<div class="lab">steering vs ablation · ${esc(m.name)}</div><div class="lcard">`
    + cell("steer", "steering · is it sufficient?", "pushing the neuron: does that alone move the measure?", x.steer, v => `${v.mode === "mul" ? "×" : "add"} ${v.dose}`)
    + cell("ablate", "ablation · is it necessary?", "removing the neuron: does the measure depend on it?", x.ablate, v => `${v.method} ablation${v.set > 1 ? ` of a set of ${v.set}` : ""}`) + `</div>`;
}

// ---------- behavioral tasks: a paradigm from tasks/, run at every dose, one fitted parameter per dose ----------
X.tasks = [];
const curTask = () => X.tasks.find(k => k.id === $("#x-task").value) || null;
async function loadTasks() {
  try { X.tasks = (await api("/api/tasks")).tasks; } catch { return; }
  let saved = null; try { saved = localStorage.getItem("ne.task"); } catch {}
  $("#x-task").innerHTML = X.tasks.map(k => `<option value="${esc(k.id)}">${esc(k.name)}</option>`).join("");
  if (X.tasks.some(k => k.id === saved)) $("#x-task").value = saved;
  renderTaskNote();
}
let likert = {}; try { likert = JSON.parse(localStorage.getItem("ne.likert") || "{}"); } catch {}
const statementsOf = k => likert[k.id] || k.grid.statement;
// the methods note, and for a Likert task its statements: one line until opened, then one per line, kept in this browser
let editStatements = false;
function renderTaskNote() {
  const k = curTask(), el = $("#x-task-note"), open = $("#x-task-how").getAttribute("aria-expanded") === "true";
  el.hidden = !k || (!open && k.kind !== "likert");
  if (el.hidden) return;
  const n = k.kind === "likert" ? statementsOf(k).length : 0;
  el.innerHTML = (open ? `<p class="method">${esc(k.method)}</p>` : "")
    + (!n && k.kind !== "likert" ? "" : editStatements
      ? `<label class="xk" for="x-statements">statements, one per line · <button type="button" id="x-st-edit" class="dim">done</button></label><textarea id="x-statements" rows="5" spellcheck="false">${esc(statementsOf(k).join("\n"))}</textarea>`
      : `<div><button type="button" id="x-st-edit" class="dim" title="the statements the model rates; yours are kept in this browser">${n} statement${n === 1 ? "" : "s"} · edit</button></div>`);
}
$("#x-task-note").addEventListener("click", e => { if (e.target.id === "x-st-edit") { editStatements = !editStatements; renderTaskNote(); } });
$("#x-task").onchange = () => { try { localStorage.setItem("ne.task", $("#x-task").value); } catch {} renderTaskNote(); };
$("#x-task-how").onclick = () => {
  const b = $("#x-task-how"); b.setAttribute("aria-expanded", String(b.getAttribute("aria-expanded") !== "true")); renderTaskNote();
};
$("#x-task-note").addEventListener("input", e => {
  if (e.target.id !== "x-statements") return;
  likert[curTask().id] = e.target.value.split("\n").map(s => s.trim()).filter(Boolean);
  try { localStorage.setItem("ne.likert", JSON.stringify(likert)); } catch {}
});
async function runTask() {
  const f = state.focus, k = curTask(); if (!f || f.kind !== "neuron" || !X.cfg || !k) return;
  if (X.job) { $("#status").textContent = "a run is already going"; return; }
  const statements = k.kind === "likert" ? statementsOf(k) : null;
  if (statements && !statements.length) { $("#status").textContent = "write at least one statement to rate"; return; }
  const doses = X.cfg.sweep_doses;
  const t = { kind: "task", task: k, statements, layer: f.layer, neuron: f.n, doses: [...new Set([...doses, 0])].sort((a, b) => a - b),
              measure: { name: k.parameter, type: "task" }, points: {}, live: true, progress: null, prediction: takePrediction() };
  state.thread.push(t); renderThread();
  let r;
  try { r = await api("/api/experiment/task", { layer: f.layer, n: f.n, id: k.id, doses, prediction: t.prediction, ...(statements ? { statements } : {}) }); }
  catch (err) { r = { error: err.message }; }
  if (r.error) { t.error = r.error; t.live = false; renderTurn(t); return; }
  X.job = { id: r.id, t };
  try {
    await sse(await fetch(`/api/experiment/events?id=${r.id}`), ev => {
      if (ev.error) t.error = ev.error;
      if (ev.start) t.items = ev.items;
      if (ev.scoring) Object.assign(t, { scoring: ev.scoring, device: ev.device, reproducible: ev.reproducible });
      if (ev.point) { t.points[ev.point.dose] = ev.point; t.readout = ev.point.readout; }
      if (ev.progress) t.progress = ev.progress;
      if (ev.end) Object.assign(t, { live: false, window: ev.window, expId: ev.id, cancelled: ev.cancelled });
      const th = $("#c-thread"), near = th.scrollHeight - th.scrollTop - th.clientHeight < 80;
      renderTurn(t); if (near) th.scrollTop = th.scrollHeight;
    });
  } catch (err) { t.error = String(err); }
  finally {
    X.job = null; t.live = false; renderTurn(t);
    try { X.windows = await api("/api/experiment/windows"); } catch {}
    renderWindow();
  }
}
$("#x-task-run").onclick = runTask;
const pct = v => v == null ? "—" : `${Math.round(v * 100)}%`;
function taskHTML(t) {
  const k = t.task, p = t.progress, z = t.points[0], likertTask = k.kind === "likert";
  const head = `<div class="user sweep-head">task · ${esc(k.name)} · ${esc(k.parameter)}${t.items ? ` · ${t.items} items, each in both ${likertTask ? "scale directions" : "option orders"}` : ""}<span class="recipe">L${t.layer} N${t.neuron} · ${t.scoring === "sampled" ? "answers sampled and parsed" : "scored from next-token probabilities, no text generated"}</span></div>`;
  const prog = t.live ? `<div class="xnote dim">${p ? `${p.done}/${p.total} prompts · about ${Math.round(p.eta)} s left` : "starting…"} · <button type="button" class="x-cancel">cancel</button></div>`
    : t.cancelled ? `<div class="xnote dim">cancelled${t.expId ? " · partial results saved" : ""}</div>` : "";
  // the readout is the fitted parameter, unless the baseline fit sat at the edge of its range: then the choice rate
  const byRate = t.readout && t.readout !== k.parameter, name = byRate ? t.readout : k.parameter;
  const plot = sweepPlot(t, { top: byRate ? `${name} · at each dose` : `${name} · fitted at each dose`, bottom: "answer validity · log of the probability on valid answers",
    thr: "validity 1.0 below dose 0, where the model has stopped answering the question",
    dots: "dots: the value at each dose · bars: 95% interval from resampling the items" });
  // the baseline's own order effect: every item runs both ways, so it cancels out of the parameter, but it is worth knowing
  const bias = z?.bias?.mean == null ? "" : `<div class="xnote dim">${likertTask
    ? `At dose 0 the average digit picked sits ${sgn(z.bias.mean)} from the scale's midpoint whichever end means agree (0 is no number bias); the two directions cancel it out of the rating.`
    : `At dose 0 the model picks whichever option is listed first ${pct(z.bias.mean)} of the time (50% is no position bias); every item runs in both orders, so this cancels out of the parameter.`}</div>`;
  const edge = byRate ? `<div class="xnote warn">At dose 0 the ${esc(k.parameter)} fit ran to the edge of its range: the model's choices barely depend on the amounts, so the fitted value means little. This run reads ${esc(name)} instead.</div>`
    : Object.values(t.points).some(q => q.fit?.at_bound) ? `<div class="xnote warn">The ${esc(k.parameter)} fit sits at the edge of its range at some doses, where the choices barely depend on the amounts.</div>` : "";
  const lines = Object.values(t.points).sort((a, b) => a.dose - b.dose).map(q =>
    `<div class="xline">${fmtDose(q.dose)}: ${esc(k.parameter)} ${num(q.param.mean)} ${ivl(q.param.ci)}${k.rate ? ` · ${esc(k.rate)} ${pct(q.rate.mean)} [${pct(q.rate.ci[0])}, ${pct(q.rate.ci[1])}]` : ""}${q.vs0 ? `<span class="dim"> · ${esc(q.vs0.sentence)}</span>` : ""}</div>`).join("");
  const repro = t.reproducible === false ? `<div class="xnote dim">Running on ${esc(t.device)}: the same inputs may give slightly different probabilities elsewhere.</div>` : "";
  return head + prog + plot + (t.window ? windowHTML(t.window, name, name, "answers") : "") + edge + bias + (t.error ? `<div class="err">${esc(t.error)}</div>` : "")
    + (lines ? `<details class="xsent"><summary class="dim">each dose</summary>${lines}</details>` : "") + statementsHTML(t)
    + `<details class="xsent"><summary class="dim">how to read this</summary><div class="xline method">${esc(k.method)}</div></details>` + predHTML(t) + repro;
}

// a Likert run's statements, each with its rating at every dose so far (flipped back where 7 meant disagree)
function statementsHTML(t) {
  const ds = Object.keys(t.points).map(Number).sort((a, b) => a - b);
  if (!t.statements || !ds.length) return "";
  const head = `<tr><th></th>${ds.map(d => `<th>${fmtDose(d)}</th>`).join("")}</tr>`;
  const rows = t.statements.map((s, j) => `<tr><td>${esc(s)}</td>${ds.map(d => `<td>${num(t.points[d].items[j])}</td>`).join("")}</tr>`).join("");
  return `<details class="xsent"><summary class="dim">each statement</summary><table class="xstat">${head}${rows}</table></details>`;
}

// ---------- blind rating: columns shuffled as X, Y, Z; name a property, pick the column that shows it, then reveal ----------
const BLIND_NAMES = ["X", "Y", "Z"];
let lastProperty = "";
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
$("#c-blind").onclick = () => {
  state.chat.blind = !state.chat.blind;
  $("#c-blind").classList.toggle("on", state.chat.blind); $("#c-blind").setAttribute("aria-pressed", String(state.chat.blind));
};
function blindHTML(t) {
  const b = t.blind;
  if (b.picked) {
    const j = b.order.indexOf(ROWS.indexOf(b.picked)), hit = b.picked === b.target, want = b.target === "up" ? "▲ pushed up" : "▼ pushed down";
    return `<div class="blind-res"><span class="${hit ? "hit" : "miss"}">${hit ? "✓" : "✗"}</span> “${esc(b.property)}”: you picked ${BLIND_NAMES[j]}, the ${b.picked === "zero" ? "unsteered" : b.picked === "up" ? "▲ pushed up" : "▼ pushed down"} column; the ${want} column was ${BLIND_NAMES[b.order.indexOf(ROWS.indexOf(b.target))]}.</div>`
      + (b.stats ? `<div class="xline">${esc(b.stats.sentence)}</div>` : b.error ? `<div class="err">${esc(b.error)}</div>` : "");
  }
  const ready = !t.live && !t.error;
  return `<div class="blind-q">
    <input class="bprop" placeholder="property, e.g. more about aliens" value="${esc(b.property)}" aria-label="property" spellcheck="false">
    <select class="btarget" aria-label="which push should show it" title="a hit is picking the column pushed this way"><option value="up"${b.target === "up" ? " selected" : ""}>from ▲ push up</option><option value="down"${b.target === "down" ? " selected" : ""}>from ▼ push down</option></select>
    <span class="dim">${ready ? "shows it most:" : "replies coming…"}</span>${BLIND_NAMES.map((x, j) => `<button type="button" data-j="${j}"${ready ? "" : " disabled"}>${x}</button>`).join("")}</div>`;
}
$("#c-thread").addEventListener("input", e => {
  const t = state.thread[+e.target.closest(".turn")?.dataset.i]; if (!t?.blind) return;
  if (e.target.classList.contains("bprop")) t.blind.property = lastProperty = e.target.value;
});
$("#c-thread").addEventListener("change", e => {
  const t = state.thread[+e.target.closest(".turn")?.dataset.i];
  if (t?.blind && e.target.classList.contains("btarget")) t.blind.target = e.target.value;
});
async function pickBlind(t, j) {
  const b = t.blind, f = state.focus;
  if (!b || b.picked || t.live) return;
  if (!b.property.trim()) { const el = $(`#c-thread .turn[data-i="${state.thread.indexOf(t)}"] .bprop`); el?.focus(); el?.classList.add("need"); return; }
  b.picked = ROWS[b.order[j]]; renderTurn(t);
  try {
    b.stats = await api("/api/blind/trial", { layer: f.layer, n: f.n, property: b.property.trim(), target: b.target, picked: b.picked,
      order: b.order.map(r => ROWS[r]), dose: t.dose, mode: t.mode, prompt: t.user, sample: t.pick, experiment: t.summary?.id ?? null });
  } catch (err) { b.error = err.message; }
  renderTurn(t);
}

// ---------- hypothesis log: a prediction written before a run, saved with it, judged once the results are in ----------
const VERDICTS = ["confirmed", "disconfirmed", "inconclusive"];
function takePrediction() { const el = $("#x-pred"), v = el.value.trim(); el.value = ""; return v || null; }
const expIdOf = t => t.expId || t.summary?.id || null;
const verdictSeg = (v, id) => `<span class="seg vseg"${id ? ` data-id="${esc(id)}"` : ""}>${VERDICTS.map(k => `<button type="button" data-v="${k}" class="${v === k ? "on" : ""}" title="mark the prediction ${k}">${k}</button>`).join("")}</span>`;
function predHTML(t) {
  if (!t.prediction) return "";
  const id = expIdOf(t);
  const marks = id ? verdictSeg(t.verdict, null) : t.live ? `<span class="dim">judge it once the results are in</span>`
    : `<span class="dim">not saved: ${t.kind ? "the run didn't finish" : "chat runs are saved only with a measure"}</span>`;
  return `<div class="xpred"><span class="xk">prediction</span><span class="ptxt">${esc(t.prediction)}</span>${marks}</div>`;
}
async function setVerdict(id, verdict) {
  try { await api("/api/experiment/verdict", { id, verdict }); return true; }
  catch (err) { $("#status").textContent = err.message; return false; }
}

// ---------- lab: the notebook, the field guide and the games, in one tab that stays up without a neuron ----------
const LAB = { view: "notebook", only: false, notebook: null };
$("#lab-btn").onclick = () => { focusEl.classList.add("on"); syncTabs(); setTab("lab"); setFocusCollapsed(false); };
$("#lab-nav").onclick = e => {
  const b = e.target.closest("button"); if (!b) return;
  showLab(b.dataset.l);
};
function showLab(view) {
  LAB.view = view; document.querySelectorAll("#lab-nav button").forEach(x => x.classList.toggle("on", x.dataset.l === view));
  if (curTab() === "lab") renderLab(); else { focusEl.classList.add("on"); syncTabs(); setTab("lab"); setFocusCollapsed(false); }
}
function renderLab() {
  ({ notebook: renderNotebook, guide: renderGuide, challenges: renderChallenges, guess: renderGuess })[LAB.view]?.();
}
const when = ts => { const d = new Date(ts); return isNaN(d) ? "" : `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ${d.toTimeString().slice(0, 5)}`; };
const neuronChip = x => `<span class="chip nchip" data-l="${x.layer}" data-n="${x.n}" title="open this neuron">L${x.layer} N${x.n}</span>`;
// the notebook: every saved experiment newest first; a prediction made before the run sits above its result
async function renderNotebook() {
  const el = $("#lab-body");
  if (!LAB.notebook) el.innerHTML = `<div class="xnote dim">loading…</div>`;
  let r;
  try { r = await api("/api/notebook"); } catch (err) { el.innerHTML = `<div class="err">${esc(err.message)}</div>`; return; }
  if (curTab() !== "lab" || LAB.view !== "notebook") return;
  LAB.notebook = r.entries;
  const preds = r.entries.filter(e => e.prediction), count = v => preds.filter(e => e.verdict === v).length;
  const rows = (LAB.only ? preds : r.entries).map(e => `<div class="nb-row" data-id="${esc(e.id)}">
      <div class="nb-meta mono"><span>${esc(when(e.timestamp))}</span>${e.neurons.slice(0, 3).map(neuronChip).join("")}${e.neurons.length > 3 ? `<span class="dim">+${e.neurons.length - 3}</span>` : ""}<span class="dim">${esc(e.kind)}</span></div>
      <div class="nb-what">${esc(e.what)}</div>
      ${e.prediction ? `<div class="xpred"><span class="xk">prediction</span><span class="ptxt">${esc(e.prediction)}</span>${verdictSeg(e.verdict, e.id)}</div>` : ""}
      <div class="xline">${e.result ? esc(e.result) : `<span class="dim">no result line</span>`}</div></div>`).join("");
  el.innerHTML = `<div class="nb-head"><span class="dim">${r.total} experiment${r.total === 1 ? "" : "s"} · ${preds.length} with a prediction${preds.length ? ` · ${VERDICTS.map(v => `${count(v)} ${v}`).join(" · ")}` : ""}</span>
      <label class="xf"><input type="checkbox" id="nb-only"${LAB.only ? " checked" : ""}> only with a prediction</label></div>
    <div class="xnote dim">A prediction is written before its run and can't be changed after, so this works as a light preregistration: the verdict is the only thing you set afterward.</div>
    ${rows || `<div class="xnote dim">${LAB.only ? "No predictions yet: write one in the box above the chat input before a run." : "No experiments yet. Runs from the Chat tab land here."}</div>`}`;
}
$("#lab-body").addEventListener("click", e => {
  if (e.target.id === "nb-only") { LAB.only = e.target.checked; renderNotebook(); return; }
  const ca = e.target.closest(".ch-go"); if (ca) { runChallenge(ca.closest(".ch").dataset.id); return; }
  if (e.target.id === "g-new") { newGuess(); return; }
  const fc = e.target.closest(".fg-card"); if (fc) { FG.open = { layer: +fc.dataset.l, n: +fc.dataset.n }; renderGuide(); return; }
  if (e.target.id === "fg-back") { FG.open = null; renderGuide(); return; }
  if (e.target.id === "fg-md") { cardMarkdown(); return; }
  if (e.target.id === "fg-png") { cardPNG(); return; }
  if (e.target.id === "fg-add") { saveCard({ note: $("#fg-note").value }); return; }
  const rn = e.target.closest(".fg-rm"); if (rn) { saveCard({ remove_note: +rn.dataset.k }); return; }
  const go = e.target.closest(".g-opt"); if (go) { answerGuess(+go.dataset.k); return; }
  if (e.target.closest(".ch-cancel") && CH.run?.job) { api("/api/experiment/cancel", { id: CH.run.job }).catch(() => {}); return; }
  const c = e.target.closest(".nchip"); if (c) { goToNeuron(+c.dataset.l, +c.dataset.n); return; }
  const v = e.target.closest(".vseg[data-id] button[data-v]");
  if (v) {
    const id = v.parentElement.dataset.id, ent = LAB.notebook?.find(x => x.id === id), to = ent?.verdict === v.dataset.v ? null : v.dataset.v;
    setVerdict(id, to).then(ok => {
      if (!ok) return;
      for (const t of state.thread) if (expIdOf(t) === id) { t.verdict = to; renderTurn(t); }
      renderNotebook();
    });
  }
});

// ---- challenges: a goal and a constraint; an attempt steers the open neuron (or the set) and must keep the text coherent ----
const CH = { list: null, toxic: null, dose: {}, run: null };
const attemptNeurons = () => pinned.length ? pinned : state.focus?.kind === "neuron" ? [{ layer: state.focus.layer, n: state.focus.n }] : [];
const chValue = (m, v) => v == null ? "—" : m.type === "contains" ? pct(v) : m.type === "words" ? `${num(v)} /100 words` : `${num(v)} /reply`;
async function renderChallenges(reload = true) {
  const el = $("#lab-body");
  if (reload || !CH.list) {
    try { const r = await api("/api/challenges"); CH.list = r.challenges; CH.toxic = r.toxic; }
    catch (err) { el.innerHTML = `<div class="err">${esc(err.message)}</div>`; return; }
  }
  if (curTab() !== "lab" || LAB.view !== "challenges") return;
  const who = attemptNeurons();
  const whoHTML = who.length ? `${pinned.length ? "the set: " : "the open neuron "}${who.map(neuronChip).join("")}` : `<span class="warn">no neuron: open one, or build a set with + set</span>`;
  const rule = c => c.minimize === "neurons" ? "fewest neurons wins, then the smallest |dose|" : "smallest |dose| wins, then the fewest neurons";
  el.innerHTML = `<div class="xnote dim">An attempt runs the prompt ${X.cfg?.n ?? 8} times unsteered and ${X.cfg?.n ?? 8} times steered, paired by seed. It counts only if the goal is met and coherence stays within ${CH.toxic?.coherence_drop ?? 1} of the unsteered replies (the same toxic threshold as the dose sweep).</div>
    <div class="xnote">Steering with ${whoHTML}</div>`
    + CH.list.map(c => {
      const r = CH.run?.id === c.id ? CH.run : null;
      const best = c.best.length ? `<table class="xstat ch-best"><tr><th></th><th>dose</th><th>neurons</th><th>${esc(c.measure.name)}</th><th></th></tr>${c.best.map((a, k) =>
          `<tr><td>${k + 1}</td><td>${fmtDose(a.dose)}</td><td>${a.neurons.map(neuronChip).join("")}</td><td>${chValue(c.measure, a.value)}</td><td class="dim">${esc(when(a.time))}</td></tr>`).join("")}</table>`
        : `<div class="xnote dim">No passing attempt yet${c.tries ? ` (${c.tries} tr${c.tries === 1 ? "y" : "ies"})` : ""}.</div>`;
      const status = !r ? "" : r.live ? `<div class="xnote dim">${r.progress ? `${r.progress.done}/${r.progress.total} replies · about ${Math.round(r.progress.eta)} s left` : "starting…"} · <button type="button" class="ch-cancel x-cancel">cancel</button></div>`
        : r.error ? `<div class="err">${esc(r.error)}</div>` : r.cancelled ? `<div class="xnote dim">cancelled</div>`
        : r.result ? `<div class="xline ${r.result.passed ? "pass" : "fail"}">${esc(r.result.sentence)}</div><div class="xnote dim">unsteered: ${chValue(c.measure, r.result.baseline)}</div>
            <details class="xsent"><summary class="dim">steered replies</summary>${r.result.texts.map(t => `<div class="ch-reply">${esc(t)}</div>`).join("")}</details>` : "";
      return `<div class="ch" data-id="${esc(c.id)}"><div class="ch-head"><b>${esc(c.name)}</b><span class="dim"> · ${rule(c)}</span></div>
        <div class="xline">${esc(c.goal)}</div><div class="recipe mono dim">“${esc(c.prompt)}” · ${esc(c.measure.name)}</div>${best}
        <div class="ch-try"><label class="xf">dose<input class="mono ch-dose" inputmode="decimal" value="${esc(CH.dose[c.id] ?? "0.5")}" aria-label="dose"></label>
          <button type="button" class="ch-go"${who.length && !X.job ? "" : " disabled"}>attempt</button></div>${status}</div>`;
    }).join("");
}
$("#lab-body").addEventListener("input", e => { if (e.target.classList.contains("ch-dose")) CH.dose[e.target.closest(".ch").dataset.id] = e.target.value; });
$("#lab-body").addEventListener("change", e => {
  if (e.target.id === "fg-sort") { FG.sort = e.target.value; renderGuide(); }
  if (e.target.id === "fg-conf") saveCard({ confidence: e.target.value });
  if (e.target.id === "fg-hyp") saveCard({ hypothesis: e.target.value }, false);
});
$("#lab-body").addEventListener("keydown", e => { if (e.target.id === "fg-note" && e.key === "Enter") { e.preventDefault(); saveCard({ note: e.target.value }); } });
async function runChallenge(id) {
  const neurons = attemptNeurons(), dose = parseFloat(String(CH.dose[id] ?? "0.5").replace("−", "-"));
  if (!neurons.length || X.job) return;
  if (!Number.isFinite(dose) || !dose) { $("#status").textContent = "the dose must be a number other than 0"; return; }
  const r = CH.run = { id, live: true, progress: null };
  renderChallenges(false);
  let s;
  try { s = await api("/api/challenge/attempt", { id, neurons, dose, samples: X.cfg?.n ?? 8, temperature: X.cfg?.temperature, seed: X.cfg?.seed }); }
  catch (err) { s = { error: err.message }; }
  if (s.error) { Object.assign(r, { live: false, error: s.error }); renderChallenges(false); return; }
  r.job = s.id; X.job = { id: s.id };
  try {
    await sse(await fetch(`/api/experiment/events?id=${s.id}`), ev => {
      if (ev.error) r.error = ev.error;
      if (ev.progress) r.progress = ev.progress;
      if (ev.result) r.result = ev.result;
      if (ev.end) Object.assign(r, { live: false, cancelled: ev.cancelled });
      if (curTab() === "lab" && LAB.view === "challenges") renderChallenges(false);
    });
  } catch (err) { r.error = String(err); }
  finally { X.job = null; r.live = false; if (curTab() === "lab" && LAB.view === "challenges") renderChallenges(); }
}

// ---- guess the neuron: a hidden neuron's replies at +dose and −dose; its label among three others of the same kind ----
const G = { score: null, round: null };
async function renderGuess() {
  const el = $("#lab-body");
  if (!G.score) { try { G.score = await api("/api/game/guess"); } catch (err) { el.innerHTML = `<div class="err">${esc(err.message)}</div>`; return; } }
  if (curTab() !== "lab" || LAB.view !== "guess") return;
  const sc = G.score, r = G.round;
  const head = `<div class="nb-head"><span class="mono">${sc.correct}/${sc.played} right · streak ${sc.streak} · best ${sc.best}</span>
    <button type="button" id="g-new" class="xbtn"${r?.live || X.job ? " disabled" : ""}>${r ? "next round" : "start a round"}</button></div>`;
  if (!r) { el.innerHTML = head + `<div class="xnote dim">A round picks a hidden neuron: one you've named, or one whose lens agrees on a single word. You see its replies to one prompt pushed up (▲) and pushed down (▼), then pick its label from four.</div>`; return; }
  const col = (c, k) => `<div class="col ${k}"><span class="lab">${k === "up" ? "▲" : "▼"} ${fmtDose(c ? -r.dose : r.dose)}</span>${r.replies[c].map(t => `<div class="g-reply txt">${esc(t)}</div>`).join("")}</div>`;
  const opts = r.options ? r.options.map((o, k) => {
    const cls = r.result ? (k === r.result.answer ? " right" : k === r.picked ? " wrong" : "") : "";
    return `<button type="button" class="g-opt${cls}" data-k="${k}"${r.result || r.live ? " disabled" : ""}>${esc(o)}</button>`;
  }).join("") : "";
  const res = r.result ? `<div class="xline">${r.result.correct ? "Right." : "Not this time."} It was ${neuronChip(r.result.truth)}, labeled from ${r.result.truth.source === "name" ? "your name for it" : "its logit lens"}.</div>` : "";
  el.innerHTML = head + `<div class="recipe mono dim">“${esc(r.prompt || "")}”${r.live ? " · writing…" : ""}</div>`
    + `<div class="turn"><div class="cols two">${col(0, "up")}${col(1, "down")}</div></div>`
    + (r.error ? `<div class="err">${esc(r.error)}</div>` : "")
    + `<div class="g-opts">${opts}</div>${r.live && r.options ? `<div class="xnote dim">answer once the replies finish</div>` : ""}${res}`;
}
async function newGuess() {
  if (X.job || G.round?.live) return;
  const r = G.round = { live: true, replies: [[], []], picked: null };
  renderGuess();
  let s;
  try { s = await api("/api/game/guess", {}); } catch (err) { s = { error: err.message }; }
  if (s.error) { Object.assign(r, { live: false, error: s.error }); renderGuess(); return; }
  X.job = { id: s.id };
  try {
    await sse(await fetch(`/api/experiment/events?id=${s.id}`), ev => {
      if (ev.error) r.error = ev.error;
      if (ev.round) { Object.assign(r, ev.round); r.replies = [Array(ev.round.samples).fill(""), Array(ev.round.samples).fill("")]; }
      if (ev.row != null) r.replies[ev.row][ev.s] += ev.t;
      if (ev.end) r.live = false;
      if (curTab() === "lab" && LAB.view === "guess") renderGuess();
    });
  } catch (err) { r.error = String(err); }
  finally { X.job = null; r.live = false; if (curTab() === "lab" && LAB.view === "guess") renderGuess(); }
}
async function answerGuess(k) {
  const r = G.round; if (!r || r.live || r.result) return;
  r.picked = k;
  try { r.result = await api("/api/game/guess/answer", { round: r.id, pick: k }); G.score = r.result.score; }
  catch (err) { r.error = err.message; }
  renderGuess();
}

// ---- field guide: a card per named neuron with your hypothesis, confidence and notes, and everything run on it ----
const FG = { sort: "layer", open: null, card: null };
const CONF = ["hunch", "supported", "replicated"];
async function renderGuide() {
  if (FG.open) return renderCard();
  const el = $("#lab-body");
  let r;
  try { r = await api("/api/guide"); } catch (err) { el.innerHTML = `<div class="err">${esc(err.message)}</div>`; return; }
  if (curTab() !== "lab" || LAB.view !== "guide" || FG.open) return;
  const by = { layer: (a, b) => a.layer - b.layer || a.n - b.n, confidence: (a, b) => CONF.indexOf(b.confidence) - CONF.indexOf(a.confidence) || a.layer - b.layer,
               index: (a, b) => (b.index ?? -1) - (a.index ?? -1) || a.layer - b.layer };
  const cards = r.cards.sort(by[FG.sort]);
  el.innerHTML = `<div class="nb-head"><span class="dim">${cards.length} card${cards.length === 1 ? "" : "s"}</span>
      <label class="xf">sort<select id="fg-sort">${["layer", "confidence", "index"].map(k => `<option value="${k}"${k === FG.sort ? " selected" : ""}>${k === "index" ? "therapeutic index" : k}</option>`).join("")}</select></label></div>`
    + (cards.length ? `<div class="fg-grid">${cards.map(c => `<button type="button" class="fg-card" data-l="${c.layer}" data-n="${c.n}">
        <span class="fg-name">${esc(c.name || "unnamed")}</span><span class="mono dim">L${c.layer} N${c.n} · <span class="conf-${c.confidence}">${c.confidence}</span></span>
        <span class="fg-hyp">${c.hypothesis ? esc(c.hypothesis) : `<span class="dim">no hypothesis yet</span>`}</span>
        <span class="mono dim">${c.index != null ? `index ${num(c.index)} · ` : ""}${c.experiments} experiment${c.experiments === 1 ? "" : "s"}</span></button>`).join("")}</div>`
      : `<div class="xnote dim">No cards yet. Naming a neuron (the name field at the top of its panel) gives it a card; so does its Details tab › field guide card.</div>`);
}
async function renderCard() {
  const el = $("#lab-body"), o = FG.open;
  if (!FG.card || FG.card.layer !== o.layer || FG.card.n !== o.n) el.innerHTML = `<div class="xnote dim">loading…</div>`;
  try { FG.card = await api(`/api/guide/card?layer=${o.layer}&n=${o.n}`); } catch (err) { el.innerHTML = `<div class="err">${esc(err.message)}</div>`; return; }
  if (curTab() === "lab" && LAB.view === "guide" && FG.open === o) drawCard();
}
function drawCard() {
  const c = FG.card, el = $("#lab-body"), sec = (t, body) => `<div class="fg-sec"><div class="lab">${t}</div>${body}</div>`;
  const sw = c.sweep, plot = sw ? sweepPlot({ doses: [...sw.doses].sort((a, b) => a - b), points: Object.fromEntries(sw.points.map(p => [p.dose, p])), measure: sw.measure })
    + `<div class="xnote dim">from the newest dose sweep: “${esc(sw.prompt)}”, ${sw.n} replies per dose</div>` : `<div class="xnote dim">No dose sweep yet: run one from the Chat tab.</div>`;
  const wins = Object.entries(c.windows).map(([k, v]) => k.startsWith("task: ")
    ? windowHTML(v.window, k.split(" · ").slice(1).join(" · "), k.slice(6), "answers") + `<div class="xnote dim">from the ${esc(v.prompt)} task</div>`
    : windowHTML(v.window, k) + `<div class="xnote dim">from a dose sweep on “${esc(v.prompt)}”</div>`).join("");
  const res = Object.entries(c.results).map(([m, r]) => `<div class="lcard">${[["steer", "steering · sufficient?"], ["ablate", "ablation · necessary?"]].map(([k, t]) =>
    `<div class="lcell"><div class="lab">${t} · ${esc(m)}</div>${r[k] ? `<div class="xline">${esc(r[k].sentence)}</div>` : `<div class="xnote dim">not run</div>`}</div>`).join("")}</div>`).join("");
  const tu = c.tuning ? (() => { const { svg, legend } = tuneSVG(c.tuning); return svg + legend + `<div class="xnote">item selectivity ${num(c.tuning.si)} · ${Object.entries(c.tuning.d).map(([k, v]) => `d' ${esc(k)} ${num(v)}`).join(" · ")}</div>`; })() : "";
  const ev = [...c.notes.map((x, k) => ({ ...x, k, t: x.time })), ...c.evidence.map(x => ({ ...x, t: x.timestamp }))].sort((a, b) => (b.t || "").localeCompare(a.t || ""));
  const KEEP = 12;  // the PNG keeps the newest entries; the page shows them all
  const log = ev.map((x, k) => (k < KEEP ? x : { ...x, older: true })).map(x => x.text != null
    ? `<div class="fg-ev${x.older ? " noprint" : ""}"><span class="mono dim">${esc(when(x.t))} · note</span> ${esc(x.text)} <button type="button" class="fg-rm dim noprint" data-k="${x.k}" aria-label="remove note">×</button></div>`
    : `<div class="fg-ev${x.older ? " noprint" : ""}"><span class="mono dim">${esc(when(x.t))} · ${esc(x.kind)}</span> ${esc(x.what)}${x.result ? `<div class="xline">${esc(x.result)}</div>` : ""}${x.prediction ? `<div class="xnote">prediction “${esc(x.prediction)}”${x.verdict ? ` · <span class="conf-${x.verdict}">${x.verdict}</span>` : ""}</div>` : ""}</div>`).join("");
  el.innerHTML = `<div class="fg-top noprint"><button type="button" id="fg-back" class="dim">‹ all cards</button><span class="xgap"></span><button type="button" id="fg-md" class="xbtn" title="download this card as Markdown">Markdown</button><button type="button" id="fg-png" class="xbtn" title="download this card as an image">PNG</button></div>
    <div id="fg-sheet">
      <div class="fg-title">${esc(c.name || `L${c.layer} N${c.n}`)}</div>
      <div class="nb-meta">${neuronChip(c)}<label class="xf">confidence<select id="fg-conf">${CONF.map(k => `<option${k === c.confidence ? " selected" : ""}>${k}</option>`).join("")}</select></label>${c.name ? "" : `<span class="dim noprint">name it in the neuron panel</span>`}</div>
      ${sec("hypothesis", `<textarea id="fg-hyp" rows="3" placeholder="what you think this neuron does, and what would show you wrong" spellcheck="false">${esc(c.hypothesis)}</textarea>`)}
      ${sec("dose-response", plot)}
      ${wins ? sec("therapeutic window", wins) : ""}
      ${res ? sec("steering vs ablation", res) : ""}
      ${tu ? sec(`tuning curve · ${esc(c.tuning.set)}`, tu) : ""}
      ${c.blind.n ? sec("blind rating", `<div class="xline">${esc(c.blind.sentence)}</div>`) : ""}
      ${sec("evidence log", `<div class="fg-add noprint"><input id="fg-note" placeholder="add a note: an observation, a replication, a doubt" spellcheck="false"><button type="button" id="fg-add" class="xbtn">add</button></div>${log}${ev.length > KEEP ? `<div class="xnote dim printonly">+ ${ev.length - KEEP} older entries in the notebook</div>` : ""}${log ? "" : `<div class="xnote dim">Nothing yet: experiments on this neuron and your notes land here.</div>`}`)}
    </div>`;
}
async function saveCard(change, redraw = true) {
  const o = FG.open; if (!o) return;
  try { const c = await api("/api/guide/card", { layer: o.layer, n: o.n, ...change }); if (FG.open === o) { FG.card = c; if (redraw) drawCard(); } }
  catch (err) { $("#status").textContent = err.message; }
}
const cardFile = ext => `L${FG.card.layer}_N${FG.card.n}${FG.card.name ? "_" + FG.card.name.replace(/[^\w-]+/g, "_").slice(0, 40) : ""}.${ext}`;
function download(blob, name) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
async function cardMarkdown() {
  const r = await fetch(`/api/guide/card.md?layer=${FG.card.layer}&n=${FG.card.n}`);
  download(new Blob([await r.text()], { type: "text/markdown" }), cardFile("md"));
}
// the card as a PNG: its text and plots drawn in reading order onto a canvas, plots as standalone SVG images with
// their computed styles inlined (the page's stylesheet doesn't reach an SVG drawn as an image)
async function cardPNG() {
  const sheet = $("#fg-sheet"), W = 720, PAD = 28, S = 2, ops = [];
  const scratch = document.createElement("canvas").getContext("2d");
  const fontOf = cs => `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  const wrap = (text, font, width) => {
    scratch.font = font; const lines = [];
    for (const para of text.split("\n")) {
      let line = "";
      for (const w of para.split(/(\s+)/)) {
        if (line && scratch.measureText(line + w).width > width) { lines.push(line.trimEnd()); line = w.trimStart(); } else line += w;
      }
      lines.push(line);
    }
    return lines;
  };
  const svgImage = async svg => {
    const clone = svg.cloneNode(true), src = [svg, ...svg.querySelectorAll("*")], dst = [clone, ...clone.querySelectorAll("*")];
    src.forEach((n, k) => { const cs = getComputedStyle(n);
      dst[k].setAttribute("style", ["fill", "stroke", "stroke-width", "stroke-dasharray", "opacity", "font-size", "font-family", "font-weight"].map(p => `${p}:${cs.getPropertyValue(p)}`).join(";")); });
    const vb = svg.viewBox.baseVal;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg"); clone.setAttribute("width", vb.width); clone.setAttribute("height", vb.height);
    const img = new Image();
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(clone));
    await img.decode();
    return { img, h: (W - 2 * PAD) * vb.height / vb.width };
  };
  const walk = async node => {
    if (node.nodeType !== 1 || node.classList.contains("noprint") || node.closest(".noprint")) return;
    if (node.tagName.toLowerCase() === "svg") { ops.push({ svg: await svgImage(node) }); return; }
    if (node.querySelector("svg") || (node.children.length && [...node.children].some(c => c.matches(".fg-sec, .lab, .xwin, .lcard, .lcell, .fg-ev, .xnote, .xline, .tlegend, .nb-meta")))) {
      for (const c of node.children) await walk(c);
      return;
    }
    const cs = getComputedStyle(node);
    const textOf = n => n.nodeType === 3 ? n.textContent : n.nodeType !== 1 || n.classList.contains("noprint") ? ""
      : n.tagName === "TEXTAREA" || n.tagName === "SELECT" ? ` ${n.value} ` : [...n.childNodes].map(textOf).join("") + (n.tagName === "SPAN" ? " " : "");
    const text = textOf(node).replace(/\s+/g, " ").trim();
    if (!text) return;
    const size = parseFloat(cs.fontSize), font = fontOf(cs);
    ops.push({ lines: wrap(text, font, W - 2 * PAD), font, color: cs.color, lh: size * 1.45, top: node.matches(".lab") ? 14 : 4 });
  };
  for (const c of sheet.children) await walk(c);
  const H = PAD * 2 + ops.reduce((h, o) => h + (o.svg ? o.svg.h + 8 : o.top + o.lines.length * o.lh), 0);
  const cv = document.createElement("canvas"); cv.width = W * S; cv.height = Math.ceil(H * S);
  const g = cv.getContext("2d"); g.scale(S, S);
  g.fillStyle = "#0d0f11"; g.fillRect(0, 0, W, H); g.textBaseline = "top";
  let y = PAD;
  for (const o of ops) {
    if (o.svg) { g.drawImage(o.svg.img, PAD, y, W - 2 * PAD, o.svg.h); y += o.svg.h + 8; continue; }
    y += o.top; g.font = o.font; g.fillStyle = o.color;
    for (const line of o.lines) { g.fillText(line, PAD, y + (o.lh - parseFloat(o.font.split(" ")[2])) / 2); y += o.lh; }
  }
  cv.toBlob(b => download(b, cardFile("png")), "image/png");
}

// ---------- learn: how a language model works, in short chapters run live on one sentence ----------
// Each chapter is a plain explanation, an optional "real terms" note with the math, one live picture, a key, a
// suggestion and a check question. Progress, the sentence and the real-terms switch are kept in this browser.
const LN = { ch: 0, real: false, done: new Set(), text: "The cat sat on the", gen: 0, toks: null, pick: {}, cache: {} };
try {
  const s = JSON.parse(localStorage.getItem("ne.learn") || "{}");
  Object.assign(LN, { ch: s.ch || 0, real: !!s.real, done: new Set(s.done || []), text: s.text || LN.text });
} catch {}
function saveLearn() {
  try { localStorage.setItem("ne.learn", JSON.stringify({ ch: LN.ch, real: LN.real, done: [...LN.done], text: LN.text })); } catch {}
}
const lnTok = s => esc(s.replace(/ /g, "·").replace(/\n/g, "⏎")) || "∅";
const lnPct = p => p >= 0.995 ? "100%" : p >= 0.1 ? `${Math.round(p * 100)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : "<0.1%";
const lnInt = n => Number(n).toLocaleString();
// the sentence's tokens and the model's shape, fetched once per sentence and shared by every chapter
async function lnTokens() {
  if (LN.toks?.text !== LN.text) LN.toks = { text: LN.text, ...(await api("/api/learn/tokens", { text: LN.text })) };
  return LN.toks;
}
// a row of the sentence's tokens to pick one from; `at` is the picked index
const tokPicker = (toks, at) => `<div class="ln-toks">${toks.map(([, s], i) =>
  `<button type="button" class="ln-tok${i === at ? " on" : ""}" data-p="${i}">${lnTok(s)}</button>`).join("")}</div>`;
const pickOf = (ch, toks, dflt) => { const p = LN.pick[ch]; return p != null && p < toks.length ? p : dflt; };

const CHAPTERS = [
  { id: "tokens", title: "Words become tokens",
    text: "A language model never sees letters or words. It first cuts your sentence into <b>tokens</b>: a common word is one token, a rare word is split into pieces, and the space before a word usually belongs to the word. Each token is then just a number, its place in a fixed list the model was trained with.",
    real: a => `This is <b>byte-pair encoding</b> (BPE): start from single bytes and keep merging the pair seen most often in training text. This model's vocabulary has <code>V = ${lnInt(a.vocab)}</code> entries, and a text becomes a list of ids <code>t₁ … t_T</code>, each <code>0 ≤ tᵢ &lt; V</code>.`,
    key: "Each box is one token, and the small number under it is its id. A · is a space that belongs to the token.",
    tryit: "Type a long, rare word like “antidisestablishmentarianism”, a typo, or a sentence in another language, and count the tokens.",
    quiz: { q: "Why can “ cat” and “cat” be different tokens?", a: 1,
      options: ["The model ignores capital letters", "The space in front is part of the token", "Tokens are always whole words"],
      why: "The leading space is stored inside the token, so “ cat” in the middle of a sentence and “cat” at the start get different ids." },
    async live(el, g) {
      const t = await lnTokens(); if (g !== LN.gen) return;
      const chars = [...LN.text].length;
      el.innerHTML = `<div class="ln-toks big">${t.tokens.map(([id, s]) => `<span class="ln-tok"><b>${lnTok(s)}</b><small>${id}</small></span>`).join("")}</div>
        <div class="xnote dim">${t.tokens.length} token${t.tokens.length === 1 ? "" : "s"} from ${chars} characters · a vocabulary of ${lnInt(t.vocab)} tokens</div>`;
    } },
  { id: "embed", title: "Tokens become lists of numbers",
    text: "A number like 7993 means nothing by itself, so the model looks each token up in a table and swaps it for a long list of numbers, a <b>vector</b>. Tokens that get used in similar ways end up with similar vectors, so meaning turns into position: “cat” sits near “cats”, near other spellings of cat, and even near the Chinese word for cat.",
    real: a => `The <b>embedding</b> is one row of a matrix, <code>x = E[t]</code> with <code>E ∈ ℝ^(V×d)</code> and <code>d = ${lnInt(a.hidden)}</code> numbers per token. Closeness is the <b>cosine similarity</b>, <code>cos(a, b) = a·b / (‖a‖ ‖b‖)</code>: 1 for the same direction, near 0 for unrelated ones.`,
    key: "Pick a token. The bars show the tokens whose vectors point most nearly the same way (cosine similarity; 1 would be the same direction). Variants that differ only in capitals or spaces are folded together.",
    tryit: "Pick a common word, then a name or a number. Which neighbors are about meaning, and which are only about spelling?",
    quiz: { q: "What makes two tokens neighbors here?", a: 2,
      options: ["They sit next to each other in your sentence", "Their ids are close together", "Their vectors point in nearly the same direction"],
      why: "Neighbors are measured by the angle between the two vectors, not by position or id." },
    async live(el, g) {
      const t = await lnTokens(); if (g !== LN.gen) return;
      const longest = t.tokens.reduce((b, [, s], i) => s.trim().length > t.tokens[b][1].trim().length ? i : b, 0);
      const at = pickOf("embed", t.tokens, longest), id = t.tokens[at][0];
      el.innerHTML = tokPicker(t.tokens, at) + `<div class="xnote dim">looking up ${lnTok(t.tokens[at][1])}…</div>`;
      const key = `nb/${id}`, r = LN.cache[key] ||= await api("/api/learn/neighbors", { id }); if (g !== LN.gen) return;
      el.innerHTML = tokPicker(t.tokens, at) + `<div class="lab">nearest to ${lnTok(r.token)} among ${lnInt(t.vocab)} tokens, each a list of ${lnInt(r.hidden)} numbers</div>`
        + `<div class="ln-bars">${r.neighbors.map(([s, c]) => `<div class="ln-bar"><span class="mono">${lnTok(s)}</span><i style="width:${Math.max(2, c * 100)}%"></i><b class="mono">${c.toFixed(2)}</b></div>`).join("")}</div>`;
    } },
  { id: "layers", title: "Passing through the layers",
    text: "Those vectors now pass through 32 <b>layers</b>. Each layer reads the running vector at every token, adds a small correction and passes it on, like editors marking up the same draft in turn. We can peek at the draft after any layer and ask: if the model had to name the next word right now, what would it say? The early layers look like noise; the answer usually forms in the last few.",
    real: () => `The running vector is the <b>residual stream</b>: <code>h(l+1) = h(l) + attn_l(h) + mlp_l(h)</code>. The <b>logit lens</b> reads it at layer <code>l</code> as <code>softmax(W_U · norm(h(l)))</code>, borrowing the final norm and the unembedding <code>W_U</code> that the model only meant for the last layer, so the early rows are a rough, biased reading.`,
    key: "One row per layer, first at the top. The word is that layer's top guess for the next token. The bar is how much that layer already believes the model's final answer. ◆ marks layers with full attention (next chapter). Click a row to fly the map to that layer.",
    tryit: "Pick another word of your sentence to see the guesses at that point. Then try a sentence with an obvious ending, like “The capital of France is”.",
    quiz: { q: "Why do the early layers' guesses look like random tokens?", a: 0,
      options: ["The meaning is still being built, and the lens reads it with tools meant for the last layer", "The early layers are broken", "The early layers only see the first word"],
      why: "Every layer adds to the same running vector; the lens just reads it early, before it's in a shape the output step understands." },
    async live(el, g) {
      const t = await lnTokens(); if (g !== LN.gen) return;
      const at = pickOf("layers", t.tokens, t.tokens.length - 1);
      el.innerHTML = tokPicker(t.tokens, at) + `<div class="xnote dim">reading all 32 layers…</div>`;
      const key = `lens/${LN.text}/${at}`, r = LN.cache[key] ||= await api("/api/learn/lens", { text: LN.text, pos: at }); if (g !== LN.gen) return;
      const [w, p] = r.layers[r.layers.length - 1].top[0];
      el.innerHTML = tokPicker(t.tokens, at) + `<div class="lab">after “…${lnTok(r.token)}” the model's final answer is <b class="ln-ans">${lnTok(r.answer)}</b> (${lnPct(p)})</div>`
        + `<div class="ln-layers">${r.layers.map((x, l) => `<div class="ln-layer" data-l="${l}" title="layer ${l}: ${x.top.map(([s, q]) => `${s.replace(/\n/g, "⏎")} ${lnPct(q)}`).join(" · ")}">
          <span class="mono dim">L${l}</span><span class="dim">${r.types[l] === "full_attention" ? "◆" : ""}</span>
          <span class="mono${x.top[0][0] === r.answer ? " ln-ans" : ""}">${lnTok(x.top[0][0])}</span><span class="mono dim">${lnPct(x.top[0][1])}</span>
          <span class="ln-meter"><i style="width:${Math.max(0.5, x.final * 100)}%"></i></span></div>`).join("")}</div>`;
    } },
  { id: "attention", title: "Looking back: attention",
    text: "So far each token's vector has moved on its own. <b>Attention</b> is how tokens share information: at a layer, each token can look back at the earlier tokens and copy in what it needs, so “she” can pull in “Alice”, or “sat” can check who is sitting. A layer has many <b>heads</b>, each looking for something different. A token can only look back, never ahead, because the model writes left to right.",
    real: a => `Full attention: each head makes queries <code>Q</code>, keys <code>K</code> and values <code>V</code> from the residual and outputs <code>softmax(QKᵀ / √d_k) · V</code>, with a causal mask so token <code>i</code> only sees tokens <code>j ≤ i</code>. Qwen3.5 is a hybrid: ${a.layer_types ? a.layer_types.filter(t => t === "full_attention").length : 8} of its ${a.n_layers || 32} layers use full attention and the rest use <b>linear attention</b> (Gated DeltaNet), which keeps a fixed-size running state instead of weighing every past token. Heads often park spare weight on the first token (the <b>attention sink</b>), so that token is left out when picking heads to show.`,
    key: "Pick a word. Below are the 3 heads, from the layers with full attention, whose attention from that word lands hardest on one earlier word. The shading shows how much weight each earlier token gets; the selected word is outlined.",
    tryit: "Type “Alice gave Bob the book because she” and pick “she”. Do any heads look back at Alice or Bob?",
    quiz: { q: "Can a token attend to a word that comes after it?", a: 1,
      options: ["Yes, every token sees the whole sentence", "No, attention only looks back at earlier tokens", "Only in the last layer"],
      why: "The model writes one token at a time, so when a token is processed the words after it don't exist yet; a causal mask enforces this." },
    async live(el, g) {
      const t = await lnTokens(); if (g !== LN.gen) return;
      const at = pickOf("attention", t.tokens, t.tokens.length - 1);
      el.innerHTML = tokPicker(t.tokens, at) + `<div class="xnote dim">finding heads…</div>`;
      const key = `att/${LN.text}/${at}`, r = LN.cache[key] ||= await api("/api/learn/attention", { text: LN.text, pos: at }); if (g !== LN.gen) return;
      if (at === 0) { el.innerHTML = tokPicker(t.tokens, at) + `<div class="xnote dim">The first token has nothing earlier to look at. Pick a later word.</div>`; return; }
      const h = r.heads[Math.min(LN.head || 0, r.heads.length - 1)];
      el.innerHTML = tokPicker(t.tokens, at) + `<div class="lab">heads reading back from ${lnTok(r.tokens[at])}, out of ${r.full_layers * r.heads_per_layer} in the ${r.full_layers} full-attention layers</div>
        <div class="seg ln-heads">${r.heads.map((x, i) => `<button type="button" data-h="${i}" class="${x === h ? "on" : ""}">L${x.layer} head ${x.head}</button>`).join("")}</div>
        <div class="ln-toks ln-att">${r.tokens.slice(0, at + 1).map((s, i) => `<span class="ln-tok${i === at ? " q" : ""}" style="background:rgba(56,200,255,${(i === at ? 0 : Math.min(0.9, h.row[i] * 1.1)).toFixed(3)});${h.row[i] > 0.45 && i !== at ? "color:#000" : ""}"><b>${lnTok(s)}</b><small>${i === at ? "here" : lnPct(h.row[i])}</small></span>`).join("")}</div>
        <div class="xnote dim">${h.row[0] > 0.3 ? `This head also parks ${lnPct(h.row[0])} on the first token, the attention sink.` : ""}</div>`;
    } },
  { id: "neurons", title: "Neurons: detectors and writers",
    text: "After attention, each layer runs its <b>MLP</b>: 9,216 neurons per layer, 294,912 in all. Each neuron checks the running vector for a pattern and fires when it finds one, and when it fires it adds its own fixed direction back into the vector. So a neuron is both a <b>detector</b> (what makes it fire) and a <b>writer</b> (what it pushes toward). The map now lights up the neurons that fire hardest on your sentence.",
    real: () => `Qwen's MLP is gated: <code>a = silu(W_gate·x) ⊙ (W_up·x)</code> gives one value per neuron, and the layer writes <code>W_down·a</code>, so neuron <code>n</code> adds <code>a_n</code> times column <code>n</code> of <code>W_down</code>. The words beside a neuron are its <b>logit lens</b>: that column read through the unembedding, <code>W_U · norm(W_down[:, n])</code>. “Fires” here is the activation divided by the neuron's usual peak on web text.`,
    key: "Each row is a neuron: its layer and number, the token it fired hardest on, how hard (compared with its usual peak, so above 1 is unusually strong) and the words its direction pushes toward. Click one to open it. The lit dots on the map are all the neurons above the threshold, the same list as the run panel on the left.",
    tryit: "Rewrite the sentence about food, code or a city and see which neurons change. Open one and read its Details tab.",
    quiz: { q: "What does a neuron do when it fires?", a: 2,
      options: ["It picks the next word by itself", "It removes earlier words from the sentence", "It adds its own direction to the running vector"],
      why: "A neuron's output is a fixed direction scaled by how hard it fires, added into the residual stream for later layers to read." },
    async live(el, g) {
      $('#q-mode button[data-q="fires"]').click();
      $("#lens-q").value = LN.text; $("#inf-text").value = LN.text; showLeft("run");
      await runLensSearch(); if (g !== LN.gen) return;
      const rs = fireResults?.results;
      if (!rs) { el.innerHTML = `<div class="xnote dim">${esc(state.statusText || "The neurons' usual ranges are still being measured; try again in a minute.")}</div>`; return; }
      el.innerHTML = `<div class="lab">${lnInt(fireResults.total)} neurons fire above their threshold on this sentence · the strongest:</div>
        <table class="dtable ln-fires">${rs.slice(0, 10).map(x => `<tr><td>${neuronChip(x)}</td><td class="mono">${lnTok(x.token)}</td>
          <td class="mono ${x.sign === "+" ? "up" : "down"}">${x.sign === "+" ? "" : "−"}${x.score}×</td><td class="mono dim">${esc(x.name || x.lens.slice(0, 4).join(" "))}</td></tr>`).join("")}</table>`;
    } },
  { id: "next", title: "Choosing the next word",
    text: "After the last layer, the model scores every token in its vocabulary and turns the scores into <b>probabilities</b>. Then it picks one, adds it to the text and runs everything again for the word after. That is all generation is: one token at a time. <b>Temperature</b> sets how adventurous the pick is: low almost always takes the top choice, high spreads the chances out.",
    real: () => `The logits are <code>z = W_U · norm(h(L))</code> and the probabilities <code>p = softmax(z / T)</code>, i.e. <code>pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)</code>. <code>T → 0</code> is <b>greedy decoding</b> (always the argmax), <code>T = 1</code> is the model's own distribution, and <code>T &gt; 1</code> flattens it. <b>Sampling</b> draws one token from <code>p</code>.`,
    key: "The 10 likeliest next tokens and the chance left for everything else, at this temperature. Sample draws one token from the whole distribution; add puts it on the end of your sentence, which reruns every chapter.",
    tryit: "Sample a few times at 0.2, then at 1.5. Add a word and watch the next guesses change.",
    quiz: { q: "What does raising the temperature do?", a: 0,
      options: ["Moves probability toward less likely words", "Makes the model more accurate", "Changes the model's weights"],
      why: "Dividing the scores by a larger T shrinks the gaps between them, so the softmax spreads probability more evenly." },
    async live(el, g) {
      const T = LN.temp ??= 1;
      const r = await api("/api/learn/next", { text: LN.text, temperature: T }); if (g !== LN.gen) return;
      const top = r.top[0][1] || 1, samples = (LN.samples?.text === LN.text ? LN.samples.list : []);
      // the controls are drawn once, so a slider being dragged is never replaced under the pointer
      if (!el.querySelector("#ln-t")) el.innerHTML = `<div class="ln-temp"><span class="lab">temperature</span><input type="range" id="ln-t" min="0.1" max="2" step="0.1" value="${T}"><b class="mono">${T.toFixed(1)}</b>
          <button type="button" class="xbtn" id="ln-sample">sample</button></div><div class="ln-out"></div>`;
      el.querySelector(".ln-out").innerHTML = `<div class="lab">after “${esc(LN.text.slice(-40))}”</div>
        <div class="ln-bars">${r.top.map(([s, p]) => `<div class="ln-bar"><span class="mono">${lnTok(s)}</span><i style="width:${Math.max(0.5, p / top * 100)}%"></i><b class="mono">${lnPct(p)}</b></div>`).join("")}
          <div class="ln-bar dim"><span>everything else</span><span></span><b class="mono">${lnPct(r.rest)}</b></div></div>
        ${samples.length ? `<div class="ln-samples"><span class="lab">sampled</span>${samples.map((s, i) => `<button type="button" class="ln-tok" data-add="${i}" title="add to the sentence">${lnTok(s.tok)} <small class="dim">T ${s.t.toFixed(1)}</small></button>`).join("")}<span class="dim">click one to add it</span></div>` : ""}`;
    } },
  { id: "steer", title: "Steering: change one neuron",
    text: "Because a neuron writes a fixed direction, we can write it ourselves: push the neuron at every word and see what the model says. This is <b>L22 N2742</b>, a neuron whose direction leans toward words like “alien” and “extra”. Push it while the model answers “Who are you?” and compare with the normal answer. A small push nudges the topic; too big a push breaks the text.",
    real: () => `Steering adds <code>d · s · ŵ</code> to the residual leaving layer 22 at every position, where <code>ŵ</code> is the neuron's <code>W_down</code> column as a unit vector and <code>s</code> is the median size of that residual, so the dose <code>d</code> is a fraction of a typical residual. Pushing toward 1 and beyond starts to overwrite what was there, and the text falls apart.`,
    key: "Three answers to “Who are you?”: pushed up, normal, and pushed the other way. All use greedy decoding, so the push is the only difference.",
    tryit: "Try 0.3, 0.7 and 1.2. Where does the topic change, and where does the text break? Then open the neuron to run real experiments in Chat and the Lab.",
    quiz: { q: "Why is the normal answer the same every time you run it here?", a: 1,
      options: ["The model remembers its last answer", "Greedy decoding always takes the top token, so nothing is random", "The push is always zero"],
      why: "With temperature 0 the model always takes its single most likely token, so the same input always gives the same text." },
    async live(el, g) {
      const d = LN.dose ??= 0.6, key = `steer/${d}`, out = LN.cache[key];
      const col = (r, label) => `<div class="ln-col"><div class="lab">${label}</div><div class="ln-reply">${esc(out?.[r] || "")}${out && !out.done ? "▍" : ""}</div></div>`;
      if (!el.querySelector("#ln-d")) el.innerHTML = `<div class="ln-temp"><span class="lab">push</span><input type="range" id="ln-d" min="0.1" max="1.5" step="0.1" value="${d}"><b class="mono">${d.toFixed(1)}</b>
          <button type="button" class="xbtn" id="ln-steer"></button><button type="button" class="xbtn" id="ln-open">open L22 N2742 in Chat ›</button></div><div class="ln-out"></div>`;
      el.querySelector("#ln-steer").textContent = out && !out.done ? "running…" : out ? "run again" : "run";
      el.querySelector(".ln-out").innerHTML = `${out ? col(0, `pushed +${d.toFixed(1)}`) + col(1, "normal") + col(2, `pushed −${d.toFixed(1)}`) : `<div class="xnote dim">Press run to generate the three answers (about 10 seconds).</div>`}`;
    } },
];

function renderLearn() {
  const c = CHAPTERS[LN.ch] || CHAPTERS[0];
  if ($("#ln-text").value !== LN.text && document.activeElement !== $("#ln-text")) $("#ln-text").value = LN.text;
  $("#ln-real").checked = LN.real;
  $("#ln-nav").innerHTML = CHAPTERS.map((x, i) => `<button type="button" data-c="${i}" class="${i === LN.ch ? "on" : ""}" title="${esc(x.title)}">${LN.done.has(x.id) ? "✓" : i + 1}</button>`).join("");
  const q = c.quiz, arch = LN.toks?.arch ? { ...LN.toks.arch, vocab: LN.toks.vocab } : { vocab: 248077, hidden: 2560 };
  $("#ln-body").innerHTML = `<div class="lab">chapter ${LN.ch + 1} of ${CHAPTERS.length}</div><h3 class="ln-title">${esc(c.title)}</h3>
    <p class="ln-text">${c.text}</p>${LN.real ? `<div class="ln-real"><div class="lab">real terms</div>${c.real(arch)}</div>` : ""}
    <div id="ln-live" class="ln-live"></div>
    <div class="ln-note"><div class="lab">what you're looking at</div>${c.key}</div>
    <div class="ln-note"><div class="lab">try this</div>${c.tryit}</div>
    <div class="ln-quiz"><div class="lab">check</div><div>${esc(q.q)}</div>${q.options.map((o, i) => `<button type="button" class="ln-opt" data-o="${i}">${esc(o)}</button>`).join("")}<div class="ln-fb"></div></div>
    <div class="ln-foot">${LN.ch > 0 ? `<button type="button" class="xbtn" data-go="-1">‹ ${esc(CHAPTERS[LN.ch - 1].title)}</button>` : "<span></span>"}${LN.ch < CHAPTERS.length - 1 ? `<button type="button" class="xbtn" data-go="1">${esc(CHAPTERS[LN.ch + 1].title)} ›</button>` : ""}</div>`;
  renderLearnLive();
}
async function renderLearnLive() {
  const g = ++LN.gen, el = $("#ln-live"), c = CHAPTERS[LN.ch];
  if (!el.innerHTML) el.innerHTML = `<div class="xnote dim">running your sentence through the model…</div>`;
  try { await c.live(el, g); }
  catch (err) { if (g === LN.gen) el.innerHTML = `<div class="err">${esc(err.message)}</div>`; }
  // the real-terms note may quote sizes that only arrive with the first tokens call
  if (g === LN.gen && LN.real && LN.toks?.arch) $(".ln-real").innerHTML = `<div class="lab">real terms</div>${c.real({ ...LN.toks.arch, vocab: LN.toks.vocab })}`;
}
function showLearn(ch) {
  if (ch != null) { LN.ch = ch; saveLearn(); }
  if (curTab() === "learn") renderLearn(); else { focusEl.classList.add("on"); syncTabs(); setTab("learn"); setFocusCollapsed(false); }
  $("#ln-body").scrollTop = 0;
}
$("#learn-btn").onclick = () => showLearn();
$("#ln-nav").onclick = e => { const b = e.target.closest("button"); if (b) showLearn(+b.dataset.c); };
$("#ln-real").onchange = e => { LN.real = e.target.checked; saveLearn(); renderLearn(); };
let lnTimer;
$("#ln-text").addEventListener("input", e => {
  clearTimeout(lnTimer);
  lnTimer = setTimeout(() => { const t = e.target.value.trim(); if (!t || t === LN.text) return; LN.text = t; LN.pick = {}; saveLearn(); renderLearnLive(); }, 600);
});
$("#ln-body").addEventListener("click", e => {
  const go = e.target.closest("[data-go]"); if (go) return showLearn(LN.ch + +go.dataset.go);
  const tk = e.target.closest("button.ln-tok"); if (tk) { LN.pick[CHAPTERS[LN.ch].id] = +tk.dataset.p; return renderLearnLive(); }
  const lay = e.target.closest(".ln-layer"); if (lay) { if (map.layerMode !== null) map.exitLayer(); map.viewLayer(+lay.dataset.l); return; }
  const o = e.target.closest(".ln-opt");
  if (o) {
    const c = CHAPTERS[LN.ch], ok = +o.dataset.o === c.quiz.a;
    o.closest(".ln-quiz").querySelectorAll(".ln-opt").forEach(b => b.classList.toggle("picked", b === o));
    o.classList.add(ok ? "right" : "wrong");
    $(".ln-fb").innerHTML = ok ? `Right. ${esc(c.quiz.why)}` : "Not quite. Try another answer.";
    if (ok && !LN.done.has(c.id)) { LN.done.add(c.id); saveLearn(); $(`#ln-nav button[data-c="${LN.ch}"]`).textContent = "✓"; }
  }
});

// chapter 7's three answers stream in; each delta repaints only the chapter's live picture
async function learnSteer() {
  const d = LN.dose, key = `steer/${d}`, out = LN.cache[key] = { 0: "", 1: "", 2: "", done: false };
  LN.steerCtl?.abort(); const ctl = LN.steerCtl = new AbortController();
  const paint = () => { if (CHAPTERS[LN.ch].id === "steer" && curTab() === "learn") renderLearnLive(); };
  paint();
  try {
    const res = await fetch("/api/steer", { method: "POST", headers: { "Content-Type": "application/json" }, signal: ctl.signal,
      body: JSON.stringify({ layer: 22, n: 2742, prompt: "Who are you?", dose: d, mode: "add", samples: 1, temperature: 0, max_new: 70 }) });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 160)}`);
    let last = 0;
    await sse(res, ev => {
      if (ev.error) out[1] = ev.error;
      if (ev.row == null) return;
      out[ev.row] += ev.t;
      if (performance.now() - last > 120) { last = performance.now(); paint(); }
    });
  } catch (err) { if (err.name !== "AbortError") out[1] = String(err.message || err); }
  out.done = true; paint();
}
let lnTempTimer;
$("#ln-body").addEventListener("input", e => {
  if (e.target.id === "ln-t") {
    LN.temp = +e.target.value; e.target.nextElementSibling.textContent = LN.temp.toFixed(1);
    clearTimeout(lnTempTimer); lnTempTimer = setTimeout(renderLearnLive, 120);
  }
  if (e.target.id === "ln-d") { LN.dose = +e.target.value; e.target.nextElementSibling.textContent = LN.dose.toFixed(1); renderLearnLive(); }
});
$("#ln-body").addEventListener("click", async e => {
  const hb = e.target.closest("[data-h]"); if (hb) { LN.head = +hb.dataset.h; return renderLearnLive(); }
  const nc = e.target.closest(".nchip"); if (nc) return goToNeuron(+nc.dataset.l, +nc.dataset.n);
  if (e.target.id === "ln-sample") {
    const r = await api("/api/learn/next", { text: LN.text, temperature: LN.temp ?? 1, sample: true });
    if (LN.samples?.text !== LN.text) LN.samples = { text: LN.text, list: [] };
    LN.samples.list = [{ tok: r.sampled, t: r.temperature }, ...LN.samples.list].slice(0, 8);
    return renderLearnLive();
  }
  const add = e.target.closest("[data-add]");
  if (add) {
    LN.text += LN.samples.list[+add.dataset.add].tok; LN.pick = {}; saveLearn();
    $("#ln-text").value = LN.text; return renderLearnLive();
  }
  if (e.target.id === "ln-steer" && !(LN.cache[`steer/${LN.dose}`] && !LN.cache[`steer/${LN.dose}`].done)) return learnSteer();
  if (e.target.id === "ln-open") { goToNeuron(22, 2742); setTab("chat"); }
});

// ---------- tuning curves: a stimulus set's items, each neuron's response per item, two selectivity scores ----------
const T = { sets: [], scores: {}, pending: new Set(), running: false };
const tuneSet = () => T.sets.find(s => s.name === $("#t-set").value) || T.sets[0] || null;
const tuneCat = () => $("#t-cat").value;
async function loadTuning(pick) {
  try { T.sets = (await api("/api/tuning/sets")).sets; } catch { return; }
  let saved = pick ?? null; if (saved == null) try { saved = localStorage.getItem("ne.tset"); } catch {}
  $("#t-set").innerHTML = T.sets.map(s => `<option>${esc(s.name)}</option>`).join("");
  if (T.sets.some(s => s.name === saved)) $("#t-set").value = saved;
  setCats(); T.scores = {}; renderTuning();
}
function setCats() {
  const st = tuneSet(), cats = st ? [...new Set(st.items.map(i => i.category))] : [], keep = $("#t-cat").value;
  $("#t-cat").innerHTML = cats.map(c => `<option>${esc(c)}</option>`).join("");
  if (cats.includes(keep)) $("#t-cat").value = keep;
}
$("#t-set").onchange = () => { try { localStorage.setItem("ne.tset", $("#t-set").value); } catch {} setCats(); T.scores = {}; renderTuning(); if (["si", "dprime"].includes($("#sort").value)) renderList(); };
$("#t-cat").onchange = () => { T.scores = {}; renderTuning(); if ($("#sort").value === "dprime") renderList(); };
// a layer's scores for the list sort; null until fetched (the list redraws when they arrive)
function tuneScores(layer) {
  const st = tuneSet(); if (!st?.ready) return null;
  const key = `${st.name}/${tuneCat()}/${layer}`;
  if (T.scores[key]) return T.scores[key];
  if (!T.pending.has(key)) {
    T.pending.add(key);
    api(`/api/tuning/scores?layer=${layer}&set=${encodeURIComponent(st.name)}&category=${encodeURIComponent(tuneCat())}`)
      .then(r => { if (r.ready) { T.scores[key] = r; if (state.layer === layer) renderList(); } }).finally(() => T.pending.delete(key));
  }
  return null;
}
const CAT_COLORS = ["var(--up)", "var(--link)", "var(--down)", "#73ff8c", "#b399ff", "#e0e0e0"];
async function renderTuning() {
  const el = $("#t-body"), f = state.focus, st = tuneSet();
  if (!el || !f || f.kind !== "neuron") return;
  if (!st) { el.innerHTML = `<div class="xnote dim">No stimulus sets. Press edit to make one.</div>`; return; }
  if (!st.ready) {
    el.innerHTML = `<div class="xnote">“${esc(st.name)}” hasn't been run yet. Running it passes each of its ${st.items.length} items through the model once (no generation) and scores every neuron at the same time.</div>
      <button type="button" id="t-run" class="xbtn">${T.running ? "running…" : "run this set"}</button>`;
    return;
  }
  const r = await api(`/api/tuning/neuron?layer=${f.layer}&n=${f.n}&set=${encodeURIComponent(st.name)}`).catch(() => null);
  if (!r?.ready || !isFocus(f.layer, f.n, "neuron") || tuneSet() !== st) return;
  const { cats, svg, legend } = tuneSVG(r);
  const cat = cats.includes(tuneCat()) ? tuneCat() : cats[0], d = r.d[cat];
  el.innerHTML = `${svg}
    <div class="xnote dim">bar: the neuron's strongest |activation ÷ typical extreme| on the item's own tokens${st.template ? ` in “${esc(st.template)}”` : ""}</div>
    ${legend}
    <div class="xs tstats"><div title="(r_max − r_others) ÷ (r_max + r_others), r_others the mean of every other item"><span class="k">item selectivity</span><span>${num(r.si)}<span class="ci">0 responds evenly to every item, 1 to one item only${r.si > 0.5 ? ` (here: ${esc(r.items.reduce((a, b) => b.r > a.r ? b : a).text)})` : ""}</span></span></div>
      <div title="(mean on ${esc(cat)} − mean on the rest) ÷ pooled SD"><span class="k">d' ${esc(cat)}</span><span>${num(d)}<span class="ci">how far above the other items ${esc(cat)} items sit, in SDs (0.8 is large)</span></span></div></div>
    <div class="xline">${esc(tuneReading(r.si, d, cat))}</div>`;
}
// a tuning curve's bars, one per item, grouped and colored by category, and the category legend
function tuneSVG(r) {
  const cats = [...new Set(r.items.map(i => i.category))], col = c => CAT_COLORS[cats.indexOf(c) % CAT_COLORS.length];
  const items = cats.flatMap(c => r.items.filter(i => i.category === c));  // grouped by category
  const W = 560, H = 190, L = 34, B = 54, bw = (W - L - 8) / items.length, top = Math.max(1e-6, ...items.map(i => i.r));
  const Y = v => 10 + (H - B - 10) * (1 - v / top);
  const bars = items.map((it, k) => `<rect x="${L + k * bw + 2}" y="${Y(it.r)}" width="${bw - 4}" height="${H - B - Y(it.r)}" fill="${col(it.category)}"><title>${esc(it.text)}: ${num(it.r)}</title></rect>
    <text class="tk" transform="translate(${L + k * bw + bw / 2 + 3},${H - B + 8}) rotate(50)">${esc(it.text)}</text>`).join("");
  return { cats, svg: `<svg class="xplot" viewBox="0 0 ${W} ${H}" role="img" aria-label="tuning curve">
      <line class="ax" x1="${L}" x2="${W - 4}" y1="${H - B}" y2="${H - B}"/><text class="tk" x="${L - 6}" y="${Y(top) + 4}" text-anchor="end">${num(top)}</text><text class="tk" x="${L - 6}" y="${H - B + 4}" text-anchor="end">0</text>${bars}</svg>`,
    legend: `<div class="tlegend">${cats.map(c => `<span><i style="background:${col(c)}"></i>${esc(c)}</span>`).join("")}</div>` };
}
function tuneReading(si, d, cat) {
  if (d > 0.8 && si > 0.5) return `Responds to ${cat} and to one item well above the rest: item-selective as well as category-selective.`;
  if (d > 0.8) return `Responds to ${cat} items generally (a category response) without singling out one item.`;
  if (si > 0.5) return `Singles out one item, but not ${cat} as a category.`;
  return `No clear preference for ${cat} or for any one item in this set.`;
}
$("#t-body").addEventListener("click", async e => {
  if (!e.target.closest("#t-run") || T.running) return;
  const st = tuneSet(); T.running = true; renderTuning();
  const r = await api("/api/tuning/run", { set: st.name }).catch(err => ({ error: err.message }));
  T.running = false;
  if (r.error) { $("#t-body").innerHTML = `<div class="err">${esc(r.error)}</div>`; return; }
  loadTuning(st.name);
  if (["si", "dprime"].includes($("#sort").value)) renderList();
});
// the stimulus set editor: name, optional template with {item}, one "item, category" per line
function renderTuneEditor(st = tuneSet() || { name: "", template: "Let's talk about {item}.", items: [] }) {
  $("#t-editor").innerHTML = `<div class="row"><input id="te-name" placeholder="set name" value="${esc(st.name)}" spellcheck="false"><input id="te-tpl" placeholder="template, e.g. Let's talk about {item}." value="${esc(st.template || "")}" spellcheck="false"></div>
    <textarea id="te-items" rows="6" spellcheck="false" placeholder="one per line: item, category">${esc(st.items.map(i => `${i.text}, ${i.category}`).join("\n"))}</textarea>
    <div class="row"><button type="button" id="te-save">save</button><button type="button" id="te-new" class="dim">new</button><button type="button" id="te-del" class="dim">delete</button><span id="te-msg" class="dim"></span><button type="button" id="te-close" class="dim">close</button></div>`;
  $("#t-editor").dataset.orig = st.name;
}
async function saveSets(sets, pick) {
  const r = await api("/api/tuning/sets", { sets }).catch(err => ({ error: err.message }));
  if (r.error) { $("#te-msg").textContent = r.error; return; }
  await loadTuning(pick); renderTuneEditor();
}
$("#t-editor").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  const orig = $("#t-editor").dataset.orig, strip = s => ({ name: s.name, template: s.template, items: s.items });
  if (b.id === "te-close") $("#t-editor").hidden = true;
  if (b.id === "te-new") renderTuneEditor({ name: "", template: "Let's talk about {item}.", items: [] });
  if (b.id === "te-del") saveSets(T.sets.filter(s => s.name !== orig).map(strip), "");
  if (b.id === "te-save") {
    const name = $("#te-name").value.trim();
    if (!name) { $("#te-msg").textContent = "name it first"; return; }
    const items = $("#te-items").value.split("\n").map(l => l.split(",")).filter(p => p[0].trim())
      .map(([t, ...c]) => ({ text: t.trim(), category: c.join(",").trim() || "other" }));
    const st = { name, template: $("#te-tpl").value.trim(), items };
    const list = T.sets.filter(s => s.name !== orig && s.name !== name).map(strip), at = T.sets.findIndex(s => s.name === orig);
    list.splice(at < 0 ? list.length : Math.min(at, list.length), 0, st);
    saveSets(list, name);
  }
});
$("#t-edit").onclick = () => { const ed = $("#t-editor"); ed.hidden = !ed.hidden; if (!ed.hidden) renderTuneEditor(); };

// ---------- lesion: the neuron (or the pinned set) ablated against the intact model, paired by seed ----------
let pinned = []; try { pinned = JSON.parse(localStorage.getItem("ne.set") || "[]"); } catch {}
const inSet = (l, n) => pinned.some(x => x.layer === l && x.n === n);
function renderSet() {
  try { localStorage.setItem("ne.set", JSON.stringify(pinned)); } catch {}
  const f = state.focus, on = f?.kind === "neuron" && inSet(f.layer, f.n);
  $("#f-pin").style.display = f?.kind === "neuron" ? "" : "none";
  $("#f-pin").textContent = on ? "− set" : "+ set";
  $("#f-pin").classList.toggle("on", on);
  const el = $("#x-set"); el.hidden = !pinned.length;
  el.innerHTML = pinned.length ? `<span class="xk">set</span>${pinned.map(x => `<span class="chip" data-l="${x.layer}" data-n="${x.n}" title="open">L${x.layer} N${x.n}</span>`).join("")}
    <label class="xf" title="ablate every neuron in the set together; single-neuron ablations often show nothing when other neurons carry the same signal"><input type="checkbox" id="x-ablate-set"${X.ablateSet ? " checked" : ""}> ablate the whole set</label><button type="button" id="x-set-clear" class="dim">clear set</button>` : "";
}
$("#f-pin").onclick = () => {
  const f = state.focus; if (f?.kind !== "neuron") return;
  pinned = inSet(f.layer, f.n) ? pinned.filter(x => !(x.layer === f.layer && x.n === f.n)) : [...pinned, { layer: f.layer, n: f.n }];
  renderSet();
};
$("#x-set").addEventListener("click", e => {
  if (e.target.id === "x-set-clear") { pinned = []; X.ablateSet = false; renderSet(); return; }
  if (e.target.id === "x-ablate-set") { X.ablateSet = e.target.checked; return; }
  const c = e.target.closest(".chip"); if (c) goToNeuron(+c.dataset.l, +c.dataset.n, "neuron");
});
async function runAblate() {
  const f = state.focus; if (!f || f.kind !== "neuron" || !X.cfg) return;
  if (X.job) { $("#status").textContent = "a run is already going"; return; }
  await lensReady(); if (X.job || !isFocus(f.layer, f.n, "neuron")) return;
  const x = xSettings();
  if (!x.measure) { $("#status").textContent = "pick a measure before ablating"; return; }
  const prompt = $("#c-input").value.trim() || $("#c-input").placeholder, method = $("#x-method").value;
  const neurons = X.ablateSet && pinned.length ? pinned : [{ layer: f.layer, n: f.n }];
  const t = { kind: "ablate", prompt, measure: x.measure, method, neurons, n: x.n, temperature: x.temperature, pick: 0, live: true,
              prediction: takePrediction() };
  state.thread.push(t); renderThread();
  let r;
  try { r = await api("/api/experiment/ablate", { layer: f.layer, n: f.n, neurons, prompt, measure: x.measure, method, samples: x.n,
                                                   temperature: x.temperature, seed: x.seed, max_new: x.max_new, prediction: t.prediction }); }
  catch (err) { r = { error: err.message }; }
  if (r.error) { t.error = r.error; t.live = false; renderTurn(t); return; }
  X.job = { id: r.id, t };
  try {
    await sse(await fetch(`/api/experiment/events?id=${r.id}`), ev => {
      if (ev.error) t.error = ev.error;
      if (ev.means) t.means = ev.means;
      if (ev.progress) { t.progress = ev.progress; t.means = null; }
      if (ev.summary) { t.summary = ev.summary; renderLesionCard(); }
      if (ev.end) Object.assign(t, { live: false, cancelled: ev.cancelled });
      const th = $("#c-thread"), near = th.scrollHeight - th.scrollTop - th.clientHeight < 80;
      renderTurn(t); if (near) th.scrollTop = th.scrollHeight;
    });
  } catch (err) { t.error = String(err); }
  finally { X.job = null; t.live = false; renderTurn(t); renderLesionCard(); }
}
$("#x-ablate").onclick = runAblate;
function ablateHTML(t, i) {
  const S = t.summary, m = t.measure, p = t.progress;
  const who = t.neurons.length === 1 ? `L${t.neurons[0].layer} N${t.neurons[0].n}` : `a set of ${t.neurons.length}: ${t.neurons.map(x => `L${x.layer} N${x.n}`).join(", ")}`;
  const head = `<div class="user sweep-head">${t.method} ablation of ${esc(who)} · ${t.n} replies each · temp ${t.temperature}<span class="recipe">“${esc(t.prompt)}” · ${esc(m.name)}${m.auto ? ` (${esc(m.words.join(", "))})` : ""}</span></div>`;
  const prog = t.live ? `<div class="xnote dim">${t.means ? `computing every neuron's mean activation over web text, once: ${Math.round(t.means.done / 1000)}k / ${Math.round(t.means.total / 1000)}k tokens`
      : p ? `${p.done}/${p.total} replies · about ${Math.round(p.eta)} s left` : "starting…"} · <button type="button" class="x-cancel">cancel</button></div>`
    : t.cancelled ? `<div class="xnote dim">cancelled</div>` : "";
  if (!S) return `<div class="turn ablate" data-i="${i}">${head}${prog}${t.error ? `<div class="err">${esc(t.error)}</div>` : ""}${predHTML(t)}</div>`;
  const picker = t.n > 1 ? `<div class="picker" role="group" aria-label="reply"><span class="xk" title="reply i used the same seed in both columns">reply</span>${Array.from({ length: t.n }, (_, s) =>
    `<button type="button" data-s="${s}" class="${s === t.pick ? "on" : ""}">${s + 1}</button>`).join("")}</div>` : "";
  const tips = ["the model as it is", "the neuron held at its " + (t.method === "mean" ? "mean activation" : "zero") + " on every token"];
  const cols = S.columns.map((C, r) => `<div class="col ${r ? "down" : "zero"}"><span class="lab" title="${tips[r]}">${C.label}</span>${colStats(C, m)}<span class="txt">${esc(C.texts[t.pick] || "")}</span>
    <span class="one">reply ${t.pick + 1}: ${num(C.scores[t.pick])}${C.coh_scores[t.pick] != null ? ` · coherence ${num(C.coh_scores[t.pick])}` : ""}</span></div>`).join("");
  return `<div class="turn ablate" data-i="${i}">${head}${picker}<div class="cols two">${cols}</div>
    <div class="xline">${esc(S.columns[1].vs0.sentence)}</div>
    ${S.warn ? `<div class="xnote warn">${S.n} replies per column (under ${X.cfg?.warn_n ?? 5}), so these intervals are rough.</div>` : ""}
    ${S.reproducible === false ? `<div class="xnote dim">Running on ${esc(S.device)}: the same seed may not reproduce this exact text.</div>` : ""}${predHTML(t)}</div>`;
}

async function runReads(f, text) {
  if (reading) return;
  reading = true; $("#c-go").textContent = "…";
  let r;
  try { r = await api("/api/reads", { layer: f.layer, n: f.n, text, chat: true }); }
  catch (err) { $("#c-reads").textContent = err.message; return; }
  finally { reading = false; $("#c-go").textContent = t("chat.run"); }
  if (!isFocus(f.layer, f.n, "neuron")) return;
  const m = Math.max(1e-6, ...r.units.map(u => Math.abs(u.act)));
  $("#c-reads").innerHTML = r.units.map(u => {
    const a = u.act / m, c = a >= 0 ? `rgba(255,181,71,${a * 0.85})` : `rgba(255,79,154,${-a * 0.85})`;
    return `<span style="background:${c}" title="${u.pieces.map(v => v.toFixed(3)).join(" · ")}">${esc(u.text)}</span>`;
  }).join("");
}

// ---------- status ----------
// keeps polling (slowly once ready), so a server restart or a one-off error message clears itself
async function pollStatus() {
  try {
    const s = await api("/api/status");
    state.statusText = s.status;
    $("#status").textContent = localizeStatus(s.status);
    if (s.status === "ready" && lensResults && !lensResults.ready) runLensSearch();
    setTimeout(pollStatus, s.status === "ready" ? 8000 : 2000);
  } catch {
    state.statusText = "server unreachable";
    $("#status").textContent = t("status.unreachable");
    setTimeout(pollStatus, 2000);
  }
}
pollStatus();

// server status strings are stable English identifiers; keep the API language-neutral and translate here
function localizeStatus(s) {
  if (!s || s === "ready") return "";
  if (s === "idle") return t("status.idle");
  if (s === "loading") return t("status.loading");
  if (s === "no model") return t("status.noModel");
  if (s === "server unreachable") return t("status.unreachable");
  let m = s.match(/^indexing lens (\d+)\/(\d+)$/);
  if (m) return t("status.indexing", { i: m[1], n: m[2] });
  m = s.match(/^measuring neuron ranges (\d+)k\/(\d+)k tokens$/);
  if (m) return t("status.measuring", { seen: m[1], tokens: m[2] });
  return s;
}
window.__ne = { map, state, scatter };
// ---------- start ----------
const have = Object.keys(index.layers).map(Number);
let saved = null; try { const v = localStorage.getItem("ne.layer"); if (v !== null) saved = +v; } catch {}
const first = have.includes(saved) ? saved : have.includes(22) ? 22 : (have[0] ?? 0);
loadExperiments(); loadTuning(); renderSet();
await setLayer(first);
if (location.hash) fromHash(); else restoreFocus();

// ---------- first-run tour: six steps pointing at the main controls; "seen" is kept in this browser ----------
const TOUR = [
  { sel: "#map", title: "The map", text: "Each sheet is one layer of the model and each dot is one neuron. Click a dot to open it. Brighter amber dots cluster on words more than random controls do; the legend at the bottom explains every color." },
  { sel: "#q-mode", title: "Search", text: "lens finds neurons whose lens has your word, so they push the model toward or away from it. fires runs your text through the model and lists the neurons that fire most on it." },
  { sel: "#jump", title: "Go to a neuron", text: "Type a layer and neuron, like L22 N2742, and press Enter. In the layer menu, ● marks the layers with sweep rollouts." },
  { sel: "#list", title: "The list", text: "The open neuron's lenses and connection web, then this layer's neurons ranked by word mass. On a search, the hits: + and − are both strong, the sign is the direction." },
  { sel: "#focus", title: "Chat with a steered model", text: "Open a neuron, then Chat. The three columns push the neuron up, leave it alone and push it down. Pick a measure and a number of samples to get means, 95% intervals and a one-line reading." },
  { sel: "#x-sweep", title: "Dose sweep", text: "Runs the prompt from −1 to +1 and plots the measure and coherence, then reports where the effect starts and where the text breaks. Hover any control to see what it does." },
];
let tourAt = -1;
function tourSeen() { try { localStorage.setItem("ne.tour", "seen"); } catch {} }
function endTour() { tourAt = -1; $("#tour").hidden = true; tourSeen(); }
function showTour(i) {
  tourAt = i;
  const st = TOUR[i], el = $(st.sel), box = $("#tour"), ring = $("#tour-ring"), card = $("#tour-card");
  box.hidden = false;
  const r = el && el.offsetParent !== null ? el.getBoundingClientRect() : null;
  const on = r && r.width > 0 && r.height > 0 && r.width < innerWidth * 0.8;  // a full-screen target (the map) gets a centered card
  ring.style.display = on ? "" : "none";
  if (on) Object.assign(ring.style, { left: `${r.left - 4}px`, top: `${r.top - 4}px`, width: `${r.width + 8}px`, height: `${r.height + 8}px` });
  card.innerHTML = `<div class="lab">${i + 1} / ${TOUR.length}</div><b>${esc(st.title)}</b><p>${esc(st.text)}</p>
    <div class="row">${i ? `<button type="button" data-t="back" class="dim">back</button>` : ""}<button type="button" data-t="skip" class="dim">skip</button><button type="button" data-t="next">${i === TOUR.length - 1 ? "done" : "next"}</button></div>`;
  // beside the target where it fits (below, above, then centered), kept on screen
  const cw = Math.min(340, innerWidth - 32), ch = card.offsetHeight || 160;
  let x = on ? r.left : (innerWidth - cw) / 2, y = on ? r.bottom + 12 : (innerHeight - ch) / 2;
  if (on && y + ch > innerHeight - 16) y = r.top - ch - 12;
  if (on && y < 16) { y = Math.max(16, (innerHeight - ch) / 2); x = r.right + 12 + cw > innerWidth ? r.left - cw - 12 : r.right + 12; }
  Object.assign(card.style, { width: `${cw}px`, left: `${Math.max(16, Math.min(x, innerWidth - cw - 16))}px`, top: `${Math.max(16, Math.min(y, innerHeight - ch - 16))}px` });
  card.querySelector('[data-t="next"]').focus();
}
$("#tour-card").addEventListener("click", e => {
  const t = e.target.closest("button")?.dataset.t; if (!t) return;
  if (t === "skip" || (t === "next" && tourAt === TOUR.length - 1)) return endTour();
  showTour(tourAt + (t === "next" ? 1 : -1));
});
addEventListener("keydown", e => { if (tourAt >= 0 && e.key === "Escape") endTour(); });
addEventListener("resize", () => { if (tourAt >= 0) showTour(tourAt); });
$("#tour-btn").onclick = () => showTour(0);
let seenTour = false; try { seenTour = localStorage.getItem("ne.tour") === "seen"; } catch {}
if (!seenTour) showTour(0);

// ---------- claude: the Claude tab, one claude -p session per conversation ----------
const agent = { chat: null, kind: "claude", busy: false, sending: 0, box: null, seg: null, segText: "", reply: "", waiters: [] };
const agentEl = $("#tab-claude");
function mdInline(t) {
  return esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\bL(\d{1,2})[ \/]N(\d{1,4})\b/g, (m, l, n) => `<a class="nref" data-l="${l}" data-n="${n}">${m}</a>`);
}
function md(src) {
  const out = []; let list = false;
  for (const line of src.split("\n")) {
    const h = line.match(/^#{1,4}\s+(.*)/), li = line.match(/^\s*[-*]\s+(.*)/);
    if (li) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${mdInline(li[1])}</li>`); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (h) out.push(`<h3>${mdInline(h[1])}</h3>`);
    else if (line.trim()) out.push(`<p>${mdInline(line)}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}
const EMPTY = () => `<div class="a-empty dim">${t("agent.empty")}</div>`;
// the conversation persists in this browser: chat id (= claude session, resumed server side) and the rendered thread
const AGENT_KEY = "explorer.agent";
function saveAgent() {
  try { localStorage.setItem(AGENT_KEY, JSON.stringify({ chat: agent.chat, kind: agent.kind, html: $("#a-thread").innerHTML })); } catch {}
}
// which agent the chat runs on: the tab is named after it, and the dropdown picks it for the next new chat
function setAgentKind(kind) {
  agent.kind = kind === "codex" ? "codex" : "claude";
  $("#a-agent").value = agent.kind;
  $('#tabs button[data-tab="claude"]').textContent = agent.kind === "codex" ? "Codex" : "Claude";
}
$("#a-agent").addEventListener("change", e => {
  const kind = e.target.value;
  try { localStorage.setItem("explorer.agentKind", kind); } catch {}
  if (agent.chat || $("#a-thread .a-msg")) { pastStash = null; $("#a-past").classList.remove("on"); resetAgent(); clearAgentMarks(); }
  setAgentKind(kind);
});
function loadAgent() {
  try {
    const v = JSON.parse(localStorage.getItem(AGENT_KEY) || "null"); if (!v) return;
    agent.chat = v.chat; $("#a-thread").innerHTML = v.html || EMPTY(); setAgentKind(v.kind);
    $("#a-thread").querySelectorAll("#a-usename, .a-msg.queued, .a-busy, .a-qctl").forEach(b => b.remove());
    $("#a-thread").scrollTop = $("#a-thread").scrollHeight;
    addTurnControls();
    relabelAgentThread();
  } catch {}
}
// quick actions above the input, for the open neuron
function renderAgentActions() {
  const f = state.focus;
  const ref = f && f.kind === "head" ? `L${f.layer} h${f.n}` : f ? `L${f.layer} N${f.n}` : "";
  $("#a-actions").innerHTML = f && f.kind === "neuron" ? `<button data-a="explain">${t("agent.explain", { ref })}</button><button data-a="web">${t("agent.web")}</button>`
    : f && f.kind === "head" ? `<button data-a="explain">${t("agent.explain", { ref })}</button>` : "";
}
$("#a-actions").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b || agent.busy) return;
  if (b.dataset.a === "explain") explainFocus();
  else if (b.dataset.a === "web") { const f = state.focus; agentTurn(`Look at the cone of L${f.layer} N${f.n} (what it reads from and writes into, a few hops out) and tell me briefly what that circuit seems to do. Highlight the neurons you mention.`, t("agent.webShown", { ref: `L${f.layer} N${f.n}` })); }
});
function resetAgent() {
  closeEvents(); if (agent.busy) finishTurn();
  agent.chat = null; $("#a-thread").innerHTML = EMPTY();
  let pref = "claude"; try { pref = localStorage.getItem("explorer.agentKind") || "claude"; } catch {}
  setAgentKind(pref);
  saveAgent();
}
agentEl.addEventListener("click", e => {
  const a = e.target.closest("a.nref"); if (a) goToNeuron(+a.dataset.l, +a.dataset.n);
});
// what Claude does to the screen (see agent_mcp.py)
// Claude's labels and highlights last until its next reply, or until the person opens a neuron themselves
function clearAgentMarks() {
  map.setNotes([]);
  if (agentMarked) { agentMarked = false; syncLeftMap(); }
}
async function agentUi(e) {
  if (e.action === "head") {
    agentSelecting = true;
    try { await openHead(e.layer, e.head); } finally { agentSelecting = false; }
  } else if (e.action === "select") {  // visibly go there
    agentSelecting = true;
    try { await openNeuron(e.layer, e.n, "neuron", true, false); } finally { agentSelecting = false; }
    map.flyToNeuron(e.layer, e.n, 0.3);
  }
  else if (e.action === "highlight") { agentMarked = true; map.setMarks((e.neurons || []).map(x => ({ layer: x.layer, n: x.n, sign: x.sign || "+", w: 1 }))); }
  else if (e.action === "rename") {
    const u = state.unitMap[e.layer]?.get(e.n); if (u) u.name = e.name.trim();
    if (isFocus(e.layer, e.n, "neuron")) setNameField(e.name.trim());
    if (e.layer === state.layer && state.units[e.layer]?.length) map.setUnits(e.layer, state.units[e.layer]);
    renderList();
  }
  else if (e.action === "annotate") map.setNotes(e.items || []);
  else if (e.action === "camera") {
    if (map.layerMode !== null) map.exitLayer();
    if (e.mode === "home") map.home();
    else if (e.mode === "layer") map.viewLayer(e.layer);
    else if (e.mode === "neuron") map.flyToNeuron(e.layer, e.n, e.closeness || 0.22);
  } else if (e.action === "tab") { if (state.focus && tabBtn(e.tab)) { if (curTab() === "claude") markFresh(e.tab); else setTab(e.tab); } }
  else if (e.action === "search") {
    if (state.qMode !== "lens") $('#q-mode button[data-q="lens"]').click();
    $("#lens-q").value = e.query; $("#lens-q").dispatchEvent(new Event("input"));
  } else if (e.action === "text") showText(e.text, e.token);
}
// put text in the fires box, run it, and pin the first token containing `token` (or the strongest one)
async function showText(text, token) {
  if (state.qMode !== "fires") $('#q-mode button[data-q="fires"]').click();
  $("#inf-text").value = text; $("#lens-q").value = text;
  clearTimeout(lensTimer);
  const g = ++qGen;
  await runInfer(text, g);
  if (g !== qGen || !state.inf.tokens) return;
  const i = token ? state.inf.tokens.findIndex(t => t.includes(token)) : -1;
  const pos = i >= 0 ? +document.querySelectorAll("#inf-toks button")[i].dataset.i : -1;
  state.inf.pos = pos;
  document.querySelectorAll("#inf-toks button").forEach(x => x.classList.toggle("on", +x.dataset.i === pos));
  inferView(); showTokenPaths(pos);
}
// a tool call as one short inline line: neuron L15 N6254 · search_lens "Sure"
function toolLabel(name, a = {}) {
  const ref = a.layer != null && a.n != null ? `L${a.layer} N${a.n}` : a.layer != null ? `L${a.layer}` : "";
  const short = s => (s.length > 90 ? s.slice(0, 90) + "…" : s);
  const extra = a.command ? short(a.command) : a.file_path ? a.file_path.replace(/^\/Users\/[^/]+/, "~")
    : a.pattern ? `"${a.pattern}"` : a.query ? `"${a.query}"` : a.neurons ? t("agent.neurons", { n: a.neurons.length }) : "";
  return [name.replace(/_/g, " "), ref, extra].filter(Boolean).join(" ");
}
function prettyOut(t) {
  try { return JSON.stringify(JSON.parse(t), null, 1); } catch { return t; }
}
function addMsg(cls, html) {
  const th = $("#a-thread"); th.querySelector(".a-empty")?.remove();
  const d = document.createElement("div"); d.className = `a-msg ${cls}`; d.innerHTML = html; th.append(d);
  th.scrollTop = th.scrollHeight; return d;
}
// one turn: the reply streams into a new assistant message, text and tool calls interleaved in the order they happen
// ---- the chat: one claude process per chat on the server and one event stream here. Messages go in any time:
// mid-reply they reach Claude between tool calls. Stop ends the reply only; background work keeps running and
// reports back in a turn of its own, which renders like any other. ----
function ensureChatId() { if (!agent.chat) agent.chat = crypto.randomUUID(); return agent.chat; }
let evCtl = null, evChat = null;
async function openEvents() {
  const chat = agent.chat; if (!chat || (evCtl && evChat === chat)) return;
  if (evCtl) evCtl.abort();
  const ctl = evCtl = new AbortController(); evChat = chat;
  while (!ctl.signal.aborted) {
    try {
      const r = await fetch(`/api/agent/events?chat=${encodeURIComponent(chat)}`, { signal: ctl.signal });
      const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done } = await rd.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 1);
          if (line && evCtl === ctl) onAgentEvent(JSON.parse(line));
        }
      }
    } catch { if (ctl.signal.aborted) return; }
    await new Promise(r => setTimeout(r, 1500));  // reconnect (server restarts); its state event settles a turn that ended meanwhile
  }
}
function closeEvents() { if (evCtl) evCtl.abort(); evCtl = evChat = null; }
// the reply box under the latest message; its "thinking…" line moves with it
function replyBox() {
  if (agent.box) return agent.box;
  $("#a-thread").querySelectorAll(".a-busy").forEach(x => x.remove());
  $("#a-thread").querySelectorAll(".a-msg.assistant:empty").forEach(x => x.remove());  // a reply that moved under a newer message
  const box = agent.box = addMsg("assistant", "");
  agent.seg = null;
  if (agent.busy) { const busy = document.createElement("div"); busy.className = "a-busy"; box.append(busy); }
  return box;
}
const putInBox = el => { const box = replyBox(), busy = box.querySelector(".a-busy"); busy ? box.insertBefore(el, busy) : box.append(el); };
function setBusy(b) { agent.busy = b; setSendMode(b); }
function onAgentEvent(e) {
  const th = $("#a-thread"), stick = th.scrollHeight - th.scrollTop - th.clientHeight < 80;
  if (e.t === "state") {
    // (re)connected: the turn in progress replays from its start, so what was drawn of it is drawn again
    if (agent.box && e.busy) { agent.box.innerHTML = ""; agent.seg = null; agent.reply = ""; }
    if (e.busy) { setBusy(true); replyBox(); if (!agent.box.querySelector(".a-busy")) { const d = document.createElement("div"); d.className = "a-busy"; agent.box.append(d); } }
    else if (agent.busy && !agent.sending) finishTurn();  // a message still on its way makes the server busy in a moment
    return;
  }
  if (e.t === "ping" || e.t === "sent") return;
  if (e.t === "ui") return agentUi(e);
  if (e.t === "done") return finishTurn();
  if (!agent.busy) { setBusy(true); clearTurnControls(); }  // a turn Claude started itself (background work finished)
  if (e.t === "text") {
    if (!agent.seg) { agent.seg = document.createElement("div"); agent.seg.className = "a-md"; agent.segText = ""; putInBox(agent.seg); }
    agent.segText += e.text; agent.reply += e.text; agent.seg.innerHTML = md(agent.segText);
  } else if (e.t === "tool") { agent.seg = null; putInBox(toolDetails(e)); }
  else if (e.t === "tool_result") { if (agent.box) fillTool(agent.box, e); }
  else if (e.t === "error") { const d = document.createElement("div"); d.className = "a-err"; d.textContent = e.error; putInBox(d); }
  if (stick) th.scrollTop = th.scrollHeight;
}
function finishTurn() {
  const box = agent.box, reply = agent.reply;
  $("#a-thread").querySelectorAll(".a-busy").forEach(x => x.remove());
  agent.box = agent.seg = null; agent.reply = "";
  setBusy(false); addTurnControls(); markFresh("claude"); saveAgent();
  for (const w of agent.waiters.splice(0)) w({ reply, box });
}
async function agentTurn(message, shown, shownCls = "user", atts = []) {
  const f = state.focus, th = $("#a-thread");
  if (pastStash !== null) { th.innerHTML = pastStash; pastStash = null; $("#a-past").classList.remove("on"); }
  clearAgentMarks(); clearTurnControls();
  const ub = addMsg(shownCls, esc(shown) + (atts.length ? `<span class="a-att">📎 ${atts.map(a => esc(a.name)).join(", ")}</span>` : ""));
  ub.dataset.msg = message; ub.dataset.shown = shown; ub.dataset.cls = shownCls; ub.dataset.att = JSON.stringify(atts);
  agent.box = agent.seg = null;  // the reply, or the rest of the one in progress, goes under this message
  ensureChatId(); openEvents();
  setBusy(true); replyBox(); th.scrollTop = th.scrollHeight; saveAgent();
  const out = new Promise(res => agent.waiters.push(res));
  agent.sending++;
  try {
    const r = await api("/api/agent/send", { chat: agent.chat, agent: agent.kind, message, attachments: atts.map(a => a.id),
      ...(f && f.kind === "neuron" ? { layer: f.layer, n: f.n } : f && f.kind === "head" ? { layer: f.layer, head: f.n } : {}) });
    if (r.error) throw new Error(r.error);
  } catch (err) {
    const d = document.createElement("div"); d.className = "a-err"; d.textContent = String(err.message || err); putInBox(d); finishTurn();
  } finally { agent.sending--; }
  return out;
}
async function stopAgent() {
  if (!agent.busy) return;
  const d = document.createElement("div"); d.className = "a-stopped"; d.textContent = t("agent.stopped"); putInBox(d);
  const r = agent.chat ? await api("/api/agent/interrupt", { chat: agent.chat }).catch(() => ({})) : {};
  if (!r.running && agent.busy) finishTurn();  // nothing running server side: settle here
}
function toolDetails(e) {
  const d = document.createElement("details"); d.className = "a-tool"; if (e.id) d.dataset.id = e.id;
  d.innerHTML = `<summary>${esc(toolLabel(e.name, e.input || {}))}</summary><pre class="a-io"><b>${esc(e.name)}</b> ${esc(JSON.stringify(e.input || {}, null, 1))}</pre><pre class="a-io out running">${t("agent.running")}</pre>`;
  return d;
}
function fillTool(box, e) {
  const d = [...box.querySelectorAll("details.a-tool")].find(x => x.dataset.id === e.id);
  if (d) { const o = d.querySelector(".out"); o.textContent = prettyOut(e.output); o.classList.remove("running"); o.classList.toggle("err", e.error); }
}
// ---- past chats: "past" lists the sessions on disk; picking one rebuilds its thread and later turns resume it ----
let pastStash = null;  // the thread shown before the list opened
const ago = ts => {
  const s = Date.now() / 1000 - ts;
  return s < 3600 ? t("agent.minutesAgo", { n: Math.max(1, Math.round(s / 60)) })
    : s < 86400 ? t("agent.hoursAgo", { n: Math.round(s / 3600) })
    : new Date(ts * 1000).toLocaleDateString(getLang());
};
async function renderPastList() {
  const th = $("#a-thread");
  th.innerHTML = `<div class="dim">…</div>`;
  const list = await api("/api/agent/chats");
  if (pastStash === null) return;
  th.innerHTML = list.length ? list.map(c => `<button class="a-pastrow${c.chat === agent.chat ? " on" : ""}" data-chat="${c.chat}" data-agent="${c.agent || "claude"}"><span class="t">${esc(c.title || "—")}</span><span class="m">${c.agent === "codex" ? "codex · " : ""}${ago(c.t)} · ${tp("agent.messages", c.turns)}</span></button>`).join("")
    : `<div class="dim">—</div>`;
}
async function togglePast() {
  const th = $("#a-thread");
  if (pastStash !== null) { th.innerHTML = pastStash; pastStash = null; $("#a-past").classList.remove("on"); addTurnControls(); return; }
  pastStash = th.innerHTML; $("#a-past").classList.add("on");
  await renderPastList();
}
async function openPast(chat, kind) {
  const r = await api(`/api/agent/history?chat=${encodeURIComponent(chat)}`);
  pastStash = null; $("#a-past").classList.remove("on");
  const th = $("#a-thread"); th.innerHTML = EMPTY();
  let box = null, seg = null, segText = "";
  for (const e of r.events || []) {
    if (e.t === "user") {
      const ub = addMsg("user", esc(e.text));
      Object.assign(ub.dataset, { msg: e.text, shown: e.text, cls: "user", att: "[]" });
      box = seg = null; continue;
    }
    if (!box) { box = addMsg("assistant", ""); seg = null; }
    if (e.t === "text") {
      if (!seg) { seg = document.createElement("div"); seg.className = "a-md"; box.append(seg); segText = ""; }
      segText += (segText ? "\n\n" : "") + e.text; seg.innerHTML = md(segText);
    } else if (e.t === "tool") { seg = null; box.append(toolDetails(e)); }
    else if (e.t === "tool_result") fillTool(box, e);
    else if (e.t === "error") { seg = null; const d = document.createElement("div"); d.className = "a-err"; d.textContent = e.error; box.append(d); }
  }
  closeEvents(); if (agent.busy) { agent.box = null; finishTurn(); }
  agent.chat = chat; setAgentKind(kind); clearAgentMarks(); addTurnControls(); saveAgent(); openEvents();
  th.scrollTop = th.scrollHeight;
}
$("#a-thread").addEventListener("click", e => { const b = e.target.closest(".a-pastrow"); if (b) openPast(b.dataset.chat, b.dataset.agent); });
async function explainFocus() {
  const f = state.focus; if (!f) return;
  if (f.kind === "head") return agentTurn(`Explain attention head L${f.layer} h${f.n}: write the head report.`, t("agent.explain", { ref: `L${f.layer} h${f.n}` }));
  if (f.kind !== "neuron") return;
  const { reply, box } = await agentTurn(`Explain L${f.layer} N${f.n}: write the autointerp report.`, t("agent.explain", { ref: `L${f.layer} N${f.n}` }));
  const name = reply.match(/\*\*Name\*\*:?\s*(.+)/)?.[1].replace(/[*`]/g, "").trim();
  if (name) {
    const b = document.createElement("button"); b.id = "a-usename"; b.dataset.name = name; b.textContent = t("agent.useName", { name });
    b.onclick = () => { if (isFocus(f.layer, f.n, "neuron")) setNameField(name); saveName({ layer: f.layer, n: f.n, name }); b.remove(); };
    box.append(b);
  }
}
function sendChat() {
  const msg = $("#a-input").value.trim();
  if (agent.pending.some(a => !a.id) || (!msg && !agent.pending.length)) return false;
  const atts = agent.pending; agent.pending = []; renderPending();
  $("#a-input").value = ""; growInput();
  agentTurn(msg || "(see the attached files)", msg || t("agent.attached"), "user", atts);
  return true;
}
function setSendMode(busy) {
  const b = $("#a-send"); b.disabled = false; b.classList.toggle("stop", busy);
  b.textContent = busy ? "■" : "↑"; const label = busy ? t("agent.stop") : t("agent.send"); b.title = label; b.setAttribute("aria-label", label);
}

// ---- attachments: picked, dropped or pasted files upload right away and go with the next message ----
agent.pending = [];
function renderPending() {
  $("#a-files").innerHTML = agent.pending.map((a, i) => `<span class="a-file${a.id ? "" : " up"}">${esc(a.name)}<button data-i="${i}" aria-label="${esc(t("agent.remove"))}">×</button></span>`).join("");
}
$("#a-files").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { agent.pending.splice(+b.dataset.i, 1); renderPending(); } });
async function attachFiles(files) {
  for (const file of files) {
    const a = { name: file.name || t("agent.pasted"), id: null }; agent.pending.push(a); renderPending();
    const data = await new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1] || ""); r.onerror = bad; r.readAsDataURL(file); });
    try {
      const res = await api("/api/agent/upload", { name: a.name, type: file.type, data });
      if (res.error) throw new Error(res.error);
      a.id = res.id;
    } catch (err) { agent.pending.splice(agent.pending.indexOf(a), 1); addMsg("assistant", `<div class="a-err">${esc(a.name)}: ${esc(String(err.message || err))}</div>`); }
    renderPending();
  }
}
$("#a-attach").onclick = () => $("#a-file").click();
$("#a-file").addEventListener("change", e => { attachFiles([...e.target.files]); e.target.value = ""; });
agentEl.addEventListener("dragover", e => { if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); agentEl.classList.add("drop"); } });
agentEl.addEventListener("dragleave", e => { if (!agentEl.contains(e.relatedTarget)) agentEl.classList.remove("drop"); });
agentEl.addEventListener("drop", e => { if (!e.dataTransfer?.files?.length) return; e.preventDefault(); agentEl.classList.remove("drop"); attachFiles([...e.dataTransfer.files]); });
$("#a-input").addEventListener("paste", e => { const fs = [...(e.clipboardData?.files || [])]; if (fs.length) { e.preventDefault(); attachFiles(fs); } });

// ---- edit / reroll the latest turn: cut the claude session back to before it, then resend or hand it back ----
function clearTurnControls() { document.querySelectorAll("#a-thread .a-ctl").forEach(x => x.remove()); }
function lastTurn() {
  const users = [...document.querySelectorAll("#a-thread .a-msg.user")], u = users[users.length - 1];
  if (!u) return null;
  const after = []; for (let x = u.nextElementSibling; x; x = x.nextElementSibling) after.push(x);
  return { u, after };
}
function addTurnControls() {
  clearTurnControls();
  const turn = lastTurn(); if (!turn) return;
  const uc = document.createElement("div"); uc.className = "a-ctl"; uc.innerHTML = `<button data-act="edit" title="${esc(t("agent.editTitle"))}">${t("agent.edit")}</button>`;
  turn.u.append(uc);
  const a = turn.after.find(x => x.classList.contains("assistant"));
  if (a) { const ac = document.createElement("div"); ac.className = "a-ctl"; ac.innerHTML = `<button data-act="reroll" title="${esc(t("agent.rerollTitle"))}">${t("agent.reroll")}</button>`; a.append(ac); }
}
async function rewindLast(act) {
  const turn = lastTurn(); if (!turn) return;
  if (agent.busy) { agent.box = null; finishTurn(); }
  const u = turn.u, msg = u.dataset.msg ?? u.childNodes[0]?.textContent ?? "", shown = u.dataset.shown ?? msg;
  const cls = u.dataset.cls || "user", atts = JSON.parse(u.dataset.att || "[]");
  if (agent.chat) { try { await api("/api/agent/rewind", { chat: agent.chat, text: msg }); } catch {} }
  for (const x of [u, ...turn.after]) x.remove();
  if (!$("#a-thread").children.length) $("#a-thread").innerHTML = EMPTY();
  if (act === "reroll") agentTurn(msg, shown, cls, atts);
  else { $("#a-input").value = cls.includes("interp") || shown !== msg ? shown : msg; growInput(); agent.pending = atts; renderPending(); $("#a-input").focus(); addTurnControls(); saveAgent(); }
}
$("#a-thread").addEventListener("click", e => { const b = e.target.closest(".a-ctl button"); if (b) rewindLast(b.dataset.act); });
function growInput() { const t = $("#a-input"); t.style.height = "auto"; t.style.height = `${Math.min(200, t.scrollHeight)}px`; }
$("#a-input").addEventListener("input", growInput);
// Enter sends, also while Claude works: the message reaches it between tool calls
$("#a-input").addEventListener("keydown", e => {
  if (e.key !== "Enter" || e.shiftKey) return;
  e.preventDefault();
  sendChat();
});
// the button sends, or stops the reply while Claude works (Enter always sends: mid-reply the message goes straight in)
$("#a-form").addEventListener("submit", e => { e.preventDefault(); if (agent.busy) stopAgent(); else sendChat(); });
$("#a-new").onclick = () => { pastStash = null; $("#a-past").classList.remove("on"); resetAgent(); clearAgentMarks(); };
$("#a-past").onclick = togglePast;
{ let pref = "claude"; try { pref = localStorage.getItem("explorer.agentKind") || "claude"; } catch {} setAgentKind(pref); }
loadAgent(); renderAgentActions(); openEvents();
syncEdge(); updateInsets();

// ---------- language ----------
// a switch redraws everything derived from a translated string; open neurons, searches and conversations stay put
function relabelAgentThread() {
  const th = $("#a-thread");
  th.querySelectorAll(".a-empty").forEach(el => { el.innerHTML = t("agent.empty"); });
  th.querySelectorAll(".a-stopped").forEach(el => { el.textContent = t("agent.stopped"); });
  th.querySelectorAll("details.a-tool .a-io.out.running").forEach(el => { el.textContent = t("agent.running"); });
  th.querySelectorAll(".a-ctl button[data-act=edit]").forEach(b => { b.textContent = t("agent.edit"); b.title = t("agent.editTitle"); });
  th.querySelectorAll(".a-ctl button[data-act=reroll]").forEach(b => { b.textContent = t("agent.reroll"); b.title = t("agent.rerollTitle"); });
  const un = $("#a-usename");
  if (un) un.textContent = t("agent.useName", { name: un.dataset.name || un.textContent.replace(/^use name:\s*/, "").replace(/^使用名称：/, "") });
}
function relabel() {
  hideTip();
  applyTranslations();
  syncPlaceholders();
  syncEdge();
  setAgentKind(agent.kind);
  setSendMode(agent.busy);
  $("#c-go").textContent = running ? t("chat.stop") : t("chat.run");
  if (map.layerMode !== null) $("#layer-back").textContent = t("map.allLayers", { l: map.layerMode });
  if (state.webTitle) $("#web-z").title = t("web.title", state.webTitle);
  if (state.statusText) $("#status").textContent = localizeStatus(state.statusText);
  const f = state.focus;
  if (f?.kind === "random") $("#f-id").textContent = t("focus.idRandom", { layer: f.layer, n: f.n });
  if ((f?.kind === "neuron" || f?.kind === "random") && state.unit) {
    if (state.unit.stats) $("#f-rank").textContent = t("focus.rank", { rank: state.unit.rank ?? "—", n: state.unit.n_ranked, mass: (state.unit.stats.mass ?? state.unit.stats.z).toFixed(1), nullPct: (state.unit.stats.null_pct * 100).toFixed(0) });
    renderExamples(state.unit);
    renderDetails(state.unit);
    renderLensDetails();
    renderConnections();
  }
  renderList();
  if (fireResults) renderResults(fireResults, $("#run-rows"));
  renderAgentActions();
  if (!running) renderThread();
  relabelAgentThread();
  if (agent.pending.length) renderPending();
  if (pastStash !== null) renderPastList();
}
document.addEventListener("i18nchange", relabel);
syncPlaceholders();
