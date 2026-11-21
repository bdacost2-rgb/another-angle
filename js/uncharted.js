// Uncharted on the website: everything the Colab version did in Python, done in the browser.
//  - The map: every photo and passage, laid out by meaning (t-SNE, made by tools/build_data.py).
//  - Seeds: your Another Angle picks and hearts warm the map, so it starts from your real taste.
//  - Checkpoints show nearby archive works directly and leave their connection to the viewer.
//  - On/off track votes train the same taste model as the feed, so both get smarter together.
import { loadArchive, normalize, dot } from "./data.js";
import { TasteModel } from "./taste.js";
import { mergeInto } from "./mine.js";
import { initGallery, loadDials } from "./uncharted_gallery.js";
import { getFocus, wireFocusButtons } from "./focus.js";

function pcaLayout(items) {
  // Fallback when the data has no t-SNE layout: the two main directions of variation.
  const D = items[0].vec.length;
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let v = new Float32Array(D).map(() => Math.random() - 0.5);
    for (let it = 0; it < 30; it++) {
      const nv = new Float32Array(D);
      for (const x of items) {
        let p = dot(x.vec, v);
        for (const u of comps) p -= dot(x.vec, u) * dot(u, v);
        for (let d = 0; d < D; d++) nv[d] += p * x.vec[d];
      }
      for (const u of comps) { const k = dot(nv, u); for (let d = 0; d < D; d++) nv[d] -= k * u[d]; }
      v = normalize(nv);
    }
    comps.push(v);
  }
  const xy = items.map((x) => comps.map((u) => dot(x.vec, u)));
  for (const k of [0, 1]) {
    const vals = xy.map((p) => p[k]), lo = Math.min(...vals), hi = Math.max(...vals);
    xy.forEach((p) => (p[k] = 0.04 + 0.92 * (p[k] - lo) / (hi - lo + 1e-9)));
  }
  items.forEach((it, i) => { it.x = xy[i][0]; it.y = xy[i][1]; });
}

function placeThemes(themes, items) {
  for (const t of themes) {
    if (t.x != null && t.y != null) continue;
    const near = items.map((it) => [dot(it.vec, t.vec), it]).sort((a, b) => b[0] - a[0]).slice(0, 20).map(([, it]) => it);
    const med = (a) => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
    t.x = med(near.map((i) => i.x)); t.y = med(near.map((i) => i.y));
  }
  // Nudge apart themes that land on top of each other, so labels stay readable.
  for (let r = 0; r < 60; r++) {
    for (const a of themes) for (const b of themes) {
      if (a === b) continue;
      const dx = a.x - b.x, dy = a.y - b.y, n = Math.hypot(dx, dy);
      if (n < 0.075) { a.x += (dx / (n + 1e-6)) * (0.075 - n) * 0.5; a.y += (dy / (n + 1e-6)) * (0.075 - n) * 0.5; }
    }
    for (const t of themes) { t.x = Math.min(0.95, Math.max(0.05, t.x)); t.y = Math.min(0.95, Math.max(0.05, t.y)); }
  }
}

// Map distances use the map's aspect ratio (64 x 40 cells), like the rest of the map code.
const mapDist2 = (ax, ay, bx, by) => (ax - bx) ** 2 + ((ay - by) * 0.625) ** 2;

function readLand(items, dials) {
  const K = dials.length;
  const scored = items.filter((it) => Number.isFinite(it.x) && Number.isFinite(it.y));
  const Z = scored.map((it) => {
    const z = new Float32Array(K);
    dials.forEach((dial, k) => {
      const [mean, std] = dial.stats[it.kind] || dial.stats.photo;
      z[k] = (dot(it.vec, dial.vec) - mean) / (std || 1);
    });
    return z;
  });
  // A Gaussian-weighted average of the works around (x, y): the local character of the land.
  function profile(x, y, sigma) {
    const p = new Float32Array(K);
    let weight = 0;
    const s2 = 2 * sigma * sigma;
    scored.forEach((it, i) => {
      const w = Math.exp(-mapDist2(it.x, it.y, x, y) / s2);
      if (w < 1e-4) return;
      weight += w;
      for (let k = 0; k < K; k++) p[k] += w * Z[i][k];
    });
    if (weight) for (let k = 0; k < K; k++) p[k] /= weight;
    return { p, weight };
  }
  const word = (k, v) => (v >= 0 ? dials[k].high : dials[k].low);
  const strongest = (p, n, floor) => [...p].map((v, k) => ({ k, v })).filter(({ v }) => Math.abs(v) > floor)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, n);

  // Named areas: the most distinctive spots, each named after a dial pole no other area uses.
  const regions = [];
  if (K) {
    const candidates = [];
    for (let gy = 0; gy < 11; gy++) for (let gx = 0; gx < 18; gx++) {
      const x = 0.06 + (gx / 17) * 0.88, y = 0.06 + (gy / 10) * 0.88;
      const { p, weight } = profile(x, y, 0.07);
      if (weight < 3) continue;
      p.forEach((v, k) => candidates.push({ x, y, k, v, p }));
    }
    candidates.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const usedPoles = new Set();
    for (const c of candidates) {
      if (regions.length >= 9 || Math.abs(c.v) < 0.35) break;
      const pole = word(c.k, c.v);
      if (usedPoles.has(pole) || regions.some((r) => mapDist2(r.x, r.y, c.x, c.y) < 0.17 ** 2)) continue;
      usedPoles.add(pole);
      // a second word only if no other area uses it, so every word on the map appears once
      const second = strongest(c.p, 3, 0.35).find(({ k, v }) => k !== c.k && !usedPoles.has(word(k, v)));
      if (second) usedPoles.add(word(second.k, second.v));
      regions.push({ x: c.x, y: c.y, text: second ? `${pole} · ${word(second.k, second.v)}` : pole });
    }
  }

  function describe(x, y) {
    const { p, weight } = profile(x, y, 0.045);
    const near = scored.map((it) => [mapDist2(it.x, it.y, x, y), it]).sort((a, b) => a[0] - b[0]).slice(0, 3)
      .map(([, it]) => ({ kind: it.mine ? "your photo" : it.kind === "photo" ? "photo" : "passage", title: it.title || "Untitled", img: it.kind === "photo" ? it.img : null, creator: it.creator || "" }));
    return {
      empty: weight < 0.6,
      words: weight < 0.6 ? [] : strongest(p, 3, 0.25).map(({ k, v }) => word(k, v)),
      works: near,
    };
  }
  return { regions, describe };
}

async function main() {
  wireFocusButtons();
  let A;
  try { A = await loadArchive(); } catch (e) { console.error(e); A = null; }
  if (A && A.themes.length) {
    const mine = await mergeInto(A); // your portfolio joins the map
    const T = new TasteModel(A);
    const items = A.items;
    if (items[0].x == null) pcaLayout(items);
    else for (const m of mine) { // place each of your photos among the archive works most like it
      const near = items.filter((it) => !it.mine).map((it) => [dot(it.vec, m.vec), it]).sort((a, b) => b[0] - a[0]).slice(0, 5);
      const w = near.reduce((s, [d]) => s + Math.max(d, 0.01), 0);
      m.x = near.reduce((s, [d, it]) => s + Math.max(d, 0.01) * it.x, 0) / w;
      m.y = near.reduce((s, [d, it]) => s + Math.max(d, 0.01) * it.y, 0) / w;
    }
    const themes = A.themes.map((t) => ({ ...t, vec: t.vec ? normalize(Float32Array.from(t.vec)) : null })).filter((t) => t.vec);
    placeThemes(themes, items);

    // Your Another Angle choices become evidence on the map: picks and hearts warm it, "less like this" cools it.
    const train = [];
    for (const [id, l] of T.labels) {
      const it = T.byId.get(id);
      if (!it) continue;
      train.push({ id, x: it.x, y: it.y, yv: l.y, w: l.y ? 0.35 * l.w : 0.15 * l.w });
    }
    // Reading the land: every work scored on the gallery's twelve CLIP design dials (in standard
    // deviations of the archive), so any spot on the map can be described by the works around it.
    const dials = await loadDials();
    const land = readLand(items, dials);

    window.COMPASS_DATA = {
      themes, train, trained: T.count, journey: null, regions: land.regions,
      works: items.map((it) => ({ x: it.x, y: it.y, kind: it.kind === "photo" ? 0 : 1, id: it.index, mine: !!it.mine, title: (it.title || "").slice(0, 60) })),
      mine: mine.length,
    };

    window.UNCHARTED_BRIDGE = {
      // What a spot on the map is like: its strongest dial words and the works nearest to it.
      "compass.describe": (x, y) => land.describe(x, y),
      // The real works closest in meaning to a checkpoint, excluding works already shown.
      "compass.peek": (ids, seenIds = []) => {
        const seen = new Set(Array.isArray(seenIds) ? seenIds : []);
        const pts = ids.map(([, i]) => items[i]).filter(Boolean);
        const cx = pts.reduce((s, p) => s + p.x, 0) / (pts.length || 1), cy = pts.reduce((s, p) => s + p.y, 0) / (pts.length || 1);
        const tw = themes.map((t) => Math.exp(-((cx - t.x) ** 2 + (cy - t.y) ** 2) / (2 * 0.12 ** 2)));
        const q = new Float32Array(A.dim);
        themes.forEach((t, k) => { for (let d = 0; d < A.dim; d++) q[d] += tw[k] * t.vec[d]; });
        normalize(q);
        // Photos, writing, or both (the "Show me" switch): three works either way.
        const focus = getFocus();
        const nearest = (kind, n) => items.filter((it) => it.kind === kind && !it.mine && !seen.has(it.index))
          .map((it) => [dot(it.vec, q), it]).sort((a, b) => b[0] - a[0]).slice(0, n).map(([, it]) => it);
        let passages = [];
        if (focus === "text") passages = nearest("text", 3);
        else if (focus === "both") {
          const nearbyPassage = ids.filter(([kind, id]) => kind === 1 && !seen.has(id)).map(([, id]) => items[id]).find(Boolean);
          const passage = nearbyPassage || nearest("text", 1)[0];
          if (passage) { passages = [passage]; for (let d = 0; d < A.dim; d++) q[d] += 0.7 * passage.vec[d]; normalize(q); }
        }
        const photos = focus === "text" ? [] : nearest("photo", focus === "photo" ? 3 : 2);
        const credit = (it) => [it.creator, it.date, it.source].filter(Boolean).join(" · ");
        return {
          photos: photos.map((it) => ({ src: it.img, title: it.title || "Untitled photograph", credit: credit(it), id: it.index })),
          passages: passages.map((it) => ({ text: it.text.split(/\s+/).slice(0, 55).join(" ") + "…", title: it.title || "Untitled passage", credit: credit(it), id: it.index })),
        };
      },
      // On/off track trains the same taste model as the feed.
      "compass.vote": (kind, i, good) => {
        const it = items[i];
        if (it) T.label(it, good ? 1 : 0);
        // the gallery listens, and regroups its rooms the next time you walk back into it
        window.dispatchEvent(new CustomEvent("taste:changed", { detail: { good, source: "checkpoint" } }));
        return { ok: true };
      },
    };
    initGallery(items, T, A.themes);
  }
  if (!(A && A.themes.length)) initGallery([], null);
  const s = document.createElement("script");
  s.src = "js/uncharted_map.js";
  document.body.append(s);
}

main();
