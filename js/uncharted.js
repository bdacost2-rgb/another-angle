// Uncharted on the website: everything the Colab version did in Python, done in the browser.
//  - The map: every photo and passage, laid out by meaning (t-SNE, made by tools/build_data.py).
//  - Seeds: your Another Angle picks and hearts warm the map, so it starts from your real taste.
//  - Checkpoints show the real archive works closest in MEANING to that spot (CLIP), and
//    a blend of two of them stands in for "what could exist here" (no paid image generation).
//  - On/off track votes train the same taste model as the feed, so both get smarter together.
//  - "I'm stuck here" uses the teammate's coach: their diagnosis and exercises, ranked by what
//    has helped you before, and saved to the same history as the full coach.
import { loadArchive, normalize, dot } from "./data.js";
import { TasteModel } from "./taste.js";
import { EXERCISES, diagnose } from "./coach_data.js";
import { mergeInto } from "./mine.js";

const COACH_STORE = "anotherangle.coach.v1";
const BROAD = { motivation: "Motivation & Energy", inspiration: "Idea Generation & Starting", confidence: "Perfectionism & Self-Criticism" };

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

function coachData() {
  try { return JSON.parse(localStorage.getItem(COACH_STORE)) || { profile: {}, history: [], learned_examples: [] }; }
  catch (e) { return { profile: {}, history: [], learned_examples: [] }; }
}

async function main() {
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
      train.push({ x: it.x, y: it.y, yv: l.y, w: l.y ? 0.35 * l.w : 0.15 * l.w });
    }
    window.COMPASS_DATA = {
      themes, train, trained: T.count, journey: null,
      works: items.map((it) => ({ x: it.x, y: it.y, kind: it.kind === "photo" ? 0 : 1, id: it.index, mine: !!it.mine })),
      mine: mine.length,
    };

    const shownPhotos = [], offTrack = new Set();
    let last = { text: "", category: null };
    window.UNCHARTED_BRIDGE = {
      // The real works closest in meaning to a checkpoint: the theme blend there + the nearest passage.
      "compass.peek": (ids) => {
        const pts = ids.map(([, i]) => items[i]).filter(Boolean);
        const cx = pts.reduce((s, p) => s + p.x, 0) / (pts.length || 1), cy = pts.reduce((s, p) => s + p.y, 0) / (pts.length || 1);
        const tw = themes.map((t) => Math.exp(-((cx - t.x) ** 2 + (cy - t.y) ** 2) / (2 * 0.12 ** 2)));
        const q = new Float32Array(A.dim);
        themes.forEach((t, k) => { for (let d = 0; d < A.dim; d++) q[d] += tw[k] * t.vec[d]; });
        normalize(q);
        const passages = ids.filter(([k]) => k === 1).map(([, i]) => items[i]).filter(Boolean);
        if (passages.length) { for (let d = 0; d < A.dim; d++) q[d] += 0.7 * passages[0].vec[d]; normalize(q); }
        const photos = items.filter((it) => it.kind === "photo" && !shownPhotos.includes(it.index) && !offTrack.has(it.index))
          .map((it) => [dot(it.vec, q), it]).sort((a, b) => b[0] - a[0]).slice(0, 2).map(([, it]) => it);
        shownPhotos.push(...photos.map((p) => p.index));
        shownPhotos.splice(0, Math.max(0, shownPhotos.length - 30));
        const credit = (it) => [it.creator, it.title, it.date].filter(Boolean).join(", ") + " - " + it.source;
        return {
          photos: photos.map((it) => ({ src: it.img, credit: credit(it), id: it.index })),
          passages: passages.map((it) => ({ text: it.text.split(/\s+/).slice(0, 55).join(" ") + "…", credit: credit(it), id: it.index })),
        };
      },
      // On/off track trains the same taste model as the feed.
      "compass.vote": (kind, i, good) => {
        const it = items[i];
        if (it) T.label(it, good ? 1 : 0);
        if (!good && kind === 0) offTrack.add(i);
        return { ok: true };
      },
      // "I'm stuck here": the teammate's diagnosis and exercises, ranked by what has helped you.
      "compass.coach": (block, text, title, setting) => {
        let category = BROAD[block] || "Idea Generation & Starting";
        if (text) { const [[top, score]] = diagnose(text); if (score > 0) category = top; }
        const data = coachData();
        const score = (e) => {
          let a = 1, b = 1;
          for (const r of data.history) if (r.category === category && r.exercise === e) { const v = { Yes: 1, Partly: 0.5, No: 0 }[r.helped] ?? 0.5; a += v; b += 1 - v; }
          return a / (a + b) + Math.random() * 0.15;
        };
        const exercise = [...EXERCISES[category]].sort((x, y) => score(y) - score(x))[0];
        last = { text: text || block, category };
        const reply = `This sounds like ${category.toLowerCase()}. Stay at "${title}"${setting ? ` (${setting})` : ""} and try this, in words or in frames: ${exercise}`;
        return { text: reply, block: category.split(" & ")[0].split(" ")[0], intervention: exercise, label: category };
      },
      "compass.coach_feedback": (block, exercise, helped) => {
        if (!last.category) return { ok: false };
        const data = coachData();
        data.history.push({ timestamp: new Date().toISOString().slice(0, 16).replace("T", " "), problem: last.text, category: last.category,
          exercise, helped: helped ? "Yes" : "No", progress: "", source: "Uncharted" });
        try { localStorage.setItem(COACH_STORE, JSON.stringify(data)); } catch (e) { /* ignore */ }
        return { ok: true };
      },
    };
  }
  const s = document.createElement("script");
  s.src = "js/uncharted_map.js";
  document.body.append(s);
}

main();
