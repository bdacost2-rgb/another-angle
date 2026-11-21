// The taste model: Bayesian logistic regression on CLIP embeddings.
// Every pick, heart, or "less like this" is a training example. The model
// keeps track of how sure it is (a diagonal Laplace approximation), and the
// feed uses Thompson sampling: it draws a plausible version of your taste
// each time, so it keeps exploring where it is unsure instead of showing
// you the same thing forever.
import { dot, normalize } from "./data.js";

const STORE = "anotherangle.taste.v1";

function gauss() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export class TasteModel {
  constructor(archive) {
    this.a = archive;
    this.dim = archive.dim;
    this.byId = new Map(archive.items.map((it) => [it.id, it]));
    this.labels = new Map(); // id -> {y: 0|1, w: weight}
    this.w = new Float32Array(this.dim);
    this.b = 0;
    this.var = new Float32Array(this.dim).fill(1);
    this.lambda = 1.0;
    this.load();
  }

  get count() { return [...this.labels.values()].filter((l) => l.w >= 1).length; }
  get liked() { return [...this.labels].filter(([, l]) => l.y === 1 && l.w >= 1).map(([id]) => this.byId.get(id)).filter(Boolean); }

  label(item, y, w = 1) {
    this.labels.set(item.id, { y, w });
    this.fit();
    this.save();
  }
  unlabel(item) { this.labels.delete(item.id); this.fit(); this.save(); }
  reset() { this.labels.clear(); this.fit(); this.save(); }

  fit() {
    const data = [...this.labels].map(([id, l]) => [this.byId.get(id), l]).filter(([it]) => it);
    const D = this.dim;
    this.w = new Float32Array(D);
    this.b = 0;
    if (!data.length) { this.var.fill(1 / this.lambda); return; }
    const scale = 8; // CLIP vectors are unit length, so features are scaled up to learn faster
    const lr = 0.5;
    for (let step = 0; step < 150; step++) {
      const g = new Float32Array(D);
      let gb = 0, W = 0;
      for (const [it, l] of data) {
        const p = sigmoid(scale * dot(this.w, it.vec) + this.b);
        const e = (p - l.y) * l.w;
        for (let d = 0; d < D; d++) g[d] += e * scale * it.vec[d];
        gb += e;
        W += l.w;
      }
      for (let d = 0; d < D; d++) this.w[d] -= lr * (g[d] / W + (this.lambda / W) * this.w[d]);
      this.b -= lr * gb / W;
    }
    // Uncertainty: inverse of the diagonal of the Hessian.
    const h = new Float32Array(D).fill(this.lambda);
    for (const [it, l] of data) {
      const p = sigmoid(scale * dot(this.w, it.vec) + this.b);
      const r = p * (1 - p) * l.w * scale * scale;
      for (let d = 0; d < D; d++) h[d] += r * it.vec[d] * it.vec[d];
    }
    for (let d = 0; d < D; d++) this.var[d] = 1 / h[d];
    this.scale = scale;
  }

  score(item, w = this.w) { return (this.scale || 8) * dot(w, item.vec) + this.b; }
  prob(item) { return sigmoid(this.score(item)); }

  uncertainty(item) {
    let s = 0;
    for (let d = 0; d < this.dim; d++) s += this.var[d] * item.vec[d] * item.vec[d];
    return Math.sqrt(s) * (this.scale || 8);
  }

  sampleWeights() {
    const w = new Float32Array(this.dim);
    for (let d = 0; d < this.dim; d++) w[d] = this.w[d] + Math.sqrt(this.var[d]) * gauss() * 0.5;
    return w;
  }

  // Pick n items: Thompson sample + diversity (no near-duplicates in a batch).
  // anchor: optional centered vector to stay close to ("more like this" or a search).
  // bonus: optional (item) -> number added to each score (the feed's design-dial steering).
  pick(n, { kind = null, exclude = new Set(), anchor = null, anchorWeight = 0.75, bonus = null } = {}) {
    const w = this.sampleWeights();
    const pool = this.a.items.filter((it) => (!kind || it.kind === kind) && !exclude.has(it.id) && !this.labels.has(it.id));
    const scored = pool.map((it) => {
      const taste = this.count ? Math.tanh(this.score(it, w) / 4) : (Math.random() - 0.5) * 0.4;
      const near = anchor ? dot(anchor, it.vec) : 0;
      const s = anchor ? anchorWeight * near * 3 + (1 - anchorWeight) * taste : taste;
      return { it, s: bonus ? s + bonus(it) : s };
    }).sort((x, y) => y.s - x.s).slice(0, n * 8);
    const chosen = [];
    while (chosen.length < n && scored.length) {
      let best = 0, bestVal = -Infinity;
      for (let i = 0; i < scored.length; i++) {
        const sim = chosen.length ? Math.max(...chosen.map((c) => dot(c.vec, scored[i].it.vec))) : 0;
        const val = scored[i].s - 0.6 * Math.max(0, sim);
        if (val > bestVal) { bestVal = val; best = i; }
      }
      chosen.push(scored.splice(best, 1)[0].it);
    }
    return chosen;
  }

  // The passage (or photo) that best matches an item's meaning, nudged toward your taste.
  partnerFor(item, exclude = new Set()) {
    const kind = item.kind === "photo" ? "text" : "photo";
    let best = null, bestVal = -Infinity;
    for (const it of this.a.items) {
      if (it.kind !== kind || exclude.has(it.id)) continue;
      const val = dot(item.vec, it.vec) + (this.count ? 0.08 * Math.tanh(this.score(it) / 4) : 0);
      if (val > bestVal) { bestVal = val; best = it; }
    }
    return best;
  }

  // Why the model showed something: the liked item it is closest to.
  because(item) {
    let best = null, bestSim = 0.15;
    for (const l of this.liked) {
      if (l.id === item.id) continue;
      const s = dot(l.vec, item.vec);
      if (s > bestSim) { bestSim = s; best = l; }
    }
    return best;
  }

  nearest(vec, n, kind = null, exclude = new Set()) {
    return this.a.items
      .filter((it) => (!kind || it.kind === kind) && !exclude.has(it.id))
      .map((it) => ({ it, s: dot(vec, it.vec) }))
      .sort((x, y) => y.s - x.s).slice(0, n).map((x) => x.it);
  }

  // Your taste as one direction in the map, for "your taste in words" etc.
  direction() { return normalize(Float32Array.from(this.w)); }

  save() {
    try { localStorage.setItem(STORE, JSON.stringify([...this.labels])); } catch (e) { /* private mode */ }
  }
  load() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORE) || "[]");
      for (const [id, l] of raw) if (this.byId.has(id)) this.labels.set(id, l);
    } catch (e) { /* ignore */ }
    this.fit();
  }
}
