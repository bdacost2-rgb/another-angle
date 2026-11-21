// Uncharted gallery: a WebGL walk over water through rooms that machine learning groups and shapes.
//
// 1. Grouping: liked works are clustered with spherical k-means on their CLIP embeddings, so rooms
//    follow what the works look and read like, not a fixed list. k is picked by silhouette score.
// 2. Naming: each cluster centroid is compared with the 26 curated theme embeddings; the closest
//    unused theme names the room.
// 3. World-building: the room's works are scored on twelve CLIP design dials (js/dials.js), and one
//    procedural generator builds terrain, growth, structures, sky, weather and light from those
//    numbers. Each room gets a signature dial no other room has. Colors blend the theme palette with
//    colors sampled from the room's own photos, and how tight the cluster is sets how clear the air is.
//    The room's strongest photo becomes a painterly backdrop you walk into.
import { dot } from "./data.js";
import { loadDials as loadSharedDials } from "./dials.js";
import { fitsFocus } from "./focus.js";

const $ = (selector) => document.querySelector(selector);
const UNIT_PX = 64; // scroll pixels per world unit
const SPACING = 6.2; // world units between works along the path
const ORBIT_UNITS = 34; // virtual distance used to turn 360 degrees inside the ring room

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}

function titleOf(item) {
  return item.title || (item.kind === "photo" ? "Untitled photograph" : "Untitled passage");
}

function creatorOf(item) {
  return item.creator || (item.kind === "text" ? "Author not credited" : "Maker not credited");
}

function workKey(item) {
  return `${item.kind}|${titleOf(item).toLocaleLowerCase()}|${creatorOf(item).toLocaleLowerCase()}`;
}

function excerptOf(text, words = 64) {
  const parts = (text || "").trim().split(/\s+/);
  return parts.length > words ? `${parts.slice(0, words).join(" ")}…` : parts.join(" ");
}

function uniqueWorks(items, count, excluded = new Set()) {
  const seen = new Set(excluded);
  return items.filter((item) => {
    const key = workKey(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, count);
}

function diverse(items, count, scoreOf, excluded = new Set()) {
  const pool = items.filter((item) => !excluded.has(item.id));
  const selected = [];
  while (selected.length < count && pool.length) {
    let bestIndex = 0, bestScore = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      const similarity = selected.length ? Math.max(...selected.map((item) => dot(item.vec, pool[i].vec))) : 0;
      const score = scoreOf(pool[i]) - Math.max(0, similarity) * 0.3;
      if (score > bestScore) { bestScore = score; bestIndex = i; }
    }
    const [item] = pool.splice(bestIndex, 1);
    selected.push(item);
  }
  return selected;
}

// ---------- machine learning: grouping and room recipes ----------

function centroidOf(works) {
  const dim = works[0].vec.length;
  const c = new Float32Array(dim);
  for (const w of works) for (let d = 0; d < dim; d++) c[d] += w.vec[d];
  let n = 0;
  for (let d = 0; d < dim; d++) n += c[d] * c[d];
  n = Math.sqrt(n) || 1;
  for (let d = 0; d < dim; d++) c[d] /= n;
  return c;
}

// Spherical k-means with deterministic farthest-point seeding (first seed: the best-scoring work).
function kmeans(works, k, iterations = 24) {
  const centers = [works[0].vec];
  while (centers.length < k) {
    let far = null, farDist = -Infinity;
    for (const w of works) {
      const nearest = Math.max(...centers.map((c) => dot(c, w.vec)));
      if (1 - nearest > farDist) { farDist = 1 - nearest; far = w; }
    }
    centers.push(far.vec);
  }
  let assign = new Array(works.length).fill(0);
  for (let it = 0; it < iterations; it++) {
    const next = works.map((w) => {
      let best = 0, bestSim = -Infinity;
      centers.forEach((c, i) => { const s = dot(c, w.vec); if (s > bestSim) { bestSim = s; best = i; } });
      return best;
    });
    const changed = next.some((a, i) => a !== assign[i]);
    assign = next;
    for (let i = 0; i < k; i++) {
      const members = works.filter((_, j) => assign[j] === i);
      if (members.length) centers[i] = centroidOf(members);
    }
    if (!changed && it > 0) break;
  }
  return Array.from({ length: k }, (_, i) => works.filter((_, j) => assign[j] === i)).filter((g) => g.length);
}

function silhouette(groups) {
  if (groups.length < 2) return -1;
  let total = 0, count = 0;
  for (const [gi, group] of groups.entries()) {
    for (const w of group) {
      const dist = (others) => others.reduce((s, o) => s + (1 - dot(w.vec, o.vec)), 0) / Math.max(1, others.length);
      const a = group.length > 1 ? dist(group.filter((o) => o !== w)) : 0;
      const b = Math.min(...groups.filter((_, j) => j !== gi).map(dist));
      total += group.length > 1 ? (b - a) / Math.max(a, b, 1e-6) : 0;
      count += 1;
    }
  }
  return total / count;
}

function clusterByAesthetic(works) {
  if (works.length < 4) return [works];
  const maxK = Math.min(6, Math.floor(works.length / 2));
  let best = [works], bestScore = -Infinity;
  for (let k = 2; k <= maxK; k++) {
    const groups = kmeans(works, k);
    if (groups.some((g) => g.length < 2)) continue;
    const score = silhouette(groups);
    if (score > bestScore) { bestScore = score; best = groups; }
  }
  return best;
}

// The room's name and atmosphere from embeddings: the nearest unused theme names it, its tightness sets the fog.
function recipeFor(works, themes, takenThemes) {
  const centroid = centroidOf(works);
  const coherence = works.reduce((s, w) => s + dot(w.vec, centroid), 0) / works.length;
  const sims = themes.map((theme) => ({ theme, sim: dot(centroid, theme.vec) })).sort((a, b) => b.sim - a.sim);
  const named = sims.find(({ theme }) => !takenThemes.has(theme.n)) || sims[0];
  if (named) takenThemes.add(named.theme.n);
  return {
    theme: named?.theme || null,
    neighbors: sims.slice(0, 3).map(({ theme, sim }) => ({ name: theme.n, sim })),
    coherence,
  };
}

// ---------- design dials ----------
// Each dial is a CLIP direction between two opposite descriptions (built ↔ overgrown, daylit ↔ nocturnal…),
// made by tools/build_dials.py. A room's works are scored along every dial; the procedural generator
// reads those twelve numbers and builds terrain, growth, structures, sky, weather and light from them.

let dialSet = null;
export async function loadDials() {
  dialSet = await loadSharedDials();
  return dialSet;
}

const DIAL_IDS = ["growth", "night", "crowd", "storm", "float", "warmth", "ruin", "dream", "wet", "scale", "order", "era"];

// How far a room's works sit along each dial, in standard deviations of the archive (each medium
// measured against its own spread, so photos and passages count equally).
function dialScores(works, dials) {
  return dials.map((dial) => {
    let total = 0;
    for (const w of works) {
      const [mean, std] = dial.stats[w.kind] || dial.stats.photo;
      total += (dot(w.vec, dial.vec) - mean) / (std || 1);
    }
    return works.length ? total / works.length : 0;
  });
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Dial values in 0–1 for every room. Each value mixes where the room sits in the whole archive with
// where it sits among your rooms, so rooms pull apart from each other. Then every room gets a signature
// dial that no other room has (an optimal one-to-one matching, as before), pushed toward its extreme:
// no two rooms are built around the same idea.
function designRooms(rooms, dials) {
  if (!dials.length) {
    rooms.forEach((room, i) => {
      const r = seeded(hashString(room.title) + i);
      room.design = Object.fromEntries(DIAL_IDS.map((id) => [id, 0.25 + r() * 0.5]));
      room.signature = [];
    });
    return;
  }
  const Z = rooms.map((room) => dialScores(room.works, dials));
  const values = rooms.map(() => []);
  dials.forEach((_, j) => {
    const col = Z.map((z) => z[j]);
    const mean = col.reduce((a, b) => a + b, 0) / col.length;
    const std = Math.sqrt(col.reduce((a, b) => a + (b - mean) ** 2, 0) / col.length);
    col.forEach((z, i) => { values[i][j] = sigmoid(1.15 * z + 0.75 * (z - mean) / (std + 0.35)); });
  });
  const lead = matchOneToOne(values.map((row) => row.map((v) => Math.abs(v - 0.5))));
  rooms.forEach((room, i) => {
    const row = values[i];
    const j = lead[i];
    if (j >= 0) {
      const sign = row[j] === 0.5 ? (Z[i][j] >= 0 ? 1 : -1) : Math.sign(row[j] - 0.5);
      row[j] = 0.5 + sign * Math.max(Math.abs(row[j] - 0.5), 0.44);
    }
    room.design = Object.fromEntries(dials.map((d, k) => [d.id, row[k]]));
    const ranked = dials.map((d, k) => ({ dial: d, value: row[k], strength: Math.abs(row[k] - 0.5) }))
      .sort((a, b) => b.strength - a.strength);
    if (j >= 0) ranked.unshift(...ranked.splice(ranked.findIndex(({ dial }) => dial === dials[j]), 1));
    room.dialRows = ranked;
    room.signature = ranked.slice(0, 3).map(({ dial, value }) => (value >= 0.5 ? dial.high : dial.low));
  });
}

// The best one-to-one matching of rows to columns (maximum total score), by dynamic programming over
// subsets of columns. Rows beyond the number of columns get -1.
function matchOneToOne(score) {
  const rowsN = score.length;
  const colsN = score[0]?.length || 0;
  const n = Math.min(rowsN, colsN);
  const full = 1 << colsN;
  const best = new Float64Array(full).fill(-Infinity);
  const choice = new Int8Array(full).fill(-1);
  best[0] = 0;
  const popcount = (m) => { let c = 0; while (m) { m &= m - 1; c += 1; } return c; };
  for (let mask = 0; mask < full; mask++) {
    if (best[mask] === -Infinity) continue;
    const r = popcount(mask);
    if (r >= n) continue;
    for (let k = 0; k < colsN; k++) {
      if (mask & (1 << k)) continue;
      const next = mask | (1 << k);
      const value = best[mask] + score[r][k];
      if (value > best[next]) { best[next] = value; choice[next] = k; }
    }
  }
  let end = -1;
  for (let mask = 0; mask < full; mask++) if (popcount(mask) === n && (end < 0 || best[mask] > best[end])) end = mask;
  const result = new Array(rowsN).fill(-1);
  for (let mask = end, r = n - 1; mask > 0 && r >= 0; r--) {
    const k = choice[mask];
    result[r] = k;
    mask &= ~(1 << k);
  }
  return result;
}

// ---------- small helpers for the 3-D world ----------

function seeded(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function noise2(seed) {
  const h = (i, j) => {
    let n = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + seed) | 0;
    n = Math.imul(n ^ (n >>> 13), 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
  };
  return (x, y) => {
    const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const a = h(i, j), b = h(i + 1, j), c = h(i, j + 1), d = h(i + 1, j + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
}
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };

function canvasTexture(THREE, width, height, draw) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext("2d"), width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function wrapLines(ctx, text, maxWidth) {
  const lines = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) { lines.push(line); line = word; } else line = test;
  }
  if (line) lines.push(line);
  return lines;
}

const WATER_VERTEX = /* glsl */`
uniform mat4 textureMatrix;
varying vec4 vUv;
varying vec3 vWorld;
void main() {
  vUv = textureMatrix * vec4(position, 1.0);
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}`;

const WATER_FRAGMENT = /* glsl */`
uniform vec3 color;
uniform sampler2D tDiffuse;
uniform float uTime;
uniform float uRipple;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform vec3 uTint;
varying vec4 vUv;
varying vec3 vWorld;
float wave(vec2 p) {
  return sin(p.x * 1.7 + uTime * 0.7) * sin(p.y * 1.3 - uTime * 0.5)
       + 0.6 * sin(p.x * 3.3 - p.y * 2.1 + uTime * 1.1)
       + 0.35 * sin(p.y * 6.1 + p.x * 0.7 - uTime * 1.6);
}
void main() {
  vec2 p = vWorld.xz;
  float e = 0.08;
  vec2 grad = vec2(wave(p + vec2(e, 0.0)) - wave(p - vec2(e, 0.0)), wave(p + vec2(0.0, e)) - wave(p - vec2(0.0, e)));
  vec4 uv = vUv;
  uv.xy += grad * uRipple * 0.06 * uv.w;
  vec3 reflected = texture2DProj(tDiffuse, uv).rgb;
  vec3 toCam = normalize(cameraPosition - vWorld);
  float fresnel = 0.35 + 0.65 * pow(1.0 - max(toCam.y, 0.0), 3.0);
  vec3 col = mix(uTint, reflected, fresnel);
  col += uTint * 0.12 * smoothstep(0.6, 1.4, length(grad));
  float d = distance(cameraPosition, vWorld);
  float f = 1.0 - exp(-uFogDensity * uFogDensity * d * d);
  gl_FragColor = vec4(mix(col, uFogColor, clamp(f, 0.0, 1.0)), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

// A painterly version of a photo: warped samples, smeared like brushwork, fading into fog at the edges.
const PAINT_VERTEX = /* glsl */`
varying vec2 vUv;
varying vec3 vWorld;
void main() {
  vUv = uv;
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}`;

const PAINT_FRAGMENT = /* glsl */`
uniform sampler2D uMap;
uniform float uHasMap;
uniform vec3 uA;
uniform vec3 uB;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uOpacity;
uniform float uTime;
uniform vec2 uRepeat;
varying vec2 vUv;
varying vec3 vWorld;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
void main() {
  vec2 uv = vUv * uRepeat;
  vec2 cell = floor(uv * vec2(90.0, 60.0));
  float angle = 0.6 + (noise(uv * 3.0) - 0.5) * 1.2;
  vec2 stroke = vec2(cos(angle), sin(angle)) * 0.0045;
  vec2 jitter = (vec2(hash(cell), hash(cell + 17.0)) - 0.5) * 0.007;
  vec3 col = vec3(0.0);
  for (int i = -2; i <= 2; i++) {
    vec2 s = uv + jitter + stroke * float(i);
    s = uRepeat.x > 1.5 ? fract(s) : clamp(s, 0.002, 0.998);
    col += mix(mix(uA, uB, vUv.y), texture2D(uMap, s).rgb, uHasMap);
  }
  col /= 5.0;
  float lum = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(vec3(lum), col, 1.2);
  col = mix(col, mix(uA, uB, vUv.y), 0.28);
  float edge = smoothstep(0.0, 0.3, vUv.y) * smoothstep(1.0, 0.65, vUv.y);
  if (uRepeat.x < 1.5) edge *= smoothstep(0.0, 0.28, vUv.x) * smoothstep(1.0, 0.72, vUv.x);
  edge *= 0.85;
  float d = distance(cameraPosition, vWorld);
  float f = 1.0 - exp(-uFogDensity * uFogDensity * d * d * 0.35);
  gl_FragColor = vec4(mix(col, uFogColor, clamp(f, 0.0, 0.9)), edge * uOpacity);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

const SKY_FRAGMENT = /* glsl */`
uniform vec3 uTop;
uniform vec3 uHorizon;
varying vec3 vDir;
void main() {
  float h = normalize(vDir).y;
  vec3 col = mix(uHorizon, uTop, smoothstep(-0.02, 0.55, h));
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

// Aurora curtains for cold nights: a waving ribbon with vertical rays, added onto the sky.
const AURORA_VERTEX = /* glsl */`
uniform float uTime;
varying vec2 vUv;
void main() {
  vUv = uv;
  vec3 p = position;
  p.z += sin(p.x * 0.04 + uTime * 0.2) * 12.0 + sin(p.x * 0.11 - uTime * 0.13) * 5.0;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;

const AURORA_FRAGMENT = /* glsl */`
uniform vec3 uA;
uniform vec3 uB;
uniform float uTime;
uniform float uOpacity;
varying vec2 vUv;
void main() {
  float curtain = smoothstep(0.0, 0.25, vUv.y) * smoothstep(1.0, 0.35, vUv.y) * smoothstep(0.0, 0.15, vUv.x) * smoothstep(1.0, 0.85, vUv.x);
  float rays = 0.55 + 0.45 * sin(vUv.x * 180.0 + sin(vUv.x * 23.0 + uTime * 0.4) * 4.0);
  gl_FragColor = vec4(mix(uA, uB, vUv.y) * curtain * rays * uOpacity, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

// ---------- the gallery ----------

export function initGallery(items, model, themes = []) {
  const stage = $("#gallery-stage");
  const canvas = $("#gallery-canvas");
  const journey = $("#gallery-journey");
  const progress = $("#gallery-progress");
  const roomName = $("#gallery-room-name");
  const intro = $("#gallery-intro");
  const opening = $("#gallery-opening");
  const loadCount = $("#gallery-load");
  const tip = $("#gallery-tip");
  const previous = $("#gallery-prev");
  const next = $("#gallery-next");
  const exit = $("#gallery-exit");
  const caption = $("#gallery-caption");
  const titleCard = $("#gallery-title");
  const mist = $("#gallery-mist");
  const line = $("#gallery-line");
  const list = $("#gallery-list");
  const dialog = $("#gallery-detail");
  const detailLayout = $("#gallery-detail-layout");
  if (!stage || !canvas || !journey || !progress || !intro || !opening || !tip || !previous || !next || !exit || !dialog || !detailLayout || !list) return;

  if (!model || !items.length) {
    intro.textContent = "The archive couldn't be loaded, so this gallery can't be personalized yet.";
    opening.classList.add("is-ready");
    previous.disabled = true;
    next.disabled = true;
    return;
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  const availableThemes = themes.filter((theme) => theme.vec?.length === model.dim);
  let rooms = [];
  let roomIndex = 0;
  let THREE = null;
  let engine = null;

  function setTip() {
    tip.textContent = finePointer
      ? "Scroll to walk · move to look · click a work"
      : "Scroll to walk · tap a work";
  }
  setTip();

  function updateIntro() {
    if (model.count >= 8 && model.liked.length >= 4) {
      intro.textContent = `Built from your ${model.count} choices. Each room is a group of works that feel alike, and the world around them is shaped by how they feel.`;
    } else if (model.count >= 8) {
      intro.textContent = `Built from your ${model.count} choices. Save a few more works (♥) and they'll become rooms of their own.`;
    } else {
      intro.textContent = "A preview. As you choose works in Inspiration, they'll be grouped into rooms built around how they feel.";
    }
  }

  function makeRooms() {
    // Only the kinds of work you asked to see (Photos · Photos + writing · Writing).
    const fits = (item) => fitsFocus(item);
    const liked = uniqueWorks([...model.liked].filter(fits).sort((a, b) => model.score(b) - model.score(a)), model.liked.length);
    const labeled = new Set(model.labels.keys());
    const used = new Set([...labeled, ...liked.map((item) => item.id)]);
    const usedTitles = new Set(liked.map(workKey));
    const taken = new Set();
    const result = [];
    const recipe = (works) => availableThemes.length
      ? recipeFor(works, availableThemes, taken)
      : { theme: null, neighbors: [], coherence: 0.7 };

    const groups = liked.length >= 2 ? clusterByAesthetic(liked) : liked.length ? [liked] : [];
    for (const works of groups) {
      const r = recipe(works);
      const near = r.neighbors.slice(1).map(({ name }) => name).join(", ");
      result.push({
        ...r,
        title: r.theme?.n || "The works you brought",
        note: `${works.length === 1 ? "One of your works" : `${works.length} of your works that look and read alike`}, grouped by the AI (CLIP)${near ? `. Also close to ${near}` : ""}. ${r.theme?.desc ? `${r.theme.desc[0].toUpperCase()}${r.theme.desc.slice(1)}.` : ""} The scenery, light and weather come from how these works feel.`,
        works,
        fromTaste: true,
      });
    }
    if (!result.length) {
      const first = uniqueWorks(model.pick(24).filter(fits), 4);
      first.forEach((item) => { used.add(item.id); usedTitles.add(workKey(item)); });
      if (first.length) result.push({ ...recipe(first), title: "A first look", note: "Choose works in Inspiration to make these rooms reflect your taste.", works: first });
    }

    const echoes = uniqueWorks(model.pick(36, { exclude: used }).filter(fits), 4, usedTitles);
    echoes.forEach((item) => { used.add(item.id); usedTitles.add(workKey(item)); });
    if (echoes.length) {
      result.push({ ...recipe(echoes), title: "Where the echoes lead", note: "The Compass finds likely matches, varied so the room doesn't repeat itself.", works: echoes });
    }

    const candidates = items.filter((item) => fits(item) && !used.has(item.id) && !usedTitles.has(workKey(item)));
    const edge = uniqueWorks(diverse(candidates, Math.min(12, candidates.length), (item) => {
      const boundary = 1 - Math.abs(2 * model.prob(item) - 1);
      return model.uncertainty(item) * (0.65 + boundary * 0.35);
    }), 4, usedTitles);
    edge.forEach((item) => { used.add(item.id); usedTitles.add(workKey(item)); });
    if (edge.length) {
      result.push({ ...recipe(edge), title: "At the edge of your taste", note: "These works sit near the model's uncertain boundary. Your reaction helps it learn.", works: edge });
    }

    const ringCandidates = [...model.pick(36, { exclude: used }), ...items.filter((item) => !used.has(item.id))].filter(fits);
    let ring = uniqueWorks(ringCandidates, 8, usedTitles);
    if (!ring.length) ring = uniqueWorks(items.filter(fits), 8);
    const lead = result[0];
    result.push({
      theme: lead?.theme || null,
      neighbors: lead?.neighbors || [],
      coherence: lead?.coherence ?? 0.7,
      title: "The room your choices opened",
      note: "A ring of new works on still water, gathered from the directions your taste has opened.",
      works: ring,
      ring: true,
    });
    const finished = result.filter((room) => room.works.length);
    designRooms(finished, dialSet || []);
    publishRooms(finished);
    return finished;
  }

  // The map draws each room as a constellation of its works, and can send you into a room.
  function publishRooms(list) {
    window.GALLERY_ROOMS = list.map((room, index) => ({
      index,
      title: room.title,
      ring: Boolean(room.ring),
      minor: !room.fromTaste, // the Compass's own rooms: drawn faintly, named only on hover
      signature: room.signature || [],
      points: room.works.filter((w) => Number.isFinite(w.x) && Number.isFinite(w.y)).map((w) => [w.x, w.y]),
    }));
    window.dispatchEvent(new CustomEvent("gallery:rooms"));
  }
  let pendingEntry = null;
  // Switching between photos, writing or both rebuilds the rooms (and the map's constellations).
  window.addEventListener("focus:change", () => { if (engine) build(); });
  window.addEventListener("gallery:enter-room", (event) => {
    const index = event.detail?.index;
    if (!Number.isInteger(index)) return;
    if (layout.starts.length) changeRoom(index, true, true);
    else pendingEntry = index;
  });

  // ---------- details dialog and feedback (unchanged behavior) ----------

  function renderFeedback(item, container) {
    const actions = node("div", "gallery-feedback");
    const current = model.labels.get(item.id);
    for (const [label, text] of [[1, "♥ Save"], [0, "× Less like this"]]) { // the same words as the inspiration feed
      const button = node("button", "", text);
      button.type = "button";
      button.setAttribute("aria-pressed", String(current?.y === label && current.w >= 1));
      button.addEventListener("click", () => {
        model.label(item, label);
        const vote = { id: item.id, x: item.x, y: item.y, yv: label, w: 1 };
        const training = window.COMPASS_DATA?.train;
        if (training) {
          const index = training.findIndex((entry) => entry.id === item.id);
          if (index >= 0) training.splice(index, 1, vote);
          else training.push(vote);
        }
        window.dispatchEvent(new CustomEvent("compass:gallery-vote", { detail: vote }));
        tip.textContent = "Your response updated your taste profile. The rooms will regroup when you close this.";
        pendingRebuild = true;
        showDetails(item);
      });
      actions.append(button);
    }
    container.append(actions);
  }

  function showDetails(item) {
    detailLayout.replaceChildren();
    const layout = node("div", "gallery-detail-layout");
    const artwork = node("div", "gallery-detail-art");
    if (item.kind === "photo") {
      const image = node("img");
      image.src = item.img;
      image.alt = titleOf(item);
      image.referrerPolicy = "no-referrer";
      image.addEventListener("error", () => image.replaceWith(node("p", "gallery-detail-words", "The image is resting at the archive. Open the archive record to see it.")));
      artwork.append(image);
    } else {
      artwork.append(node("p", "gallery-detail-words", item.text || ""));
    }

    const copy = node("div", "gallery-detail-copy");
    const close = node("button", "gallery-detail-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close artwork details");
    close.addEventListener("click", () => dialog.close());
    copy.append(close, node("div", "tag", item.kind === "photo" ? "photograph" : "writing"));
    copy.append(node("h2", "", titleOf(item)));
    const credits = node("dl", "gallery-credit-list");
    const fields = [
      [item.kind === "text" ? "Author" : "Artist / maker", creatorOf(item)],
      ["Date", item.date || "Date not listed"],
      ["Collection", item.source || "Source not listed"],
    ];
    for (const [label, value] of fields) credits.append(node("dt", "", label), node("dd", "", value));
    copy.append(credits);

    const anchor = model.because(item);
    if (anchor) {
      copy.append(node("p", "", `The Compass places this near “${titleOf(anchor)}” because their image-and-text embeddings are close in its meaning space.`));
    }
    if (item.link) {
      const source = node("a", "", "Open the archive record ↗");
      source.href = item.link;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      copy.append(source);
    } else {
      copy.append(node("p", "", "No external archive record is available for this item."));
    }

    const related = model.partnerFor(item, new Set([item.id]));
    if (related) {
      const section = node("div", "gallery-related");
      section.append(node("small", "", "A meaning-neighbor in the other medium"));
      const button = node("button", "");
      button.type = "button";
      button.append(node("strong", "", titleOf(related)));
      button.append(node("span", "", ` · ${creatorOf(related)}`));
      button.addEventListener("click", () => showDetails(related));
      section.append(button);
      copy.append(section);
    }
    renderFeedback(item, copy);
    layout.append(artwork, copy);
    detailLayout.append(layout);
    if (!dialog.open) dialog.showModal();
  }

  let pendingRebuild = false;
  dialog.addEventListener("close", () => {
    if (!pendingRebuild) return;
    pendingRebuild = false;
    const inRing = rooms[roomIndex]?.ring;
    build();
    if (inRing) changeRoom(rooms.length - 1, true);
    setTip();
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  // ---------- scroll and room navigation ----------

  let layout = { starts: [], ringCenter: 1, total: 1, works: [] };

  function scrollProgress() {
    const stickyTop = parseFloat(getComputedStyle(stage).top) || 0;
    const travel = Math.max(1, journey.offsetHeight - stage.offsetHeight);
    return Math.max(0, Math.min(1, (stickyTop - journey.getBoundingClientRect().top) / travel));
  }

  function scrollToDistance(distance, instant = false) {
    const stickyTop = parseFloat(getComputedStyle(stage).top) || 0;
    const travel = Math.max(0, journey.offsetHeight - stage.offsetHeight);
    const start = journey.getBoundingClientRect().top + window.scrollY - stickyTop;
    window.scrollTo({ top: start + travel * (distance + 6) / (layout.total + 6), behavior: instant || reducedMotion.matches ? "auto" : "smooth" });
  }

  function changeRoom(index, center = false, instant = false) {
    const target = Math.max(0, Math.min(rooms.length - 1, index));
    const room = rooms[target];
    scrollToDistance(center && room?.ring ? layout.ringCenter : layout.starts[target] + 0.5, instant);
  }

  // The end of the walk points somewhere: write from the room you're in, or explore the map further.
  const endCard = $("#gallery-end");
  let endShown = false;
  function showEnd(on) {
    if (!endCard || on === endShown) return;
    endShown = on;
    endCard.classList.toggle("is-on", on);
    endCard.setAttribute("aria-hidden", String(!on));
    endCard.querySelectorAll("button").forEach((b) => (b.tabIndex = on ? 0 : -1));
  }
  $("#gallery-write")?.addEventListener("click", () => {
    const room = rooms[roomIndex] || rooms[rooms.length - 1];
    if (!room) return;
    const photo = room.works.find((w) => w.kind === "photo");
    const text = room.works.find((w) => w.kind === "text");
    try {
      localStorage.setItem("anotherangle.handoff.v1", JSON.stringify({ photo: photo?.id || null, text: text?.id || null, words: room.signature || [], room: room.title }));
    } catch { /* without storage the coach still opens, just without the room */ }
    location.href = "index.html#write";
  });
  $("#gallery-explore")?.addEventListener("click", () => document.querySelector('.rail [data-pane="map"]')?.click());

  function updateControls(distance, p) {
    const room = rooms[roomIndex];
    showEnd(Boolean(room?.ring) && (p > 0.97 || distance >= layout.ringCenter + ORBIT_UNITS * 0.55));
    tip.classList.toggle("is-done", distance > 4); // the "scroll to walk" hint leaves once you're walking
    const label = `${room?.ring ? "ring room" : "room"} ${String(roomIndex + 1).padStart(2, "0")} / ${String(rooms.length).padStart(2, "0")}`;
    if (progress.textContent !== label) progress.textContent = label;
    if (roomName && room && roomName.textContent !== room.title) roomName.textContent = room.title;
    previous.disabled = roomIndex === 0;
    next.disabled = roomIndex >= rooms.length - 1;
    const nextLabel = roomIndex === rooms.length - 2 ? "Enter the ring →" : "Next room →";
    if (next.textContent !== nextLabel) next.textContent = nextLabel;
    if (line) line.style.setProperty("--p", String(p));
    opening.style.opacity = String(Math.max(0, 1 - (distance + 6) / 7));
    opening.style.pointerEvents = distance < -4 ? "auto" : "none";
  }

  let titleTimer = 0;
  const whyButton = $("#gallery-why");
  const whyPanel = $("#gallery-why-panel");
  whyButton?.addEventListener("click", () => {
    const open = whyPanel.hidden;
    whyPanel.hidden = !open;
    whyButton.setAttribute("aria-expanded", String(open));
  });
  function announceRoom(room, index) {
    if (!titleCard) return;
    // Short on purpose: the room's name and how it feels. The longer story is behind "why this room?".
    titleCard.querySelector("small").textContent = room.ring ? "final room" : `room ${String(index + 1).padStart(2, "0")}`;
    titleCard.querySelector("strong").textContent = room.title;
    titleCard.querySelector("span").textContent = (room.signature || []).join(" · ");
    whyPanel.querySelector("p").textContent = room.note;
    const readout = whyPanel.querySelector(".gallery-dials");
    if (readout) {
      readout.replaceChildren();
      (room.dialRows || []).slice(0, 6).forEach(({ dial, value }, i) => {
        const row = node("div", "gallery-dial");
        row.style.setProperty("--v", value.toFixed(3));
        row.style.setProperty("--i", String(i));
        const bar = node("i");
        bar.setAttribute("aria-hidden", "true");
        row.append(node("span", value < 0.5 ? "is-near" : "", dial.low), bar, node("span", value >= 0.5 ? "is-near" : "", dial.high));
        row.setAttribute("aria-label", `${Math.round(value * 100)}% ${dial.high}, ${Math.round((1 - value) * 100)}% ${dial.low}`);
        readout.append(row);
      });
    }
    titleCard.classList.remove("is-on");
    void titleCard.offsetWidth;
    titleCard.classList.add("is-on");
    clearTimeout(titleTimer);
    titleTimer = setTimeout(() => titleCard.classList.remove("is-on"), 4600);
    if (mist && !reducedMotion.matches) {
      mist.classList.remove("is-clearing");
      void mist.offsetWidth;
      mist.classList.add("is-clearing");
    }
  }

  function setCaption(entry) {
    if (!caption) return;
    const key = entry ? entry.item.id : "";
    if (caption.dataset.key === key) return;
    caption.dataset.key = key;
    caption.classList.toggle("is-on", Boolean(entry));
    if (!entry) return;
    caption.querySelector("strong").textContent = titleOf(entry.item);
    caption.querySelector("small").textContent = [creatorOf(entry.item), entry.item.date?.slice(0, 4), entry.item.source].filter(Boolean).join(" · ");
  }

  function renderList() {
    list.replaceChildren();
    for (const entry of layout.works) {
      const li = node("li");
      const button = node("button", "", `${titleOf(entry.item)} — ${creatorOf(entry.item)}`);
      button.type = "button";
      button.addEventListener("focus", () => scrollToDistance(entry.ring ? layout.ringCenter + entry.orbit * ORBIT_UNITS : entry.distance - 5));
      button.addEventListener("click", () => showDetails(entry.item));
      li.append(button);
      list.append(li);
    }
  }

  previous.addEventListener("click", () => changeRoom(roomIndex - 1));
  next.addEventListener("click", () => changeRoom(roomIndex + 1, true));
  exit.addEventListener("click", () => document.querySelector('.rail [data-pane="seed"]')?.click());
  stage.addEventListener("keydown", (event) => {
    if (event.target.closest("#gallery-list")) return;
    if (event.key === "ArrowRight") { event.preventDefault(); changeRoom(roomIndex + 1, true); }
    if (event.key === "ArrowLeft") { event.preventDefault(); changeRoom(roomIndex - 1); }
  });

  // ---------- the WebGL engine ----------

  const textureCache = new Map();
  const colorCache = new Map();

  function loadTexture(item) {
    if (textureCache.has(item.id)) return textureCache.get(item.id);
    const promise = new Promise((resolve) => {
      if (item.kind !== "photo" || !item.img) { resolve(null); return; }
      const loader = new THREE.TextureLoader();
      loader.setCrossOrigin("anonymous");
      const timer = setTimeout(() => resolve(null), 9000);
      loader.load(item.img, (texture) => {
        clearTimeout(timer);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = 4;
        try {
          const c = document.createElement("canvas");
          c.width = c.height = 12;
          const ctx = c.getContext("2d", { willReadFrequently: true });
          ctx.drawImage(texture.image, 0, 0, 12, 12);
          const data = ctx.getImageData(0, 0, 12, 12).data;
          let r = 0, g = 0, b = 0, vivid = [0, 0, 0], vividScore = -1;
          for (let i = 0; i < data.length; i += 4) {
            r += data[i]; g += data[i + 1]; b += data[i + 2];
            const sat = Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]);
            if (sat > vividScore) { vividScore = sat; vivid = [data[i], data[i + 1], data[i + 2]]; }
          }
          const n = data.length / 4;
          colorCache.set(item.id, { mean: [r / n / 255, g / n / 255, b / n / 255], vivid: vivid.map((v) => v / 255), sat: vividScore / 255 });
        } catch { /* a tainted canvas only means no sampled colors */ }
        resolve(texture);
      }, undefined, () => { clearTimeout(timer); resolve(null); });
    });
    textureCache.set(item.id, promise);
    return promise;
  }

  function textTexture(item) {
    return canvasTexture(THREE, 768, 1024, (ctx, w, h) => {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, "#f3ece0");
      grad.addColorStop(1, "#e3d8c6");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "rgba(60,40,20,.05)";
      for (let i = 0; i < 1400; i++) ctx.fillRect(Math.random() * w, Math.random() * h, 1.5, 1.5);
      ctx.fillStyle = "#2a2622";
      if (!item.text) {
        // A photo that couldn't load: a title card, like a museum label standing in for the work.
        ctx.textAlign = "center";
        ctx.font = "italic 400 58px Newsreader, Georgia, serif";
        const big = wrapLines(ctx, titleOf(item), w - 160).slice(0, 5);
        big.forEach((t, i) => ctx.fillText(t, w / 2, h / 2 - (big.length - 1) * 34 + i * 68));
        ctx.fillStyle = "#7a6f62";
        ctx.font = "500 18px 'Google Sans Flex', Arial, sans-serif";
        ctx.fillText([creatorOf(item), item.date?.slice(0, 4)].filter(Boolean).join(" · ").toUpperCase(), w / 2, h - 110);
        ctx.fillText("IMAGE AT THE ARCHIVE · SELECT TO OPEN", w / 2, h - 80);
        return;
      }
      ctx.font = "italic 400 34px Newsreader, Georgia, serif";
      const title = wrapLines(ctx, titleOf(item), w - 150).slice(0, 2);
      title.forEach((t, i) => ctx.fillText(t, 75, 120 + i * 42));
      ctx.font = "400 30px Newsreader, Georgia, serif";
      const lines = wrapLines(ctx, excerptOf(item.text, 90), w - 150);
      const maxLines = Math.floor((h - 300) / 46);
      lines.slice(0, maxLines).forEach((t, i) => ctx.fillText(i === maxLines - 1 && lines.length > maxLines ? `${t}…` : t, 75, 120 + title.length * 42 + 50 + i * 46));
      ctx.fillStyle = "#7a6f62";
      ctx.font = "500 18px 'Google Sans Flex', Arial, sans-serif";
      ctx.fillText(creatorOf(item).toUpperCase(), 75, h - 80);
    });
  }

  function paletteFor(room) {
    const C = (hex) => new THREE.Color(hex);
    const pal = (room.theme?.pal || []).filter((c) => /^#[0-9a-f]{6}$/i.test(c));
    const theme = pal.length >= 4 ? pal.map(C) : ["#1a2631", "#344e51", "#c9b48f", "#090f16"].map(C);
    const sampled = room.works.map((w) => colorCache.get(w.id)).filter(Boolean);
    const mean = new THREE.Color(0.35, 0.38, 0.42);
    const vivid = theme[1].clone();
    if (sampled.length) {
      mean.setRGB(...[0, 1, 2].map((i) => sampled.reduce((s, c) => s + c.mean[i], 0) / sampled.length), THREE.SRGBColorSpace);
      const best = sampled.reduce((a, b) => (b.sat > a.sat ? b : a));
      vivid.setRGB(...best.vivid, THREE.SRGBColorSpace).lerp(theme[1], 0.5);
    }
    const lum = (c) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    const sorted = [...theme].sort((a, b) => lum(a) - lum(b));
    const dark = sorted[0], light = sorted[3];
    const calm = (color, amount) => { const g = lum(color); return color.lerp(new THREE.Color(g, g, g), amount); };
    // The dials steer the light: warmth tilts every color toward amber or blue, daylight lifts the
    // fog and sky into a pale haze, night deepens them.
    const D = room.design || {};
    const day = 1 - (D.night ?? 0.5);
    const warm = (D.warmth ?? 0.5) - 0.5;
    const temp = new THREE.Color(warm > 0 ? 0xffb36b : 0x86a9ff);
    const lightC = calm(light.clone(), 0.3).lerp(temp, Math.abs(warm) * 0.7);
    vivid.lerp(temp, Math.abs(warm) * 0.3);
    const fog = calm(dark.clone().lerp(mean, 0.35).lerp(light, 0.12), 0.62);
    fog.lerp(calm(lightC.clone(), 0.45), day * day * 0.55).lerp(temp.clone().multiplyScalar(lum(fog) * 1.6), Math.abs(warm) * 0.25);
    const l = lum(fog);
    const cap = 0.04 + day * day * 0.17;
    if (l < 0.012) fog.lerp(light, 0.08);
    if (l > cap) fog.multiplyScalar(cap / l);
    const topNight = calm(dark.clone(), 0.55).multiplyScalar(0.5);
    const topDay = calm(lightC.clone().lerp(new THREE.Color(0x9db6d0), 0.45), 0.25).multiplyScalar(0.75);
    return {
      fog,
      top: topNight.lerp(topDay, Math.pow(day, 1.6)),
      deep: dark.clone(),
      mid: sorted[1].clone().lerp(mean, 0.3),
      accent: sorted[2].clone(),
      light: lightC,
      glow: vivid,
      tint: calm(dark.clone().lerp(sorted[1], 0.35), 0.5).multiplyScalar(0.6).lerp(fog, day * 0.4),
    };
  }

  async function start() {
    try {
      THREE = await import("three");
      const { Reflector } = await import("three/addons/objects/Reflector.js");
      engine = createEngine(Reflector);
      await loadDials();
    } catch (error) {
      console.warn("Gallery: WebGL unavailable", error);
      stage.classList.add("is-fallback");
      intro.textContent = "Your browser couldn't start the 3-D gallery, so the works are listed below instead.";
      opening.classList.add("is-ready");
      await loadDials();
      rooms = makeRooms();
      layout.works = rooms.flatMap((room) => room.works.map((item) => ({ item, distance: 0, ring: false, orbit: 0 })));
      renderList();
      return;
    }
    build();
  }

  async function build() {
    updateIntro();
    rooms = makeRooms();
    roomIndex = 0;
    opening.classList.remove("is-ready");
    const all = rooms.flatMap((room) => room.works);
    let done = 0;
    if (loadCount) loadCount.textContent = "0";
    await Promise.all(all.map((item) => loadTexture(item).then(() => {
      done += 1;
      if (loadCount) loadCount.textContent = String(Math.round((done / all.length) * 100));
    })));
    try { await document.fonts.load("italic 400 34px Newsreader"); await document.fonts.load("400 30px Newsreader"); } catch { /* fonts are optional */ }
    await engine.build(rooms);
    renderList();
    opening.classList.add("is-ready");
    if (pendingEntry != null) { const index = pendingEntry; pendingEntry = null; requestAnimationFrame(() => changeRoom(index, true, true)); }
  }

  function createEngine(Reflector) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, finePointer ? 1.75 : 1.35));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0b0f16, 0.03);
    const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 600);
    camera.rotation.order = "YXZ";
    const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 1.4);
    const sun = new THREE.DirectionalLight(0xffffff, 1.3);
    sun.position.set(6, 14, 8);
    scene.add(hemi, sun);

    const skyUniforms = { uTop: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() } };
    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(500, 32, 16),
      new THREE.ShaderMaterial({
        uniforms: skyUniforms,
        vertexShader: "varying vec3 vDir; void main(){ vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: SKY_FRAGMENT,
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
      }),
    );
    sky.renderOrder = -1;
    scene.add(sky);

    const glowTexture = canvasTexture(THREE, 128, 128, (ctx, w) => {
      const g = ctx.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
      g.addColorStop(0, "rgba(255,255,255,1)");
      g.addColorStop(0.35, "rgba(255,255,255,.35)");
      g.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, w);
    });

    let water = null;
    let world = null;
    let roomEnvs = [];
    let picks = [];
    let animators = [];
    let skies = [];
    let roomDisposables = [];
    let hovered = null;
    let lastRoom = -1;
    const pointer = new THREE.Vector2(9, 9);
    const raycaster = new THREE.Raycaster();
    const look = { yaw: 0, pitch: 0, tYaw: 0, tPitch: 0 };
    let cam = { distance: -6, orbit: 0 };
    const clock = new THREE.Clock();
    let visible = true;
    let running = false;

    function makeWater(length) {
      if (water) { water.geometry.dispose(); water.dispose(); scene.remove(water); }
      const scale = finePointer ? 0.5 : 0.35;
      water = new Reflector(new THREE.PlaneGeometry(220, length + 260), {
        textureWidth: Math.max(256, stage.clientWidth * scale * renderer.getPixelRatio()),
        textureHeight: Math.max(256, stage.clientHeight * scale * renderer.getPixelRatio()),
        clipBias: 0.003,
        color: 0x888888,
        shader: {
          name: "GalleryWater",
          uniforms: {
            color: { value: null }, tDiffuse: { value: null }, textureMatrix: { value: null },
            uTime: { value: 0 }, uRipple: { value: 0.6 }, uFogColor: { value: new THREE.Color() },
            uFogDensity: { value: 0.03 }, uTint: { value: new THREE.Color() },
          },
          vertexShader: WATER_VERTEX,
          fragmentShader: WATER_FRAGMENT,
        },
      });
      water.rotation.x = -Math.PI / 2;
      water.position.set(0, 0, -length / 2 + 40);
      scene.add(water);
    }

    function track(...things) { roomDisposables.push(...things); return things[0]; }

    function paintPlane(texture, colors, width, height, repeat = [1, 1]) {
      const material = track(new THREE.ShaderMaterial({
        uniforms: {
          uMap: { value: texture }, uHasMap: { value: texture ? 1 : 0 },
          uA: { value: colors.deep.clone().lerp(colors.glow, 0.4) }, uB: { value: colors.light.clone() },
          uFogColor: { value: colors.fog.clone() }, uFogDensity: { value: 0.03 },
          uOpacity: { value: 1 }, uTime: { value: 0 }, uRepeat: { value: new THREE.Vector2(...repeat) },
        },
        vertexShader: PAINT_VERTEX,
        fragmentShader: PAINT_FRAGMENT,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      }));
      if (texture) { texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.needsUpdate = true; }
      return material;
    }

    function sway(material, amount) {
      material.onBeforeCompile = (shader) => {
        shader.uniforms.uTime = timeUniform;
        shader.vertexShader = `uniform float uTime;\n${shader.vertexShader}`.replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
          #ifdef USE_INSTANCING
          float phase = instanceMatrix[3].x * 0.3 + instanceMatrix[3].z * 0.45;
          transformed.x += sin(uTime * 1.3 + phase) * position.y * ${amount.toFixed(2)};
          transformed.z += cos(uTime * 1.1 + phase) * position.y * ${(amount * 0.6).toFixed(2)};
          #endif`,
        );
      };
      material.customProgramCacheKey = () => `sway-${amount}`;
      return material;
    }
    const timeUniform = { value: 0 };

    function scatter(group, geometry, material, count, place) {
      const mesh = new THREE.InstancedMesh(track(geometry), track(material), Math.max(1, Math.round(count)));
      const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), e = new THREE.Euler();
      const color = new THREE.Color();
      for (let i = 0; i < mesh.count; i++) {
        const o = place(i, color);
        p.set(o.x, o.y, o.z);
        e.set(o.rx || 0, o.ry || 0, o.rz || 0);
        q.setFromEuler(e);
        s.set(o.sx ?? o.s ?? 1, o.sy ?? o.s ?? 1, o.sz ?? o.s ?? 1);
        m.compose(p, q, s);
        mesh.setMatrixAt(i, m);
        if (o.color) mesh.setColorAt(i, o.color);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      group.add(mesh);
      return mesh;
    }

    const side = (r, min, max) => (r() < 0.5 ? -1 : 1) * (min + r() * (max - min));

    // ----- the room generator: one procedural world, steered by twelve continuous CLIP dials -----
    // Nothing here is a fixed scene. Every quantity (how far the shore is, how tall the hills, whether
    // trees droop or point, how ordered the buildings are, what falls from the sky) is a function of
    // the dials, so each room is its own point in a continuous design space.
    function generate({ group, r, z0, len, D, c, ring }) {
      const zc = z0 - len / 2;
      const half = len / 2 + 12;
      const gap = ring ? 22 : 7 + D.wet * 15 + (1 - D.crowd) * 5;
      const noise = noise2((r() * 2 ** 31) | 0);
      const fbm = (x, z) => {
        let a = 0, amp = 1, f = 1, s = 0;
        for (let k = 0; k < 4; k++) { a += amp * noise(x * f + k * 17.3, z * f); s += amp; amp *= 0.5; f *= 2.03; }
        return a / s;
      };
      const amp = 1.2 + D.scale * 9 + (1 - D.wet) * 1.6;
      const dune = (1 - D.wet) * (1 - D.growth);
      const steps = D.order * (1 - D.growth);
      const phase = r() * 10;
      const lift = clamp01((D.float - 0.42) * 1.8);

      // terrain: rolling hills (organic), wind ridges (dry and bare), terraces (ordered and built),
      // jagged noise (turbulent), height (monumental). It sinks into the water at the room's ends,
      // so each room is its own island.
      const heightAt = (x, z) => {
        const ax = Math.abs(x);
        const rise = smooth(gap, gap + 9 + (1 - D.scale) * 6, ax) * smooth(0, 12, half - Math.abs(z - zc));
        if (rise <= 0) return -0.7;
        const hills = fbm(x * 0.045, z * 0.045);
        const ridge = Math.pow(Math.abs(Math.sin(x * 0.12 + Math.sin(z * 0.055 + phase) * 1.9)), 1.5);
        let h = lerp(hills, ridge * 0.75 + hills * 0.35, dune);
        h += D.storm * Math.abs(noise(x * 0.2, z * 0.2) * 2 - 1) * 0.45;
        h *= amp * (0.45 + Math.min(ax, 60) * 0.02);
        if (steps > 0.2) { const step = 1 + D.scale * 2.2; h = lerp(h, Math.round(h / step) * step, Math.min(1, steps * 1.3)); }
        return -0.7 + (h + 0.4) * rise;
      };

      const earth = c.deep.clone().lerp(c.mid, 0.45);
      const green = c.mid.clone().lerp(new THREE.Color(0x58763c), 0.55);
      const sand = c.accent.clone().lerp(new THREE.Color(0xd4a46c), 0.5);
      const rock = c.mid.clone().lerp(new THREE.Color(0x77736d), 0.45);
      const snow = new THREE.Color(0xe9eef4);
      const lush = clamp01(D.growth * 1.3 - 0.1) * (0.4 + 0.6 * D.wet);
      const low = earth.clone().lerp(green, lush).lerp(sand, dune * 0.8);
      {
        const rows = Math.max(24, Math.round((len + 24) / 1.5));
        const geo = track(new THREE.PlaneGeometry(170, len + 24, 128, rows).rotateX(-Math.PI / 2));
        const pos = geo.attributes.position;
        const colors = new Float32Array(pos.count * 3);
        const col = new THREE.Color();
        for (let i = 0; i < pos.count; i++) {
          const x = pos.getX(i), z = pos.getZ(i) + zc;
          const y = heightAt(x, z);
          pos.setY(i, y);
          const k = clamp01(y / (amp * 1.2));
          col.copy(low).lerp(rock, smooth(0.25, 0.85, k) * (1 - dune * 0.6));
          if (D.warmth < 0.42) col.lerp(snow, smooth(0.45, 0.95, k) * Math.min(1, (0.42 - D.warmth) * 3));
          col.offsetHSL(0, 0, (noise(x * 0.3, z * 0.3) - 0.5) * 0.05);
          col.toArray(colors, i * 3);
        }
        geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geo.computeVertexNormals();
        const land = new THREE.Mesh(geo, track(new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: D.storm + steps > 0.9 })));
        land.position.z = zc;
        group.add(land);
      }

      const spot = (minX, maxX, minY = 0.05) => {
        for (let k = 0; k < 10; k++) {
          const x = (r() < 0.5 ? -1 : 1) * lerp(minX, maxX, Math.pow(r(), 1.4));
          const z = zc + (r() - 0.5) * len;
          const y = heightAt(x, z);
          if (y > minY) return { x, y, z };
        }
        return null;
      };
      const many = (n, make) => { const out = []; for (let i = 0; i < n; i++) { const o = make(i); if (o) out.push(o); } return out; };

      // ----- growth: grass, flowers, trees whose form follows the dials -----
      const F = clamp01(D.growth * 1.5 - 0.3);
      const grassCol = green.clone().lerp(new THREE.Color(0xc2a35a), clamp01((D.warmth - 0.5) * 1.2 + (1 - D.wet) * 0.4));
      if (F > 0.02) {
        const blades = many(Math.round(3400 * F * (0.45 + 0.55 * D.crowd)), () => spot(gap + 0.5, 40));
        const blade = new THREE.ConeGeometry(0.05, 0.9, 3, 3).translate(0, 0.45, 0);
        if (blades.length) scatter(group, blade, sway(new THREE.MeshLambertMaterial({ color: 0xffffff }), +(0.08 + D.storm * 0.4).toFixed(2)), blades.length, (i, col) => ({
          ...blades[i], y: blades[i].y - 0.05, ry: r() * 6, rz: (r() - 0.5) * 0.4, s: 0.6 + r() * 1.1 + D.scale * 0.4,
          color: col.copy(grassCol).offsetHSL((r() - 0.5) * 0.05, 0, (r() - 0.5) * 0.14).clone(),
        }));
        const bloom = F * clamp01(D.warmth * 1.4 - 0.2) * (1 - D.ruin * 0.6);
        const flowers = many(Math.round(900 * bloom), () => spot(gap + 0.5, 30));
        if (flowers.length) scatter(group, new THREE.IcosahedronGeometry(0.07, 0), new THREE.MeshBasicMaterial({ color: 0xffffff }), flowers.length, (i) => {
          const pick = r();
          return { ...flowers[i], y: flowers[i].y + 0.5 + r() * 0.4, s: 0.7 + r(), color: (pick < 0.45 ? c.accent : pick < 0.8 ? c.light : c.glow).clone() };
        });

        const trees = many(Math.round(F * (5 + 34 * D.crowd)), () => {
          const s = spot(gap + 1, 55, 0.2);
          if (!s) return null;
          return {
            ...s,
            h: (2.6 + D.scale * 5) * (0.7 + r() * 0.6),
            conifer: r() < clamp01((0.55 - D.warmth) * 1.8),
            bare: r() < D.ruin * (1 - D.wet * 0.5) * 0.8,
          };
        });
        if (trees.length) {
          const bark = c.deep.clone().lerp(new THREE.Color(0x3a2a1e), 0.5).multiplyScalar(0.8);
          scatter(group, new THREE.CylinderGeometry(0.16, 0.3, 1, 7).translate(0, 0.5, 0), new THREE.MeshLambertMaterial({ color: bark }), trees.length, (i) => ({
            ...trees[i], y: trees[i].y - 0.15, sx: 1 + D.scale * 0.5, sz: 1 + D.scale * 0.5, sy: trees[i].h,
          }));
          const leafCol = grassCol.clone().multiplyScalar(0.78);
          const droop = D.wet * 0.9;
          const crown = 1 + D.scale * 0.5;
          const broad = trees.filter((t) => !t.conifer && !t.bare);
          const blobs = broad.flatMap((t) => Array.from({ length: 7 }, () => {
            const a = r() * Math.PI * 2, d = (0.4 + r() * 1.2) * crown * (1 + droop * 0.5);
            return { x: t.x + Math.cos(a) * d, y: t.y + t.h - r() * 1.1 - r() * droop * t.h * 0.45, z: t.z + Math.sin(a) * d, sx: (1 + r() * 0.8) * crown, sy: (1.2 + r() * 1.1) * crown * (1 + droop), sz: (1 + r() * 0.8) * crown };
          }));
          if (blobs.length) scatter(group, new THREE.IcosahedronGeometry(1, 1), sway(new THREE.MeshLambertMaterial({ color: 0xffffff }), 0.04), blobs.length, (i, col) => ({
            ...blobs[i], color: col.copy(leafCol).offsetHSL((r() - 0.5) * (0.04 + D.warmth * 0.06), 0, (r() - 0.5) * 0.1).clone(),
          }));
          const cones = trees.filter((t) => t.conifer && !t.bare).flatMap((t) => [0, 1, 2].map((k) => ({
            x: t.x, y: t.y + t.h * (0.3 + k * 0.24), z: t.z, s: (1.5 - k * 0.38) * crown, sy: (1.5 - k * 0.3) * crown,
          })));
          if (cones.length) scatter(group, new THREE.ConeGeometry(1, 2.2, 8).translate(0, 1.1, 0), new THREE.MeshLambertMaterial({ color: leafCol.clone().lerp(c.deep, 0.35) }), cones.length, (i) => ({ ...cones[i], sx: cones[i].s, sz: cones[i].s }));
          const twigs = trees.filter((t) => t.bare).flatMap((t) => [0, 1, 2, 3].map((k) => ({
            x: t.x, y: t.y + t.h * (0.5 + k * 0.12), z: t.z, ry: r() * 6.28, rz: (k % 2 ? 1 : -1) * (0.55 + r() * 0.5), sy: t.h * (0.45 - k * 0.07),
          })));
          if (twigs.length) scatter(group, new THREE.CylinderGeometry(0.04, 0.09, 1, 5).translate(0, 0.5, 0), new THREE.MeshLambertMaterial({ color: bark }), twigs.length, (i) => twigs[i]);
        }
      }

      // ----- the water's edge: lily pads where it is wet, green and calm; reeds at the shore -----
      const marsh = D.wet * D.growth * (1 - D.storm * 0.7);
      if (marsh > 0.12) {
        const pad = new THREE.CircleGeometry(0.5, 22, 0.35, Math.PI * 2 - 0.35).rotateX(-Math.PI / 2);
        const padColor = c.mid.clone().lerp(new THREE.Color(0x4f7a4a), 0.55);
        const pads = [];
        scatter(group, pad, new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide }), 520 * marsh, (i, col) => {
          const o = { x: side(r, 1.4, ring ? 20 : gap + 3), y: 0.02 + r() * 0.01, z: zc + (r() - 0.5) * len, ry: r() * 6.28, s: 0.5 + r() * 1.3 };
          pads.push(o);
          return { ...o, color: col.copy(padColor).offsetHSL((r() - 0.5) * 0.06, 0, (r() - 0.5) * 0.14).clone() };
        });
        const petal = new THREE.ConeGeometry(0.16, 0.22, 7, 1, true).translate(0, 0.11, 0);
        scatter(group, petal, new THREE.MeshLambertMaterial({ color: 0xffffff, emissive: c.glow.clone().multiplyScalar(0.25), side: THREE.DoubleSide }), pads.length * 0.22 * D.warmth + 1, () => {
          const p = pads[Math.floor(r() * pads.length)];
          return { x: p.x, y: 0.05, z: p.z, rx: Math.PI, s: 0.8 + r() * 0.6, color: c.light.clone().lerp(new THREE.Color(0xf0a7b8), 0.5 + r() * 0.3) };
        });
      }
      const reedAmount = D.wet * clamp01(D.growth * 1.4 - 0.2);
      if (reedAmount > 0.15 && !ring) {
        scatter(group, new THREE.CylinderGeometry(0.02, 0.035, 1, 4).translate(0, 0.5, 0), sway(new THREE.MeshLambertMaterial({ color: grassCol.clone().multiplyScalar(0.7) }), 0.12), 700 * reedAmount, () => ({
          x: side(r, gap - 2.5, gap + 2), y: -0.2, z: zc + (r() - 0.5) * len, rz: (r() - 0.5) * 0.25, sy: 1 + r() * 1.6, s: 1,
        }));
      }

      // ----- structures: towers (futuristic), columns (ancient, ordered), monoliths (sparse, vast) -----
      const B = clamp01(1.4 * (1 - D.growth) - 0.2);
      const buildingCount = Math.round(B * (5 + 80 * D.crowd * (0.5 + D.scale * 0.5)));
      if (buildingCount) {
        const spacing = 4.5 + (1 - D.crowd) * 7;
        const jitter = 1 - D.order;
        const weights = [0.25 + D.era * 1.2, (1 - D.era) * (0.3 + D.order), (1 - D.crowd) * D.scale * 1.2 + D.dream * 0.3];
        const totalWeight = weights[0] + weights[1] + weights[2];
        const families = [[], [], []];
        const reach = 8 + 50 * (0.3 + D.scale * 0.7);
        for (let i = 0; i < buildingCount; i++) {
          let x = side(r, gap + 1, gap + reach);
          let z = zc + (r() - 0.5) * len;
          // order snaps buildings to a grid and squares them to the path; disorder scatters and turns them
          x = lerp(Math.round(x / spacing) * spacing, x, jitter);
          z = lerp(Math.round((z - zc) / spacing) * spacing + zc, z, jitter);
          if (Math.abs(x) < gap) x = (x < 0 ? -1 : 1) * (gap + 0.5);
          const ground = Math.max(-0.6, heightAt(x, z));
          let h = 2 + Math.pow(r(), 1.6) * (5 + D.scale * 34) * (0.6 + Math.abs(x) / 40);
          const ruined = r() < D.ruin;
          if (ruined) h *= 0.3 + r() * 0.5;
          const fp = 1.2 + r() * 2.6 + D.scale * 2.5;
          const o = { x, y: ground - 0.3, z, h, fp, rx: 0, ry: jitter * r() * 1.2, rz: ruined ? (r() - 0.5) * D.ruin * 0.5 : 0, top: 0 };
          if (r() < lift * 0.7) o.y += 3 + r() * 14 * lift; // weightless: lifted off the ground
          if (r() < clamp01((D.dream - 0.55) * 1.6)) { o.rx = Math.PI; o.y = Math.max(24, h + 6) + r() * 10; } // dreamlike: hanging from the sky
          o.top = o.rx ? o.y - h : o.y + h;
          let pick = r() * totalWeight, f = 0;
          while (f < 2 && pick > weights[f]) { pick -= weights[f]; f += 1; }
          if (f === 2) { o.x += (o.x < 0 ? -1 : 1) * (10 + D.scale * 12); o.y = Math.max(-0.6, heightAt(o.x, o.z)) - 0.3; } // monoliths stand back, in the distance
          families[f].push(o);
        }
        const [towers, columns, slabs] = families;
        if (towers.length) {
          const neon = D.era > 0.6;
          const lit = 0.12 + 0.42 * (1 - D.ruin);
          const windows = track(canvasTexture(THREE, 64, 128, (ctx, W, H) => {
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, W, H);
            for (let y = 4; y < H - 4; y += 8) for (let x = 4; x < W - 4; x += 8) {
              if (r() > lit) continue;
              ctx.fillStyle = neon && r() < 0.4
                ? `hsla(${r() < 0.5 ? 188 : 312}, 90%, 65%, ${0.6 + r() * 0.4})`
                : `rgba(255,${200 + r() * 55},${140 + r() * 80},${0.5 + r() * 0.5})`;
              ctx.fillRect(x, y, 4, 5);
            }
          }));
          const facade = c.deep.clone().lerp(c.mid, 0.25 + D.ruin * 0.2).lerp(new THREE.Color(0x6b625a), D.ruin * 0.4);
          const mat = new THREE.MeshLambertMaterial({ color: facade, emissive: c.accent.clone().lerp(new THREE.Color(0xffd69a), 0.4), emissiveMap: windows, emissiveIntensity: 0.15 + D.night * 1.25 });
          scatter(group, new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0), mat, towers.length, (i) => ({ ...towers[i], sx: towers[i].fp, sy: towers[i].h, sz: towers[i].fp * 0.9 }));
          if (neon && D.night > 0.35) {
            const tops = towers.filter((t) => !t.rx && r() < 0.65);
            const hues = [c.glow, new THREE.Color(0x40e0ff), new THREE.Color(0xff4fd8)];
            if (tops.length) scatter(group, new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff }), tops.length, (i) => ({
              x: tops[i].x, y: tops[i].top + 0.05, z: tops[i].z, ry: tops[i].ry, rz: tops[i].rz, sx: tops[i].fp * 1.04, sy: 0.12, sz: tops[i].fp * 0.94, color: hues[i % 3].clone(),
            }));
          }
        }
        const stone = c.light.clone().lerp(c.mid, 0.4).lerp(new THREE.Color(0x8d857a), D.ruin * 0.4);
        if (columns.length) {
          const girth = 0.35 + D.scale * 0.5;
          scatter(group, new THREE.CylinderGeometry(0.85, 1, 1, 16).translate(0, 0.5, 0), new THREE.MeshLambertMaterial({ color: stone }), columns.length, (i) => ({ ...columns[i], sx: girth * (1 + r() * 0.6), sz: girth * (1 + r() * 0.6), sy: columns[i].h * 0.8 }));
        }
        if (slabs.length) {
          scatter(group, new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0), new THREE.MeshLambertMaterial({ color: c.deep.clone().multiplyScalar(0.55), emissive: c.glow.clone().multiplyScalar(0.05) }), slabs.length, (i) => ({ ...slabs[i], sx: slabs[i].fp * 0.3, sy: slabs[i].h * 1.6 + 6, sz: slabs[i].fp * 1.4 }));
        }
      }

      // ----- floating islands: weightless rooms lift pieces of land into the air -----
      if (lift > 0) {
        const islandCount = Math.round(3 + lift * 9);
        const under = track(new THREE.ConeGeometry(1, 1.8, 9).rotateX(Math.PI).translate(0, -0.9, 0));
        const lid = track(new THREE.CylinderGeometry(1, 1, 0.25, 9).translate(0, 0.12, 0));
        const underMat = track(new THREE.MeshLambertMaterial({ color: rock.clone().multiplyScalar(0.8), flatShading: true }));
        const lidMat = track(new THREE.MeshLambertMaterial({ color: low.clone().lerp(green, 0.3) }));
        const trunkGeo = track(new THREE.CylinderGeometry(0.1, 0.16, 1, 6).translate(0, 0.5, 0));
        const blobGeo = track(new THREE.IcosahedronGeometry(1, 1));
        const towerGeo = track(new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0));
        const leafMat = track(new THREE.MeshLambertMaterial({ color: grassCol.clone().multiplyScalar(0.78) }));
        const towerMat = track(new THREE.MeshLambertMaterial({ color: c.light.clone().lerp(c.mid, 0.4), emissive: c.glow.clone().multiplyScalar(0.12 + D.night * 0.3) }));
        for (let i = 0; i < islandCount; i++) {
          const s = 1.2 + r() * 2.4 + D.scale * 2;
          const island = new THREE.Group();
          const base = { x: ring ? side(r, 16, 40) : side(r, 6.5, 34), y: 4 + r() * (6 + 12 * lift), z: zc + (r() - 0.5) * len };
          const body = new THREE.Mesh(under, underMat); body.scale.set(s, s * (0.8 + r() * 0.8), s);
          const top = new THREE.Mesh(lid, lidMat); top.scale.set(s, 1, s);
          island.add(body, top);
          if (F > 0.15 && r() < 0.8) {
            const trunk = new THREE.Mesh(trunkGeo, underMat); trunk.scale.set(1, s * 1.2, 1);
            island.add(trunk);
            for (let k = 0; k < 3; k++) {
              const blob = new THREE.Mesh(blobGeo, leafMat);
              blob.position.set((r() - 0.5) * s * 0.6, s * 1.2 + r() * 0.6, (r() - 0.5) * s * 0.6);
              blob.scale.setScalar(s * (0.35 + r() * 0.2));
              island.add(blob);
            }
          } else if (B > 0.3) {
            const tower = new THREE.Mesh(towerGeo, towerMat); tower.scale.set(s * 0.5, s * (1.5 + r() * 2), s * 0.5);
            island.add(tower);
          }
          island.position.set(base.x, base.y, base.z);
          group.add(island);
          const bob = 0.3 + r() * 0.5, spin = (r() - 0.5) * 0.05, p = r() * 6.28;
          animators.push((t) => { island.position.y = base.y + Math.sin(t * bob + p) * 0.5; island.rotation.y = t * spin; });
        }
      }

      // ----- framing the path: colonnades and arches (ordered), lamps or floating lanterns (night) -----
      if (!ring) {
        const colonnade = D.order * (0.35 + (1 - D.growth) * 0.65);
        if (colonnade > 0.32) {
          const step = 6.4 + (1 - D.order) * 5;
          const colH = 5 + D.scale * 6;
          const arched = D.dream > 0.5 || D.era < 0.35;
          const stoneMat = track(new THREE.MeshLambertMaterial({ color: c.light.clone().lerp(c.mid, 0.35), emissive: c.glow.clone().multiplyScalar(D.dream * 0.25) }));
          const column = track(new THREE.CylinderGeometry(0.32, 0.4, 1, 20).translate(0, 0.5, 0));
          const arch = track(new THREE.TorusGeometry(6.4, 0.18, 10, 56, Math.PI));
          const beam = track(new THREE.BoxGeometry(0.8, 0.5, 1));
          for (let z = z0 - 3; z > z0 - len; z -= step) {
            let whole = true;
            for (const x of [-6.4, 6.4]) {
              if (r() < D.ruin * 0.45) { whole = false; continue; }
              const broken = r() < D.ruin * 0.8 ? 0.3 + r() * 0.5 : 1;
              if (broken < 1) whole = false;
              const col = new THREE.Mesh(column, stoneMat);
              col.position.set(x, -0.1, z);
              col.scale.set(1, colH * broken, 1);
              group.add(col);
              if (!arched && broken === 1) {
                const top = new THREE.Mesh(beam, stoneMat);
                top.position.set(x, colH + 0.2, z - step / 2);
                top.scale.z = step;
                group.add(top);
              }
            }
            if (arched && whole) {
              const a = new THREE.Mesh(arch, stoneMat);
              a.position.set(0, colH - 0.1, z);
              group.add(a);
            }
          }
        }
        const lanterns = D.night > 0.38 && (D.float > 0.55 || D.dream > 0.6);
        const lampAmount = D.night * clamp01(1.25 - D.growth);
        if (lanterns) {
          const mat = track(new THREE.SpriteMaterial({ map: glowTexture, color: c.glow.clone().lerp(new THREE.Color(0xffb55e), 0.6), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
          for (let i = 0; i < 14 + 30 * D.night; i++) {
            const s = new THREE.Sprite(mat);
            const base = { x: side(r, 1.2, 16), y: 0.6 + r() * 6, z: zc + (r() - 0.5) * len };
            s.scale.setScalar(0.7 + r() * 0.8);
            group.add(s);
            const speed = 0.15 + r() * 0.3, p = r() * 6.28;
            animators.push((t) => { s.position.set(base.x + Math.sin(t * 0.3 + p) * 0.4, base.y + ((t * speed + p) % 5), base.z); });
          }
        } else if (lampAmount > 0.25) {
          const lampMat = track(new THREE.SpriteMaterial({ map: glowTexture, color: c.glow.clone().lerp(new THREE.Color(0xffc88a), 0.5), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
          const poleMat = track(new THREE.MeshLambertMaterial({ color: 0x111111 }));
          const pole = track(new THREE.CylinderGeometry(0.05, 0.06, 4.2, 6).translate(0, 2.1, 0));
          const every = 7 + (1 - lampAmount) * 8;
          for (let z = z0 - 3; z > z0 - len; z -= every) {
            for (const x of [-5.6, 5.6]) {
              if (r() < D.ruin * 0.4) continue;
              const p = new THREE.Mesh(pole, poleMat);
              p.position.set(x, 0, z);
              const glow = new THREE.Sprite(lampMat);
              glow.position.set(x, 4.3, z);
              glow.scale.setScalar(1.8);
              group.add(p, glow);
            }
          }
        }
      }

      // ----- the sky: sun or moon, stars, extra worlds for dreamlike nights, aurora for cold ones, clouds -----
      // The sky ignores fog, so each room's sky lives in its own layer that fades in only inside that room.
      const sky = new THREE.Group();
      group.add(sky);
      skies.push({ group: sky, zc, half: len / 2 + 6 });
      const starAmount = smooth(0.4, 0.9, D.night);
      if (starAmount > 0) {
        const count = Math.round(4200 * starAmount);
        const pos = new Float32Array(count * 3), cols = new Float32Array(count * 3), col = new THREE.Color();
        for (let i = 0; i < count; i++) {
          const a = r() * Math.PI * 2, el = Math.acos(r() * 1.6 - 0.6), rad = 90 + r() * 120;
          pos.set([Math.cos(a) * Math.sin(el) * rad, Math.cos(el) * rad + 10, zc + Math.sin(a) * Math.sin(el) * rad], i * 3);
          col.copy(c.light).lerp(c.glow, r() * 0.6).toArray(cols, i * 3);
        }
        const geo = track(new THREE.BufferGeometry());
        geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        geo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
        const stars = new THREE.Points(geo, track(new THREE.PointsMaterial({ size: 1.8, sizeAttenuation: false, vertexColors: true, fog: false, transparent: true, opacity: 0.35 + 0.6 * starAmount })));
        sky.add(stars);
        animators.push((t) => { stars.rotation.y = t * 0.004; });
      }
      {
        const night = D.night > 0.55;
        const size = 7 + D.scale * 10 + (night ? D.dream * 10 : 0);
        const color = night ? c.light.clone().lerp(c.glow, 0.3) : c.light.clone().lerp(new THREE.Color(D.warmth > 0.5 ? 0xffc27a : 0xf4f1e6), 0.6);
        const disk = new THREE.Mesh(track(new THREE.CircleGeometry(size, 48)), track(new THREE.MeshBasicMaterial({ color, fog: false })));
        disk.position.set(side(r, 12, 40), night ? 30 + r() * 20 : 8 + (1 - D.warmth) * 18, z0 - len - 120);
        const halo = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: glowTexture, color: c.glow.clone().lerp(color, 0.5), transparent: true, opacity: night ? 0.35 : 0.7, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })));
        halo.position.copy(disk.position);
        halo.scale.setScalar(size * (night ? 4 : 7));
        sky.add(disk, halo);
      }
      const worlds = Math.round(clamp01(D.dream * 1.3 - 0.35) * clamp01(D.night * 1.5 - 0.2) * 3) + (D.era > 0.7 && D.night > 0.5 ? 1 : 0);
      for (let i = 0; i < worlds; i++) {
        const radius = 6 + r() * 14;
        const planet = new THREE.Mesh(track(new THREE.SphereGeometry(radius, 48, 24)), track(new THREE.MeshLambertMaterial({ color: (i % 2 ? c.accent : c.glow).clone(), emissive: c.deep.clone().lerp(c.glow, 0.25), fog: false })));
        planet.position.set(side(r, 35, 70), 18 + r() * 30, z0 - len * (0.3 + r() * 0.7) - 60);
        if (D.era > 0.5 || r() < 0.4) {
          const band = new THREE.Mesh(track(new THREE.RingGeometry(radius * 1.4, radius * 2.1, 64)), track(new THREE.MeshBasicMaterial({ color: c.light, transparent: true, opacity: 0.35, side: THREE.DoubleSide, fog: false })));
          band.rotation.x = Math.PI / 2.4;
          planet.add(band);
        }
        sky.add(planet);
        animators.push((t) => { planet.rotation.y = t * 0.03; });
      }
      const aurora = clamp01((0.5 - D.warmth) * 2.4) * smooth(0.45, 0.8, D.night);
      if (aurora > 0.05) {
        for (let i = 0; i < 2; i++) {
          const mat = track(new THREE.ShaderMaterial({
            uniforms: {
              uTime: timeUniform, uOpacity: { value: 0.55 * aurora },
              uA: { value: new THREE.Color(0x3dffb0).lerp(c.glow, 0.3) }, uB: { value: new THREE.Color(0x8a5cff).lerp(c.accent, 0.3) },
            },
            vertexShader: AURORA_VERTEX, fragmentShader: AURORA_FRAGMENT,
            transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false,
          }));
          const curtain = new THREE.Mesh(track(new THREE.PlaneGeometry(260, 42, 96, 1)), mat);
          curtain.position.set((i - 0.5) * 40, 46 + i * 8, z0 - len / 2 - 90 - i * 30);
          sky.add(curtain);
        }
      }
      const cloudAmount = clamp01(D.wet * 1.2 - 0.2) * (1 - starAmount * 0.5) + D.storm * 0.3;
      if (cloudAmount > 0.1) {
        const mat = track(new THREE.SpriteMaterial({ map: glowTexture, color: c.light.clone().lerp(c.fog, 0.6), transparent: true, opacity: (0.12 + 0.18 * D.storm) * (0.35 + 0.65 * (1 - D.night)), depthWrite: false, fog: false }));
        const drift = 0.2 + D.storm * 2;
        for (let i = 0; i < Math.round(cloudAmount * 16); i++) {
          const s = new THREE.Sprite(mat);
          const base = { x: (r() - 0.5) * 160, y: 22 + r() * 22, z: zc - r() * 90 - 20 };
          s.scale.set(30 + r() * 40, 9 + r() * 10, 1);
          sky.add(s);
          animators.push((t) => { s.position.set(((base.x + t * drift + 80) % 160 + 160) % 160 - 80, base.y, base.z); });
        }
      }

      // ----- weather: rain (wet), snow (cold), blown dust (dry, turbulent), embers (hot), fireflies, rising motes -----
      const rain = clamp01((D.wet - 0.55) * 2.6) * (0.35 + 0.65 * D.storm);
      if (rain > 0.05) weather({ group, r, zc, len, count: 2600 * rain, color: c.light, velocity: [-2 * D.storm, -18, 0], streak: 0.55, opacity: 0.35 });
      const snowfall = clamp01((0.42 - D.warmth) * 3) * (0.4 + 0.6 * D.wet);
      if (snowfall > 0.05) weather({ group, r, zc, len, count: 2000 * snowfall, color: new THREE.Color(0xf4f7fb), velocity: [0.4 * D.storm, -1.2, 0], wobble: 0.6, size: 0.13 });
      const dust = clamp01((0.45 - D.wet) * 2.5) * clamp01(D.storm * 1.6 - 0.2);
      if (dust > 0.05) weather({ group, r, zc, len, count: 1200 * dust, color: sand.clone().lerp(c.light, 0.3), velocity: [3 + 6 * D.storm, 0.1, 0], wobble: 0.4, size: 0.09, opacity: 0.5 });
      const embers = clamp01((D.warmth - 0.55) * 2.5) * clamp01((D.storm - 0.4) * 2.5 + D.ruin * 0.5);
      if (embers > 0.05) weather({ group, r, zc, len, count: 500 * embers, color: new THREE.Color(0xff8a3d).lerp(c.glow, 0.3), velocity: [0.3, 1.4, 0], wobble: 0.5, size: 0.12 });
      const fireflies = D.night * D.growth * D.warmth;
      if (fireflies > 0.12) weather({ group, r, zc, len, count: 600 * fireflies, color: new THREE.Color(0xfff0a0).lerp(c.glow, 0.3), velocity: [0, 0.05, 0], wobble: 1, size: 0.16, box: [24, 5] });
      if (lift > 0.1) weather({ group, r, zc, len, count: 400 * lift, color: c.glow.clone().lerp(c.light, 0.5), velocity: [0, 0.6, 0], wobble: 0.3, size: 0.1, box: [30, 18] });
      return { rain };
    }

    // Falling, drifting or rising particles inside a box around the path, wrapped so they never run out.
    function weather({ group, r, zc, len, count, color, velocity, wobble = 0, streak = 0, size = 0.1, opacity = 0.8, box = [40, 16] }) {
      count = Math.max(1, Math.round(count));
      const [bw, bh] = box;
      const base = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) base.set([(r() - 0.5) * bw * 2, r() * bh, zc + (r() - 0.5) * len], i * 3);
      const per = streak ? 2 : 1;
      const pos = new Float32Array(count * 3 * per);
      const geo = track(new THREE.BufferGeometry());
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const speed = Math.hypot(...velocity) || 1;
      const dir = velocity.map((v) => (v / speed) * streak);
      const object = streak
        ? new THREE.LineSegments(geo, track(new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false })))
        : new THREE.Points(geo, track(new THREE.PointsMaterial({ size, map: glowTexture, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false })));
      object.frustumCulled = false;
      group.add(object);
      const wrap = (v, min, span) => ((((v - min) % span) + span) % span) + min;
      const place = (t) => {
        for (let i = 0; i < count; i++) {
          const k = i * 3;
          const w = wobble ? Math.sin(t * 0.7 + i) * wobble : 0;
          const x = wrap(base[k] + velocity[0] * t + w, -bw, bw * 2);
          const y = wrap(base[k + 1] + velocity[1] * t, 0, bh);
          const z = base[k + 2] + velocity[2] * t + (wobble ? Math.cos(t * 0.5 + i * 1.3) * wobble * 0.6 : 0);
          const o = k * per;
          pos[o] = x; pos[o + 1] = y; pos[o + 2] = z;
          if (streak) { pos[o + 3] = x - dir[0]; pos[o + 4] = y - dir[1]; pos[o + 5] = z - dir[2]; }
        }
        geo.attributes.position.needsUpdate = true;
      };
      place(0);
      animators.push((t) => { if (Math.abs(camera.position.z - zc) < len) place(t); });
    }

    function atmosphere({ group, r, z0, len, c, count }) {
      const pos = new Float32Array(count * 3);
      const speed = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        pos.set([(r() - 0.5) * 34, r() * 9 + 0.3, z0 - r() * len], i * 3);
        speed[i] = 0.2 + r() * 0.8;
      }
      const geo = track(new THREE.BufferGeometry());
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const points = new THREE.Points(geo, track(new THREE.PointsMaterial({ size: 0.11, map: glowTexture, color: c.glow.clone().lerp(c.light, 0.5), transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false })));
      group.add(points);
      const base = pos.slice();
      animators.push((t) => {
        if (Math.abs(camera.position.z - (z0 - len / 2)) > len) return;
        for (let i = 0; i < count; i++) {
          pos[i * 3] = base[i * 3] + Math.sin(t * 0.3 * speed[i] + i) * 0.6;
          pos[i * 3 + 1] = base[i * 3 + 1] + Math.sin(t * 0.5 * speed[i] + i * 1.7) * 0.4;
        }
        geo.attributes.position.needsUpdate = true;
      });
    }

    function hangWork(group, entry, texture, c) {
      const { item } = entry;
      const isText = item.kind !== "photo" || !texture;
      const map = isText ? track(textTexture(item)) : texture;
      const aspect = isText ? 0.75 : (texture.image.width / texture.image.height) || 1;
      const height = entry.ring ? 2.6 : 2.5;
      const width = Math.min(4.2, height * aspect);
      const h = width / aspect;
      const holder = new THREE.Group();
      const frame = new THREE.Mesh(track(new THREE.BoxGeometry(width + 0.14, h + 0.14, 0.07)), track(new THREE.MeshLambertMaterial({ color: c.deep.clone().lerp(new THREE.Color(0x2a2118), 0.6) })));
      frame.position.z = -0.05;
      const art = new THREE.Mesh(track(new THREE.PlaneGeometry(width, h)), track(new THREE.MeshBasicMaterial({ map, toneMapped: false })));
      const halo = new THREE.Mesh(track(new THREE.PlaneGeometry(width * 2.6, h * 2.4)), track(new THREE.MeshBasicMaterial({ map: glowTexture, color: c.glow, transparent: true, opacity: 0.32, blending: THREE.AdditiveBlending, depthWrite: false })));
      halo.position.z = -0.12;
      holder.add(halo, frame, art);
      holder.position.copy(entry.position);
      holder.rotation.y = entry.rotationY;
      group.add(holder);
      art.userData.entry = entry;
      entry.holder = holder;
      entry.halo = halo;
      entry.baseY = entry.position.y;
      entry.phase = Math.random() * 6.28;
      picks.push(art);
    }

    async function buildRooms(roomList) {
      if (world) {
        scene.remove(world);
        roomDisposables.forEach((thing) => thing.dispose?.());
      }
      roomDisposables = [];
      animators = [];
      skies = [];
      picks = [];
      hovered = null;
      world = new THREE.Group();
      scene.add(world);
      roomEnvs = [];
      const starts = [];
      const entries = [];
      let z = 0;
      let ringCenter = 0;
      for (const [index, room] of roomList.entries()) {
        const c = paletteFor(room);
        const r = seeded(hashString(room.title) + index * 7919);
        const group = new THREE.Group();
        world.add(group);
        starts.push(z);
        const len = room.ring ? 34 : Math.max(32, room.works.length * SPACING + 16);
        const z0 = -z;
        const D = room.design;
        const { rain } = generate({ group, r, z0, len, D, c, ring: room.ring });
        roomEnvs.push({
          start: z, len, c,
          density: (room.ring ? 0.016 : 0.012) + D.wet * 0.012 + D.storm * 0.006 + (1 - D.scale) * 0.005
            + clamp01((0.85 - room.coherence) / 0.35) * 0.012,
          ripple: 0.22 + D.storm * 0.9 + D.wet * 0.25 + rain * 0.4,
          exposure: 1.1 - D.night * 0.22,
          hemi: 0.85 + (1 - D.night) * 0.85,
          sun: 0.45 + (1 - D.night) * 1.25,
        });
        atmosphere({ group, r, z0, len, c, count: Math.round((260 + room.works.length * 50) * (1 - rain * 0.8)) });

        const hero = room.works.find((item) => item.kind === "photo" && colorCache.has(item.id));
        const heroTexture = hero ? await loadTexture(hero) : null;

        if (room.ring) {
          ringCenter = z + 16;
          const center = new THREE.Vector3(0, 0, -ringCenter);
          const radius = 7.5;
          const panorama = new THREE.Mesh(track(new THREE.CylinderGeometry(radius + 6, radius + 6, 11, 128, 1, true)), paintPlane(heroTexture, c, 1, 1, [3, 1]));
          panorama.material.side = THREE.BackSide;
          panorama.position.set(0, 5, -ringCenter);
          group.add(panorama);
          room.paint = panorama.material;
          room.works.forEach((item, i) => {
            const angle = (i / room.works.length) * Math.PI * 2;
            const position = new THREE.Vector3(Math.sin(angle) * radius, 1.9, -ringCenter - Math.cos(angle) * radius);
            const entry = { item, room: index, ring: true, orbit: i / room.works.length, distance: ringCenter, position, rotationY: Math.atan2(center.x - position.x, center.z - position.z) };
            entries.push(entry);
          });
        } else {
          const backdrop = new THREE.Mesh(track(new THREE.PlaneGeometry(64, 30)), paintPlane(heroTexture, c, 1, 1));
          backdrop.position.set(0, 13, z0 - len - 8);
          group.add(backdrop);
          room.paint = backdrop.material;
          room.backdropDistance = z + len + 8;
          room.works.forEach((item, i) => {
            const sideSign = i % 2 === 0 ? -1 : 1;
            const distance = z + 10 + i * SPACING;
            const position = new THREE.Vector3(sideSign * 3.1, 1.95, -distance);
            entries.push({ item, room: index, ring: false, orbit: 0, distance, position, rotationY: -sideSign * 0.42 });
          });
        }
        for (const entry of entries.filter((e) => e.room === index)) {
          hangWork(group, entry, await loadTexture(entry.item), c);
        }
        z += len;
      }
      layout = { starts, ringCenter, total: ringCenter + (reducedMotion.matches ? 0 : ORBIT_UNITS), works: entries };
      journey.style.height = `calc(${Math.round((layout.total + 6) * UNIT_PX)}px + 100dvh)`;
      makeWater(z + 40);
      if (line) {
        line.querySelectorAll("i").forEach((tick) => tick.remove());
        starts.forEach((s) => { const tick = node("i"); tick.style.left = `${((s + 6) / (layout.total + 6)) * 100}%`; line.append(tick); });
      }
      lastRoom = -1;
      cam.distance = -6;
      resize();
      ensureRunning();
    }

    function envAt(distance) {
      const envs = roomEnvs;
      let i = 0;
      for (let k = 0; k < envs.length; k++) if (distance >= envs[k].start) i = k;
      const nextEnv = envs[i + 1];
      const blend = nextEnv ? Math.max(0, Math.min(1, (distance - (nextEnv.start - 10)) / 14)) : 0;
      const a = envs[i];
      const k = blend * blend * (3 - 2 * blend);
      return { a, b: nextEnv || a, k, index: k > 0.5 && nextEnv ? i + 1 : i };
    }

    const tmpA = new THREE.Color(), tmpB = new THREE.Color();
    function applyEnv(distance) {
      if (!roomEnvs.length) return 0;
      const { a, b, k, index } = envAt(distance);
      const mix = (key) => tmpA.copy(a.c[key]).lerp(tmpB.copy(b.c[key]), k);
      scene.fog.color.copy(mix("fog"));
      scene.fog.density = a.density + (b.density - a.density) * k;
      skyUniforms.uHorizon.value.copy(scene.fog.color);
      skyUniforms.uTop.value.copy(mix("top"));
      hemi.color.copy(mix("light"));
      hemi.groundColor.copy(mix("deep"));
      sun.color.copy(mix("light")).lerp(mix("glow"), 0.3);
      renderer.toneMappingExposure = a.exposure + (b.exposure - a.exposure) * k;
      hemi.intensity = a.hemi + (b.hemi - a.hemi) * k;
      sun.intensity = a.sun + (b.sun - a.sun) * k;
      if (water) {
        const u = water.material.uniforms;
        u.uFogColor.value.copy(scene.fog.color);
        u.uFogDensity.value = scene.fog.density;
        u.uTint.value.copy(mix("tint"));
        u.uRipple.value = a.ripple + (b.ripple - a.ripple) * k;
      }
      return index;
    }

    function frame() {
      running = false;
      if (!visible || !roomEnvs.length) return;
      running = true;
      requestAnimationFrame(frame);
      const dt = Math.min(0.25, clock.getDelta());
      const t = clock.elapsedTime;
      timeUniform.value = t;
      const p = scrollProgress();
      const target = p * (layout.total + 6) - 6;
      const ease = reducedMotion.matches ? 1 : 1 - Math.exp(-dt * 3.2);
      cam.distance += (target - cam.distance) * ease;
      if (Math.abs(target - cam.distance) > 30) cam.distance = target - Math.sign(target - cam.distance) * 30;
      const walk = Math.min(cam.distance, layout.ringCenter);
      const orbit = layout.total > layout.ringCenter ? Math.max(0, (cam.distance - layout.ringCenter) / (layout.total - layout.ringCenter)) : 0;
      look.yaw += (look.tYaw - look.yaw) * 0.06;
      look.pitch += (look.tPitch - look.pitch) * 0.06;
      const meander = Math.sin(walk * 0.06) * 0.5 * Math.max(0, Math.min(1, (layout.ringCenter - walk) / 10));
      camera.position.set(meander, 1.75 + (reducedMotion.matches ? 0 : Math.sin(t * 0.6) * 0.03), -walk);
      camera.rotation.set(look.pitch, look.yaw - orbit * Math.PI * 2 + meander * 0.05, 0);
      sky.position.copy(camera.position);

      const index = applyEnv(walk);
      roomIndex = index;
      if (index !== lastRoom && walk > -1) {
        lastRoom = index;
        if (rooms[index]) announceRoom(rooms[index], index);
      }
      updateControls(cam.distance, p);

      for (const room of rooms) {
        if (!room.paint) continue;
        room.paint.uniforms.uTime.value = t;
        room.paint.uniforms.uFogDensity.value = scene.fog.density;
        room.paint.uniforms.uFogColor.value.copy(scene.fog.color);
        if (room.backdropDistance) room.paint.uniforms.uOpacity.value = Math.max(0, Math.min(1, (room.backdropDistance - walk - 6) / 16));
      }
      if (water) water.material.uniforms.uTime.value = t;
      for (const layer of skies) {
        const fade = 1 - smooth(layer.half - 6, layer.half + 8, Math.abs(camera.position.z - layer.zc));
        if (layer.fade === fade) continue;
        layer.fade = fade;
        layer.group.visible = fade > 0.001;
        layer.group.traverse((o) => {
          const m = o.material;
          if (!m) return;
          if (m.userData.base === undefined) { m.userData.base = m.uniforms?.uOpacity ? m.uniforms.uOpacity.value : m.opacity; m.transparent = true; m.needsUpdate = true; }
          if (m.uniforms?.uOpacity) m.uniforms.uOpacity.value = m.userData.base * fade;
          else m.opacity = m.userData.base * fade;
        });
      }
      if (!reducedMotion.matches) animators.forEach((fn) => fn(t));

      raycaster.setFromCamera(pointer, camera);
      const hit = finePointer ? raycaster.intersectObjects(picks, false)[0] : null;
      const nextHover = hit && hit.distance < 22 ? hit.object.userData.entry : null;
      if (nextHover !== hovered) {
        hovered = nextHover;
        stage.classList.toggle("is-pointing", Boolean(hovered));
      }
      let focus = hovered;
      let best = Infinity;
      for (const entry of layout.works) {
        const holder = entry.holder;
        if (!holder) continue;
        if (entry.ring) {
          const facing = Math.sin(holder.rotation.y) * (camera.position.x - holder.position.x) + Math.cos(holder.rotation.y) * (camera.position.z - holder.position.z);
          holder.visible = facing > 0;
        }
        const lift = reducedMotion.matches ? 0 : Math.sin(t * 0.8 + entry.phase) * 0.05;
        holder.position.y = entry.baseY + lift;
        const target = entry === hovered ? 1.06 : 1;
        holder.scale.setScalar(holder.scale.x + (target - holder.scale.x) * 0.12);
        entry.halo.material.opacity += ((entry === hovered ? 0.7 : 0.3) - entry.halo.material.opacity) * 0.1;
        if (!hovered) {
          const d = camera.position.distanceTo(holder.position);
          const ahead = new THREE.Vector3().subVectors(holder.position, camera.position).normalize().dot(camera.getWorldDirection(tmpV));
          if (ahead > 0.6 && d < 14 && d < best) { best = d; focus = entry; }
        }
      }
      setCaption(focus);
      renderer.render(scene, camera);
    }
    const tmpV = new THREE.Vector3();

    function ensureRunning() {
      if (!running && visible) { clock.getDelta(); requestAnimationFrame(frame); running = true; }
    }

    function resize() {
      const w = stage.clientWidth, h = stage.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.fov = w < 640 ? 70 : 58;
      camera.updateProjectionMatrix();
      if (water) {
        const scale = finePointer ? 0.5 : 0.35;
        water.getRenderTarget().setSize(Math.max(256, w * scale * renderer.getPixelRatio()), Math.max(256, h * scale * renderer.getPixelRatio()));
      }
    }

    new ResizeObserver(() => { resize(); ensureRunning(); }).observe(stage);
    new IntersectionObserver(([entry]) => { visible = entry.isIntersecting && !document.hidden; ensureRunning(); }).observe(stage);
    document.addEventListener("visibilitychange", () => { visible = !document.hidden && stage.offsetParent !== null; ensureRunning(); });

    stage.addEventListener("pointermove", (event) => {
      const bounds = stage.getBoundingClientRect();
      pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
      if (event.pointerType === "mouse" && !reducedMotion.matches) {
        look.tYaw = -pointer.x * 0.32;
        look.tPitch = pointer.y * 0.09;
      }
    });
    stage.addEventListener("pointerleave", () => { pointer.set(9, 9); look.tYaw = 0; look.tPitch = 0; });
    canvas.addEventListener("click", (event) => {
      const bounds = stage.getBoundingClientRect();
      const tap = new THREE.Vector2(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
      raycaster.setFromCamera(tap, camera);
      const hit = raycaster.intersectObjects(picks, false)[0];
      if (hit && hit.distance < 26) showDetails(hit.object.userData.entry.item);
    });

    return { build: buildRooms };
  }

  start();
}
