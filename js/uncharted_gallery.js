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

// The room's sky as light: a gradient from ground to horizon to zenith with a sun, turned into an
// environment map so physically based surfaces pick up believable reflections and fill light.
const ENV_FRAGMENT = /* glsl */`
uniform vec3 uTop;
uniform vec3 uHorizon;
uniform vec3 uGround;
uniform vec3 uSun;
uniform vec3 uSunDir;
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  vec3 col = mix(uGround, uHorizon, smoothstep(-0.25, 0.02, d.y));
  col = mix(col, uTop, smoothstep(0.02, 0.7, d.y));
  float s = max(dot(d, uSunDir), 0.0);
  col += uSun * (pow(s, 48.0) * 6.0 + pow(s, 6.0) * 0.6);
  gl_FragColor = vec4(col, 1.0);
}`;

// The last step of the cinematic finish: a soft vignette and fine film grain, applied after tone mapping.
const FINISH_SHADER = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uGrain: { value: 0.04 }, uVignette: { value: 0.3 } },
  vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: /* glsl */`
uniform sampler2D tDiffuse;
uniform float uTime;
uniform float uGrain;
uniform float uVignette;
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  vec4 c = texture2D(tDiffuse, vUv);
  float v = 1.0 - uVignette * smoothstep(0.35, 0.95, length((vUv - 0.5) * vec2(1.15, 1.0)) * 1.25);
  float g = (hash(vUv * vec2(1931.0, 1213.0) + fract(uTime * 7.0)) - 0.5) * uGrain;
  gl_FragColor = vec4(c.rgb * v + g, c.a);
}`,
};

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
      intro.textContent = `Built from your ${model.count} choices. Each room of this little house holds works that feel alike, its colors and lamplight drawn from them.`;
    } else if (model.count >= 8) {
      intro.textContent = `Built from your ${model.count} choices. Save a few more works (♥) and they'll become rooms of their own.`;
    } else {
      intro.textContent = "A preview. As you choose works in Inspiration, they'll be hung in the rooms of a little house, grouped by how they feel.";
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
        note: `${works.length === 1 ? "One of your works" : `${works.length} of your works that look and read alike`}, grouped by the AI (CLIP)${near ? `. Also close to ${near}` : ""}. ${r.theme?.desc ? `${r.theme.desc[0].toUpperCase()}${r.theme.desc.slice(1)}.` : ""} The room's colors and lamplight come from how these works feel.`,
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
  let building = false;
  // Switching between photos, writing or both rebuilds the rooms (and the map's constellations).
  window.addEventListener("focus:change", () => { if (engine) build(); });
  window.addEventListener("gallery:enter-room", (event) => {
    const index = event.detail?.index;
    if (!Number.isInteger(index)) return;
    if (layout.starts.length && !building) changeRoom(index, true, true);
    else pendingEntry = index;
  });

  // Choices made elsewhere in Uncharted (saving or dismissing works at checkpoints) change your rooms.
  // They're counted, and when you walk back into the gallery it regroups and says what changed.
  let changes = { saved: 0, dismissed: 0 };
  window.addEventListener("taste:changed", (event) => {
    if (event.detail?.good) changes.saved += 1; else changes.dismissed += 1;
  });
  window.addEventListener("gallery:shown", () => {
    if (engine) requestAnimationFrame(() => engine.wake());
    if (!engine || (!changes.saved && !changes.dismissed)) return;
    const { saved } = changes;
    changes = { saved: 0, dismissed: 0 };
    build().then(() => notify(saved
      ? `${saved} work${saved === 1 ? "" : "s"} you saved at checkpoints joined your gallery. Your rooms regrouped around them.`
      : "Your rooms regrouped around your latest choices."));
  });
  // ---------- full-screen walk ----------
  // Walking (scrolling) fades the header and buttons away; pausing, tapping, keyboard focus or reaching
  // for the top or bottom edge brings them back.
  let immerseTimer = 0;
  const setImmersed = (on) => document.body.classList.toggle("is-immersed", on && document.body.classList.contains("gallery-full"));
  window.addEventListener("scroll", () => {
    if (!document.body.classList.contains("gallery-full") || dialog.open) return;
    const walked = scrollProgress() * (layout.total + 6) - 6;
    if (walked < 0) { setImmersed(false); return; } // not while the opening is on screen
    setImmersed(true);
    clearTimeout(immerseTimer);
    immerseTimer = setTimeout(() => setImmersed(false), 1500);
  }, { passive: true });
  stage.addEventListener("pointermove", (event) => {
    if (event.pointerType !== "mouse") return;
    const bounds = stage.getBoundingClientRect();
    if (event.clientY - bounds.top < 120 || bounds.bottom - event.clientY < 120) { clearTimeout(immerseTimer); setImmersed(false); }
  });
  stage.addEventListener("pointerdown", () => { clearTimeout(immerseTimer); setImmersed(false); });
  document.addEventListener("focusin", (event) => { if (event.target.closest?.(".sitebar, .gallery-hud")) setImmersed(false); });

  const notice = $("#gallery-notice");
  let noticeTimer = 0;
  function notify(text) {
    if (!notice) return;
    notice.textContent = text;
    notice.classList.add("is-on");
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => notice.classList.remove("is-on"), 6500);
  }

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
    const was = roomIndex;
    build().then(() => changeRoom(Math.min(was, rooms.length - 1), false, true));
    setTip();
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  // ---------- scroll and room navigation ----------

  let layout = { starts: [], total: 1, works: [] };

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
    scrollToDistance((layout.starts[target] ?? 0) + 0.5, instant);
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
    showEnd(roomIndex === rooms.length - 1 && p > 0.96);
    tip.classList.toggle("is-done", distance > 4); // the "scroll to walk" hint leaves once you're walking
    const label = `room ${String(roomIndex + 1).padStart(2, "0")} / ${String(rooms.length).padStart(2, "0")}`;
    if (progress.textContent !== label) progress.textContent = label;
    if (roomName && room && roomName.textContent !== room.title) roomName.textContent = room.title;
    previous.disabled = roomIndex === 0;
    next.disabled = roomIndex >= rooms.length - 1;
    const nextLabel = roomIndex === rooms.length - 2 ? "Enter the ring →" : "Next room →";
    if (next.textContent !== nextLabel) next.textContent = nextLabel;
    if (line) line.style.setProperty("--p", String(p));
    const openingOpacity = Math.max(0, 1 - (distance + 6) / 7);
    opening.style.opacity = String(openingOpacity);
    opening.style.visibility = openingOpacity < 0.01 ? "hidden" : "visible"; // never ghost into later rooms
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
      button.addEventListener("focus", () => scrollToDistance(entry.distance));
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
    const top = topNight.lerp(topDay, Math.pow(day, 1.6));
    return {
      fog,
      top,
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
      const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
      // The cinematic finish is optional: if these don't load, the gallery renders plainly.
      let post = null;
      try {
        const [{ EffectComposer }, { RenderPass }, { UnrealBloomPass }, { BokehPass }, { ShaderPass }, { OutputPass }] = await Promise.all([
          import("three/addons/postprocessing/EffectComposer.js"), import("three/addons/postprocessing/RenderPass.js"),
          import("three/addons/postprocessing/UnrealBloomPass.js"), import("three/addons/postprocessing/BokehPass.js"),
          import("three/addons/postprocessing/ShaderPass.js"), import("three/addons/postprocessing/OutputPass.js"),
        ]);
        post = { EffectComposer, RenderPass, UnrealBloomPass, BokehPass, ShaderPass, OutputPass };
      } catch (error) {
        console.warn("Gallery: post-processing unavailable", error);
      }
      engine = createEngine(GLTFLoader, post);
      await loadDials();
    } catch (error) {
      console.warn("Gallery: WebGL unavailable", error);
      stage.classList.add("is-fallback");
      intro.textContent = "Your browser couldn't start the 3-D gallery, so the works are listed below instead.";
      opening.classList.add("is-ready");
      await loadDials();
      rooms = makeRooms();
      layout.works = rooms.flatMap((room) => room.works.map((item) => ({ item, distance: 0 })));
      renderList();
      return;
    }
    build();
  }

  async function build() {
    building = true;
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
    building = false;
    if (pendingEntry != null) { const index = pendingEntry; pendingEntry = null; requestAnimationFrame(() => changeRoom(index, true, true)); }
  }

  // ---------- the cozy home: a few small furnished rooms, lit by lamps, with your works on the walls ----------
  // Furniture is KayKit "Furniture Bits" by Kay Lousberg (CC0), in assets/kaykit. Scrolling moves the camera
  // from spot to spot: a view of each room as you enter, then each work in turn, then through the doorway.
  function createEngine(GLTFLoader, post = null) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, finePointer ? 1.75 : 1.35));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x140f0b);
    scene.fog = new THREE.FogExp2(0x1c140f, 0.02);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
    camera.rotation.order = "YXZ";
    const hemi = new THREE.HemisphereLight(0xffe7cc, 0x5a3b26, 0.9);
    scene.add(hemi);
    // A few warm lamp lights follow you through the house, always sitting in the lamps nearest the camera,
    // so every lamp glows without the cost of lighting them all at once.
    const lampLights = Array.from({ length: finePointer ? 4 : 3 }, () => {
      const light = new THREE.PointLight(0xffc58a, 0, 11, 2);
      scene.add(light);
      return light;
    });

    // ----- the cinematic finish: glow on bright lights, depth of field (desktop), grain and vignette -----
    let composer = null, bloom = null, bokeh = null, finish = null, focusDistance = 4;
    if (post) {
      composer = new post.EffectComposer(renderer);
      composer.addPass(new post.RenderPass(scene, camera));
      if (finePointer) {
        bokeh = new post.BokehPass(scene, camera, { focus: 4, aperture: 0.0016, maxblur: 0.004 });
        composer.addPass(bokeh);
      }
      bloom = new post.UnrealBloomPass(new THREE.Vector2(256, 256), 0.35, 0.5, 0.9);
      composer.addPass(bloom);
      composer.addPass(new post.OutputPass());
      finish = new post.ShaderPass(FINISH_SHADER);
      composer.addPass(finish);
    }
    // If the device struggles, shed the costly parts in order: depth of field, then glow, then resolution.
    let runTime = 0, slowTime = 0;
    const keepQuality = new URLSearchParams(location.search).has("hq"); // for screenshots on slow test machines
    function adapt(dt) {
      runTime += dt;
      if (!composer || runTime < 5 || keepQuality) return;
      slowTime = dt > 1 / 30 ? slowTime + dt : Math.max(0, slowTime - dt * 0.5);
      if (slowTime < 2.5) return;
      slowTime = 0;
      if (bokeh?.enabled) bokeh.enabled = false;
      else if (bloom?.enabled) bloom.enabled = false;
      else if (renderer.getPixelRatio() > 1) { renderer.setPixelRatio(1); resize(); }
    }

    // ----- a warm indoor environment, so wood and fabric pick up soft reflections and fill light -----
    {
      const pmrem = new THREE.PMREMGenerator(renderer);
      const envScene = new THREE.Scene();
      envScene.add(new THREE.Mesh(new THREE.SphereGeometry(10, 32, 16), new THREE.ShaderMaterial({
        uniforms: {
          uTop: { value: new THREE.Color(0xf3e2c8) }, uHorizon: { value: new THREE.Color(0xc98c5a) }, uGround: { value: new THREE.Color(0x3a2516) },
          uSun: { value: new THREE.Color(0xffc285).multiplyScalar(0.6) }, uSunDir: { value: new THREE.Vector3(0.3, 0.6, -0.7).normalize() },
        },
        side: THREE.BackSide, depthWrite: false,
        vertexShader: "varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: ENV_FRAGMENT,
      })));
      scene.environment = pmrem.fromScene(envScene, 0.04).texture;
      scene.environmentIntensity = 0.45;
    }

    const glowTexture = canvasTexture(THREE, 128, 128, (ctx, w) => {
      const g = ctx.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
      g.addColorStop(0, "rgba(255,255,255,1)");
      g.addColorStop(0.35, "rgba(255,255,255,.35)");
      g.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, w);
    });
    // soft contact shadows under furniture: the cheap trick that makes things sit on the floor
    const shadowTexture = canvasTexture(THREE, 128, 128, (ctx, w) => {
      const g = ctx.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
      g.addColorStop(0, "rgba(0,0,0,.75)");
      g.addColorStop(0.6, "rgba(0,0,0,.35)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, w);
    });
    const floorTexture = canvasTexture(THREE, 512, 512, (ctx, w, h) => {
      const rows = 8, rh = h / rows;
      for (let i = 0; i < rows; i++) {
        let x = (i % 2) * -120;
        while (x < w) {
          const len = 170 + Math.random() * 150, tone = Math.random() * 18;
          ctx.fillStyle = `rgb(${176 - tone},${112 - tone},${78 - tone * 0.6})`;
          ctx.fillRect(x, i * rh, len, rh);
          ctx.fillStyle = "rgba(70,38,20,.35)";
          ctx.fillRect(x, i * rh, 2, rh);
          x += len;
        }
        ctx.fillStyle = "rgba(70,38,20,.45)";
        ctx.fillRect(0, i * rh, w, 2);
      }
    });
    floorTexture.wrapS = floorTexture.wrapT = THREE.RepeatWrapping;

    let world = null;
    let roomEnvs = [];
    let picks = [];
    let animators = [];
    let lamps = [];
    let path = [];
    let roomDisposables = [];
    let hovered = null;
    let lastRoom = -1;
    const pointer = new THREE.Vector2(9, 9);
    const raycaster = new THREE.Raycaster();
    const look = { yaw: 0, pitch: 0, tYaw: 0, tPitch: 0 };
    let cam = { distance: -6 };
    const clock = new THREE.Clock();
    let visible = true;
    let running = false;

    function track(...things) { roomDisposables.push(...things); return things[0]; }

    // ----- furniture, loaded once and cloned into each room -----
    const loader = new GLTFLoader();
    const models = new Map();
    function loadModel(name) {
      if (!models.has(name)) {
        models.set(name, new Promise((resolve) => loader.load(`assets/kaykit/${name}.gltf`, (gltf) => {
          gltf.scene.traverse((o) => { if (o.isMesh) o.material.roughness = Math.max(o.material.roughness ?? 0.8, 0.72); });
          resolve(gltf.scene);
        }, undefined, () => resolve(null))));
      }
      return models.get(name);
    }
    function shadow(group, x, z, w, d, strength = 0.4) {
      const m = new THREE.Mesh(track(new THREE.PlaneGeometry(w, d).rotateX(-Math.PI / 2)), track(new THREE.MeshBasicMaterial({ map: shadowTexture, color: 0x000000, transparent: true, opacity: strength, depthWrite: false })));
      m.position.set(x, 0.03, z);
      group.add(m);
    }
    async function put(group, name, x, z, ry = 0, { y = 0, s = 1, shade = null } = {}) {
      const proto = await loadModel(name);
      if (!proto) return null;
      const m = proto.clone();
      m.position.set(x, y, z);
      m.rotation.y = ry;
      m.scale.setScalar(s);
      group.add(m);
      if (shade && y === 0) shadow(group, x, z, shade[0], shade[1]);
      return m;
    }

    // ----- the rooms -----
    const ROOM_HALF = 4.5, WALL_H = 4.2, DOOR_W = 2.2, DOOR_H = 3.2, WAIN = 1.1;
    const SLOT_GAP = 3.4, ART_Y = 2.45;
    function plane(group, w, h, material, x, y, z, ry) {
      const m = new THREE.Mesh(track(new THREE.PlaneGeometry(w, h)), material);
      m.position.set(x, y, z);
      m.rotation.y = ry;
      group.add(m);
      return m;
    }
    function buildShell(group, zFront, zBack, wallColor, first, last) {
      const depth = zFront - zBack, zc = (zFront + zBack) / 2, W = 2 * ROOM_HALF;
      const paint = track(new THREE.MeshStandardMaterial({ color: wallColor, roughness: 0.95, envMapIntensity: 0.4 }));
      const lower = track(new THREE.MeshStandardMaterial({ color: wallColor.clone().multiplyScalar(0.72).lerp(new THREE.Color(0x6b4630), 0.35), roughness: 0.85, envMapIntensity: 0.4 }));
      const wood = track(new THREE.MeshStandardMaterial({ color: 0x7a4c32, roughness: 0.6 }));
      const ceilingMat = track(new THREE.MeshStandardMaterial({ color: 0xf1e6d4, roughness: 0.95, emissive: 0x3a2a1c, emissiveIntensity: 0.6, envMapIntensity: 0.2 }));
      const tex = floorTexture.clone(); tex.needsUpdate = true; tex.repeat.set(W / 3.2, depth / 3.2); track(tex);
      const floor = new THREE.Mesh(track(new THREE.PlaneGeometry(W, depth).rotateX(-Math.PI / 2)), track(new THREE.MeshStandardMaterial({ map: tex, roughness: 0.55, envMapIntensity: 0.6 })));
      floor.position.set(0, 0, zc);
      const ceiling = new THREE.Mesh(track(new THREE.PlaneGeometry(W, depth).rotateX(Math.PI / 2)), ceilingMat);
      ceiling.position.set(0, WALL_H, zc);
      group.add(floor, ceiling);
      // side walls: painted above, a darker panelled band below
      for (const s of [-1, 1]) {
        plane(group, depth, WALL_H - WAIN, paint, s * ROOM_HALF, WAIN + (WALL_H - WAIN) / 2, zc, s < 0 ? Math.PI / 2 : -Math.PI / 2);
        plane(group, depth, WAIN, lower, s * ROOM_HALF, WAIN / 2, zc, s < 0 ? Math.PI / 2 : -Math.PI / 2);
      }
      // end walls: the back wall has a doorway to the next room (unless it's the last room)
      const endWall = (z, ry, door) => {
        if (!door) {
          plane(group, W, WALL_H - WAIN, paint, 0, WAIN + (WALL_H - WAIN) / 2, z, ry);
          plane(group, W, WAIN, lower, 0, WAIN / 2, z, ry);
          return;
        }
        const sideW = ROOM_HALF - DOOR_W / 2;
        for (const s of [-1, 1]) {
          plane(group, sideW, WALL_H - WAIN, paint, s * (DOOR_W / 2 + sideW / 2), WAIN + (WALL_H - WAIN) / 2, z, ry);
          plane(group, sideW, WAIN, lower, s * (DOOR_W / 2 + sideW / 2), WAIN / 2, z, ry);
        }
        plane(group, DOOR_W, WALL_H - DOOR_H, paint, 0, DOOR_H + (WALL_H - DOOR_H) / 2, z, ry);
      };
      endWall(zBack, 0, !last);
      if (first) endWall(zFront, Math.PI, false);
      // mouldings: baseboards, a chair rail, crown, and a cased doorway
      const trims = [];
      for (const s of [-1, 1]) {
        trims.push({ x: s * (ROOM_HALF - 0.04), y: 0.1, z: zc, sx: 0.08, sy: 0.2, sz: depth });
        trims.push({ x: s * (ROOM_HALF - 0.04), y: WAIN, z: zc, sx: 0.09, sy: 0.08, sz: depth });
        trims.push({ x: s * (ROOM_HALF - 0.07), y: WALL_H - 0.1, z: zc, sx: 0.15, sy: 0.2, sz: depth });
      }
      for (const z of [zBack + 0.04, first ? zFront - 0.04 : null].filter((v) => v !== null)) {
        trims.push({ x: 0, y: 0.1, z, sx: W, sy: 0.2, sz: 0.08 }, { x: 0, y: WALL_H - 0.1, z, sx: W, sy: 0.2, sz: 0.15 });
      }
      if (!last) {
        for (const s of [-1, 1]) trims.push({ x: s * (DOOR_W / 2 + 0.08), y: DOOR_H / 2, z: zBack, sx: 0.16, sy: DOOR_H, sz: 0.22 });
        trims.push({ x: 0, y: DOOR_H + 0.08, z: zBack, sx: DOOR_W + 0.48, sy: 0.16, sz: 0.22 });
      }
      const box = track(new THREE.BoxGeometry(1, 1, 1));
      const mesh = new THREE.InstancedMesh(box, wood, trims.length);
      const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), sc = new THREE.Vector3();
      trims.forEach((tr, i) => { p.set(tr.x, tr.y, tr.z); sc.set(tr.sx, tr.sy, tr.sz); m4.compose(p, q, sc); mesh.setMatrixAt(i, m4); });
      group.add(mesh);
    }

    // Each room is a different room of the house; the theme shows in its colors, its lamp light and a few props.
    const ROOM_TYPES = ["living", "study", "bedroom", "nook"];
    async function furnish(group, type, zFront, zBack, room) {
      const zc = (zFront + zBack) / 2, back = zBack, front = zFront, H = ROOM_HALF;
      const theme = room.theme?.n?.toLowerCase() || "";
      const lampsHere = [];
      const lamp = (x, y, z) => lampsHere.push(new THREE.Vector3(x, y, z));
      const rug = ["rug_oval_A", "rug_oval_B", "rug_rectangle_A", "rug_rectangle_B", "rug_rectangle_stripes_A", "rug_rectangle_stripes_B"][Math.abs(hashString(room.title)) % 6];
      if (type === "living") {
        await put(group, rug, 0, zc, 0, { s: 1.15 });
        await put(group, "couch_pillows", -H + 0.95, zc, Math.PI / 2, { shade: [2, 3.4] });
        await put(group, "table_low", 0.1, zc, Math.PI / 2, { shade: [1.8, 2.6] });
        await put(group, "armchair", H - 1.1, zc + 0.6, -Math.PI / 2 - 0.3, { shade: [1.9, 1.9] });
        // lamps stand in the front corners, lighting the room without standing in front of any work
        await put(group, "lamp_standing", -H + 0.7, front - 0.8, 0, { shade: [1, 1] }); lamp(-H + 0.7, 2.2, front - 0.8);
        await put(group, "cabinet_small", H - 0.7, front - 0.75, Math.PI, { shade: [1.2, 1.2] });
        await put(group, "lamp_table", H - 0.7, front - 0.75, 0, { y: 1.0 }); lamp(H - 0.7, 1.75, front - 0.75);
        await put(group, "book_set", 0.1, zc + 0.3, 0.4, { y: 0.5, s: 0.7 });
      } else if (type === "study") {
        await put(group, rug, 0, zc, Math.PI / 2, { s: 1.15 });
        await put(group, "table_medium_long", 0, zc, Math.PI / 2, { shade: [2.4, 3.4] });
        await put(group, "chair_A_wood", -1.15, zc - 0.5, Math.PI / 2, { shade: [1, 1] });
        await put(group, "chair_B_wood", 1.15, zc + 0.5, -Math.PI / 2, { shade: [1, 1] });
        await put(group, "lamp_table", 0, zc - 0.9, 0, { y: 1.0 }); lamp(0, 1.75, zc - 0.9);
        await put(group, "book_set", 0.1, zc + 0.6, 1.2, { y: 1.25 });
        await put(group, "cabinet_medium", -H + 0.6, back + 1.4, Math.PI / 2, { shade: [1.3, 2.3] });
        await put(group, "lamp_standing", H - 0.7, front - 0.8, 0, { shade: [1, 1] }); lamp(H - 0.7, 2.2, front - 0.8);
      } else if (type === "bedroom") {
        await put(group, "bed_double_A", -H + 1.55, zc, Math.PI / 2, { shade: [3.4, 3.4] });
        await put(group, rug, 0.9, zc, Math.PI / 2, { s: 0.9 });
        await put(group, "cabinet_small", -H + 0.6, zc - 2.1, Math.PI / 2, { s: 0.8, shade: [1, 1] });
        await put(group, "lamp_table", -H + 0.6, zc - 2.1, 0, { y: 0.8, s: 0.85 }); lamp(-H + 0.6, 1.5, zc - 2.1);
        await put(group, "armchair_pillows", H - 1.1, zc + 0.8, -Math.PI / 2, { shade: [1.9, 1.9] });
        await put(group, "pillow_A", H - 1.2, zc + 0.8, 0.3, { y: 0.62, s: 0.8 });
        await put(group, "lamp_standing", H - 0.7, front - 0.8, 0, { shade: [1, 1] }); lamp(H - 0.7, 2.2, front - 0.8);
      } else {
        await put(group, rug, 0, zc, 0, { s: 1.1 });
        await put(group, "armchair_pillows", -H + 1.15, zc - 0.4, Math.PI / 2 + 0.25, { shade: [1.9, 1.9] });
        await put(group, "lamp_standing", -H + 0.7, zc - 1.7, 0, { shade: [1, 1] }); lamp(-H + 0.7, 2.2, zc - 1.7);
        await put(group, "table_small", -H + 1.2, zc + 1.1, 0, { s: 0.75, shade: [1, 1] });
        await put(group, "book_single", -H + 1.2, zc + 1.1, 0.5, { y: 0.9, s: 0.8 });
        await put(group, "cabinet_small_decorated", H - 0.7, front - 0.8, Math.PI, { shade: [1.3, 1.3] });
        await put(group, "chair_C", H - 1.2, zc, -Math.PI / 2, { shade: [1, 1] });
      }
      // a few props that hint at the theme
      if (/desert|americana|dystopia|surreal/.test(theme)) {
        await put(group, "cactus_medium_A", 1.75, back + 0.55, 0, { shade: [0.9, 0.9] });
        await put(group, "cactus_small_B", -1.6, back + 0.4, 0, { shade: [0.6, 0.6] });
      } else if (/academia|romantic|fairy|gothic|myth|documentary|harlem/.test(theme)) {
        await put(group, "book_set", 1.75, back + 0.45, 0, { y: 0.25, shade: [0.8, 0.8] });
        await put(group, "book_set", 1.75, back + 0.45, 0.2, { y: 0.75, s: 0.9 });
      } else {
        await put(group, "cactus_small_A", 1.7, back + 0.4, 0, { shade: [0.6, 0.6] });
      }
      return lampsHere;
    }

    // Works hang on the walls in simple wooden frames, glowing softly when you point at them.
    function hangWork(group, entry, texture, c) {
      const { item } = entry;
      const isText = item.kind !== "photo" || !texture;
      const map = isText ? track(textTexture(item)) : texture;
      const aspect = isText ? 0.75 : (texture.image.width / texture.image.height) || 1;
      const height = 1.45;
      const width = Math.min(2.1, height * aspect);
      const h = width / aspect;
      const holder = new THREE.Group();
      const frame = new THREE.Mesh(track(new THREE.BoxGeometry(width + 0.16, h + 0.16, 0.08)), track(new THREE.MeshStandardMaterial({ color: 0x4a2f20, roughness: 0.55 })));
      frame.position.z = -0.04;
      const art = new THREE.Mesh(track(new THREE.PlaneGeometry(width, h)), track(new THREE.MeshBasicMaterial({ map, toneMapped: false, color: new THREE.Color(1, 1, 1).multiplyScalar(composer ? 1.18 : 1) })));
      art.position.z = 0.005;
      const halo = new THREE.Mesh(track(new THREE.PlaneGeometry(width * 2, h * 1.9)), track(new THREE.MeshBasicMaterial({ map: glowTexture, color: new THREE.Color(0xffd9a8).lerp(c.glow, 0.3), transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false })));
      halo.position.z = -0.1;
      holder.add(halo, frame, art);
      holder.position.copy(entry.position);
      holder.rotation.y = entry.rotationY;
      group.add(holder);
      art.userData.entry = entry;
      entry.holder = holder;
      entry.halo = halo;
      picks.push(art);
    }

    // Dust drifting in the lamplight.
    function dust(group, zFront, zBack) {
      const count = 180, pos = new Float32Array(count * 3), base = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) base.set([(Math.random() - 0.5) * 8, 0.3 + Math.random() * 3.6, zBack + Math.random() * (zFront - zBack)], i * 3);
      pos.set(base);
      const geo = track(new THREE.BufferGeometry());
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const points = new THREE.Points(geo, track(new THREE.PointsMaterial({ size: 0.035, map: glowTexture, color: 0xffe2bd, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false })));
      group.add(points);
      animators.push((t) => {
        if (camera.position.z > zFront + 4 || camera.position.z < zBack - 4) return;
        for (let i = 0; i < count; i++) {
          pos[i * 3] = base[i * 3] + Math.sin(t * 0.13 + i) * 0.25;
          pos[i * 3 + 1] = base[i * 3 + 1] + Math.sin(t * 0.09 + i * 1.7) * 0.2;
        }
        geo.attributes.position.needsUpdate = true;
      });
    }

    // every piece of furniture the rooms use, fetched together up front (about 1 MB, shared by all rooms)
    const FURNITURE = [
      "rug_oval_A", "rug_oval_B", "rug_rectangle_A", "rug_rectangle_B", "rug_rectangle_stripes_A", "rug_rectangle_stripes_B",
      "couch_pillows", "table_low", "armchair", "armchair_pillows", "lamp_standing", "lamp_table", "cabinet_small", "cabinet_small_decorated",
      "cabinet_medium", "table_medium_long", "table_small", "chair_A_wood", "chair_B_wood", "chair_C", "bed_double_A", "pillow_A",
      "book_set", "book_single", "cactus_medium_A", "cactus_small_A", "cactus_small_B",
    ];

    async function buildRooms(roomList) {
      await Promise.all(FURNITURE.map(loadModel));
      if (world) {
        scene.remove(world);
        roomDisposables.forEach((thing) => thing.dispose?.());
      }
      roomDisposables = [];
      animators = [];
      picks = [];
      lamps = [];
      hovered = null;
      world = new THREE.Group();
      scene.add(world);
      roomEnvs = [];
      path = [];
      const entries = [];
      let z = 0;
      for (const [index, room] of roomList.entries()) {
        const c = paletteFor(room);
        const D = room.design;
        const group = new THREE.Group();
        world.add(group);
        const n = room.works.length;
        const far = Math.min(2, n);
        const sideSlots = Math.ceil((n - far) / 2);
        const depth = Math.max(9.5, sideSlots * SLOT_GAP + 4.6);
        const zFront = -z, zBack = zFront - depth;
        const first = index === 0, last = index === roomList.length - 1;
        // muted, homely wall colors drawn from the room's own palette
        const wallColor = new THREE.Color(0xe6d6be).lerp(c.mid, 0.28).lerp(c.light, 0.1);
        buildShell(group, zFront, zBack, wallColor, first, last);
        const roomLamps = await furnish(group, ROOM_TYPES[index % ROOM_TYPES.length], zFront, zBack, room);
        const warmth = D.warmth ?? 0.5;
        const lampColor = new THREE.Color(0xffb36b).lerp(new THREE.Color(0xffe2c2), 1 - warmth);
        roomLamps.forEach((position) => {
          lamps.push({ position, color: lampColor, room: index });
          const bulb = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: glowTexture, color: lampColor, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false })));
          bulb.position.copy(position);
          bulb.scale.setScalar(0.9);
          group.add(bulb);
        });
        dust(group, zFront, zBack);

        // where each work hangs: a zigzag down the side walls, then two on the far wall beside the doorway
        const slots = [];
        for (let i = 0; i < sideSlots; i++) {
          const sz = zFront - 2.7 - i * SLOT_GAP;
          slots.push({ x: -ROOM_HALF + 0.06, z: sz, ry: Math.PI / 2, nx: 1, nz: 0 });
          slots.push({ x: ROOM_HALF - 0.06, z: sz, ry: -Math.PI / 2, nx: -1, nz: 0 });
        }
        for (const s of [-1, 1].slice(0, far)) slots.push({ x: s * 2.95, z: zBack + 0.06, ry: 0, nx: 0, nz: 1 });
        // the room's opening view, then each work in turn
        path.push({ room: index, dwell: 3.2, pos: new THREE.Vector3(0, 2.45, zFront - (first ? 0.7 : -0.6)), look: new THREE.Vector3(0, 1.5, zBack + 1) });
        const roomStart = path.length - 1;
        room.works.forEach((item, i) => {
          const slot = slots[i];
          if (!slot) return;
          const position = new THREE.Vector3(slot.x, ART_Y, slot.z);
          const entry = { item, room: index, position, rotationY: slot.ry, distance: 0 };
          entries.push(entry);
          const stand = new THREE.Vector3(slot.x + slot.nx * 3.3, 2.1, slot.z + slot.nz * 3.3);
          path.push({ room: index, dwell: 2.4, pos: stand, look: position.clone(), entry });
        });
        if (!last) path.push({ room: index, dwell: 0, pos: new THREE.Vector3(0, 2.15, zBack + 0.6), look: new THREE.Vector3(0, 1.9, zBack - 6) });
        for (const entry of entries.filter((e) => e.room === index)) hangWork(group, entry, await loadTexture(entry.item), c);
        roomEnvs.push({
          c, D, roomStart,
          hemi: 0.5 + (1 - (D.night ?? 0.5)) * 0.45,
          lamp: 9 + (D.night ?? 0.5) * 9,
          exposure: 1.0,
          fog: new THREE.Color(0x1c140f).lerp(c.deep, 0.15),
          tint: new THREE.Color(0xffe7cc).lerp(c.light, 0.15),
        });
        z += depth;
      }
      // turn the path into scroll distance: each stop takes a little scrolling, the moves between take more
      let unit = 0;
      path.forEach((stop) => { // each stop's heading as angles, so moves between stops turn smoothly
        const d = stop.look.clone().sub(stop.pos);
        stop.yaw = Math.atan2(-d.x, -d.z);
        stop.pitch = Math.atan2(d.y, Math.hypot(d.x, d.z));
      });
      path.forEach((stop, i) => {
        if (i > 0) {
          const prev = path[i - 1];
          const turn = prev.look.clone().sub(prev.pos).normalize().angleTo(stop.look.clone().sub(stop.pos).normalize());
          unit += Math.max(2.4, prev.pos.distanceTo(stop.pos) * 0.75 + turn * 2.2);
        }
        stop.at = unit;
        unit += stop.dwell;
        if (stop.entry) stop.entry.distance = stop.at + stop.dwell / 2;
      });
      const starts = roomEnvs.map((env) => path[env.roomStart].at);
      layout = { starts, total: unit, works: entries };
      journey.style.height = `calc(${Math.round((layout.total + 6) * UNIT_PX)}px + 100dvh)`;
      if (line) {
        line.querySelectorAll("i").forEach((tick) => tick.remove());
        starts.forEach((s) => { const tick = node("i"); tick.style.left = `${((s + 6) / (layout.total + 6)) * 100}%`; line.append(tick); });
      }
      lastRoom = -1;
      cam.distance = -6;
      resize();
      ensureRunning();
    }

    // Where the camera is for a scroll distance: resting at a stop, or easing from one stop to the next.
    const tmpPos = new THREE.Vector3();
    const turn = (a, b, e) => { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * e; };
    function pathAt(u) {
      if (!path.length) return null;
      const rest = (stop) => ({ pos: tmpPos.copy(stop.pos), yaw: stop.yaw, pitch: stop.pitch, stop, room: stop.room, next: stop.room, k: 0 });
      if (u <= path[0].at) return rest(path[0]);
      for (let i = 0; i < path.length; i++) {
        const stop = path[i], next = path[i + 1];
        if (u <= stop.at + stop.dwell || !next) return rest(stop);
        if (u < next.at) {
          const t = (u - stop.at - stop.dwell) / (next.at - stop.at - stop.dwell);
          const e = t * t * (3 - 2 * t);
          return { pos: tmpPos.copy(stop.pos).lerp(next.pos, e), yaw: turn(stop.yaw, next.yaw, e), pitch: stop.pitch + (next.pitch - stop.pitch) * e, stop: e > 0.5 ? next : stop, room: stop.room, next: next.room, k: e };
        }
      }
      return rest(path[path.length - 1]);
    }

    const tmpColor = new THREE.Color();
    function applyEnv(at) {
      const a = roomEnvs[at.room], b = roomEnvs[at.next] || a, k = at.room === at.next ? 0 : at.k;
      scene.fog.color.copy(a.fog).lerp(b.fog, k);
      hemi.color.copy(a.tint).lerp(b.tint, k);
      hemi.intensity = a.hemi + (b.hemi - a.hemi) * k;
      renderer.toneMappingExposure = a.exposure + (b.exposure - a.exposure) * k;
      return k > 0.5 ? at.next : at.room;
    }

    // light the lamps nearest the camera, with a faint candle-like flicker
    function lightLamps(t, roomIndex) {
      const env = roomEnvs[roomIndex];
      const nearest = lamps.map((lamp) => [lamp.position.distanceToSquared(camera.position), lamp]).sort((a, b) => a[0] - b[0]);
      lampLights.forEach((light, i) => {
        const lamp = nearest[i]?.[1];
        if (!lamp) { light.intensity = 0; return; }
        light.position.copy(lamp.position);
        light.color.copy(lamp.color);
        light.intensity = (env?.lamp ?? 12) * (1 + (reducedMotion.matches ? 0 : Math.sin(t * 2.3 + i * 1.7) * 0.03));
      });
    }

    function frame() {
      running = false;
      if (!visible || !path.length || stage.offsetParent === null) return;
      running = true;
      requestAnimationFrame(frame);
      const dt = Math.min(0.25, clock.getDelta());
      const t = clock.elapsedTime;
      const p = scrollProgress();
      const target = p * (layout.total + 6) - 6;
      const ease = reducedMotion.matches ? 1 : 1 - Math.exp(-dt * 2.6);
      cam.distance += (target - cam.distance) * ease;
      if (Math.abs(target - cam.distance) > 20) cam.distance = target - Math.sign(target - cam.distance) * 20;
      const at = pathAt(Math.max(0, cam.distance));
      look.yaw += (look.tYaw - look.yaw) * 0.06;
      look.pitch += (look.tPitch - look.pitch) * 0.06;
      camera.position.copy(at.pos);
      if (!reducedMotion.matches) camera.position.y += Math.sin(t * 0.6) * 0.015;
      camera.rotation.set(at.pitch + look.pitch * 0.6, at.yaw + look.yaw * 0.5, 0);

      const index = applyEnv(at);
      roomIndex = index;
      if (index !== lastRoom && cam.distance > -1) {
        lastRoom = index;
        if (rooms[index]) announceRoom(rooms[index], index);
      }
      updateControls(cam.distance, p);
      lightLamps(t, index);
      if (!reducedMotion.matches) animators.forEach((fn) => fn(t));

      raycaster.setFromCamera(pointer, camera);
      const hit = finePointer ? raycaster.intersectObjects(picks, false)[0] : null;
      const nextHover = hit && hit.distance < 12 ? hit.object.userData.entry : null;
      if (nextHover !== hovered) {
        hovered = nextHover;
        stage.classList.toggle("is-pointing", Boolean(hovered));
      }
      const focus = hovered || at.stop.entry || null;
      for (const entry of layout.works) {
        if (!entry.holder) continue;
        const on = entry === hovered || entry === at.stop.entry;
        entry.halo.material.opacity += ((entry === hovered ? 0.55 : on ? 0.28 : 0.1) - entry.halo.material.opacity) * 0.1;
        const s = entry === hovered ? 1.04 : 1;
        entry.holder.scale.setScalar(entry.holder.scale.x + (s - entry.holder.scale.x) * 0.12);
      }
      setCaption(focus);
      if (composer) {
        if (bokeh?.enabled) {
          const want = focus?.holder ? camera.position.distanceTo(focus.holder.position) : 5;
          focusDistance += (want - focusDistance) * 0.1;
          bokeh.uniforms.focus.value = focusDistance;
        }
        finish.uniforms.uTime.value = reducedMotion.matches ? 0 : t;
        composer.render(dt);
      } else {
        renderer.render(scene, camera);
      }
      adapt(dt);
    }

    function ensureRunning() {
      if (!running && visible) { clock.getDelta(); requestAnimationFrame(frame); running = true; }
    }

    function resize() {
      const w = stage.clientWidth, h = stage.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      if (composer) { composer.setPixelRatio(renderer.getPixelRatio()); composer.setSize(w, h); }
      camera.aspect = w / h;
      camera.fov = w < 640 ? 64 : 50;
      camera.updateProjectionMatrix();
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
      if (hit && hit.distance < 14) showDetails(hit.object.userData.entry.item);
    });

    function wake() {
      visible = !document.hidden;
      resize();
      ensureRunning();
    }

    return { build: buildRooms, wake };
  }

  start();
}
