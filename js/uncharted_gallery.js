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
    // Rooms built from your likes are named from what the AI measured in them: an adjective from the
    // second-strongest dial and a noun from the lead dial (unique to this room), e.g. "Midnight Tides".
    if (room.fromTaste && room.signature.length >= 2) {
      const [noun, adj] = room.signature;
      room.title = `${POLE_ADJ[adj] || adj} ${POLE_NOUN[noun] || noun}`;
    }
  });
}

const POLE_NOUN = {
  overgrown: "Gardens", built: "Streets", nocturnal: "Nocturnes", daylit: "Daylight", crowded: "Crowds", sparse: "Horizons",
  turbulent: "Storms", still: "Stillness", weightless: "Drifts", grounded: "Stones", warm: "Embers", cold: "Frost",
  ruined: "Ruins", pristine: "Clear Light", dreamlike: "Dreams", real: "Records", wet: "Tides", dry: "Dust",
  monumental: "Monuments", intimate: "Interiors", ordered: "Patterns", irregular: "Wanderings", futuristic: "Futures", ancient: "Antiquities",
};
const POLE_ADJ = {
  overgrown: "Overgrown", built: "Built", nocturnal: "Midnight", daylit: "Sunlit", crowded: "Crowded", sparse: "Quiet",
  turbulent: "Restless", still: "Still", weightless: "Floating", grounded: "Heavy", warm: "Golden", cold: "Silver",
  ruined: "Fading", pristine: "Polished", dreamlike: "Dreaming", real: "Everyday", wet: "Misty", dry: "Dusty",
  monumental: "Vast", intimate: "Small", ordered: "Ordered", irregular: "Wandering", futuristic: "Electric", ancient: "Ancient",
};

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
      ? "Scroll · click a work"
      : "Scroll to walk · tap a work";
  }
  setTip();

  function updateIntro() {
    if (model.count >= 8 && model.liked.length >= 4) {
      intro.textContent = `What you love, then what lies just past it. Built from your ${model.count} choices.`;
    } else if (model.count >= 8) {
      intro.textContent = `Built from your ${model.count} choices. Save a few more (♥) to fill it.`;
    } else {
      intro.textContent = "A preview. Pick works in Inspiration to make it yours.";
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
        title: r.theme?.n || "The works you brought", // replaced by a name read from the works (designRooms)
        note: `${works.length === 1 ? "One of your works" : `${works.length} of your works that look and read alike`}, grouped by the AI (CLIP). Its name comes from what the AI measured in these works.${r.neighbors[0] ? ` Closest curated theme: ${r.neighbors[0].name}${r.neighbors[1] ? `, then ${r.neighbors[1].name}` : ""}.` : ""}`,
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
      note: "A last cluster of new works, gathered from the directions your taste has opened.",
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
      engine = createEngine(post);
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

  // ---------- the voyage: each room is a galaxy, and you glide forward through space from work to work ----------
  // The camera only ever moves forward. Each work floats straight ahead, glows as you arrive, and dissolves
  // into light as you pass through it to the next. Between rooms you drift through open space to the next galaxy.
  function createEngine(post = null) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, finePointer ? 1.75 : 1.35));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x03040a);
    scene.fog = new THREE.FogExp2(0x03040a, 0.012);
    const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 900);
    camera.rotation.order = "YXZ";

    // ----- the cinematic finish: glow on bright light, depth of field (desktop), grain and vignette -----
    let composer = null, bloom = null, bokeh = null, finish = null, focusDistance = 4;
    if (post) {
      composer = new post.EffectComposer(renderer);
      composer.addPass(new post.RenderPass(scene, camera));
      if (finePointer) {
        bokeh = new post.BokehPass(scene, camera, { focus: 4, aperture: 0.0012, maxblur: 0.005 });
        composer.addPass(bokeh);
      }
      bloom = new post.UnrealBloomPass(new THREE.Vector2(256, 256), 0.5, 0.6, 0.72);
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

    const glowTexture = canvasTexture(THREE, 128, 128, (ctx, w) => {
      const g = ctx.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
      g.addColorStop(0, "rgba(255,255,255,1)");
      g.addColorStop(0.3, "rgba(255,255,255,.4)");
      g.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, w);
    });
    // soft, uneven cloud for nebulae
    const nebulaTexture = canvasTexture(THREE, 256, 256, (ctx, w) => {
      for (let i = 0; i < 26; i++) {
        const x = w / 2 + (Math.random() - 0.5) * w * 0.5, y = w / 2 + (Math.random() - 0.5) * w * 0.5, r = w * (0.12 + Math.random() * 0.28);
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0, "rgba(255,255,255,.16)");
        g.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, w);
      }
    });

    // the deep sky: a far field of stars around the whole voyage, with a few distant galaxies
    let sky = null;
    function buildSky(length) {
      if (sky) { scene.remove(sky); sky.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); }); }
      sky = new THREE.Group();
      const count = 9000, pos = new Float32Array(count * 3), col = new Float32Array(count * 3), c = new THREE.Color();
      for (let i = 0; i < count; i++) {
        const a = Math.random() * Math.PI * 2, e = Math.acos(Math.random() * 2 - 1), r = 260 + Math.random() * 260;
        pos.set([Math.cos(a) * Math.sin(e) * r, Math.cos(e) * r * 0.7, -length / 2 + Math.sin(a) * Math.sin(e) * (r + length / 2)], i * 3);
        c.setHSL(0.55 + Math.random() * 0.2, 0.3, 0.75 + Math.random() * 0.25).toArray(col, i * 3);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      sky.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.4, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.85, fog: false, depthWrite: false })));
      // closer drifting stars give a sense of speed as you travel
      const near = 2600, npos = new Float32Array(near * 3);
      for (let i = 0; i < near; i++) npos.set([(Math.random() - 0.5) * 90, (Math.random() - 0.5) * 60, 20 - Math.random() * (length + 60)], i * 3);
      const ngeo = new THREE.BufferGeometry();
      ngeo.setAttribute("position", new THREE.BufferAttribute(npos, 3));
      sky.add(new THREE.Points(ngeo, new THREE.PointsMaterial({ size: 0.07, map: glowTexture, color: 0xcfdcff, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false })));
      scene.add(sky);
    }

    let world = null;
    let roomEnvs = [];
    let picks = [];
    let animators = [];
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

    // A spiral galaxy of glowing stars in the room's colors, lying just below your path so you fly over its arms.
    function galaxy(group, zc, size, c, seed) {
      const rand = seeded(seed);
      const arms = 2 + Math.floor(rand() * 3), twist = 2.4 + rand() * 2.2, count = finePointer ? 7000 : 4000;
      const pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
      const core = new THREE.Color(0xfff2dc), armColor = c.glow.clone().lerp(new THREE.Color(0xffffff), 0.15), edge = c.accent.clone().lerp(c.mid, 0.4), tmp = new THREE.Color();
      for (let i = 0; i < count; i++) {
        const r = Math.pow(rand(), 0.7) * size;
        const arm = Math.floor(rand() * arms);
        const a = (arm / arms) * Math.PI * 2 + (r / size) * twist + (rand() - 0.5) * (0.5 + (1 - r / size) * 0.6);
        const spread = 0.6 + (r / size) * 2.2;
        pos.set([Math.cos(a) * r + (rand() - 0.5) * spread, (rand() - 0.5) * (1.2 + (1 - r / size) * 2.4), Math.sin(a) * r + (rand() - 0.5) * spread], i * 3);
        const k = r / size;
        tmp.copy(core).lerp(armColor, Math.min(1, k * 2.2)).lerp(edge, Math.max(0, k - 0.55) * 1.6).toArray(col, i * 3);
      }
      const geo = track(new THREE.BufferGeometry());
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      const disk = new THREE.Points(geo, track(new THREE.PointsMaterial({ size: 0.2, map: glowTexture, vertexColors: true, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false })));
      const holder = new THREE.Group();
      holder.add(disk);
      // a bright core and a haze of nebula clouds in the room's colors
      const coreGlow = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: glowTexture, color: core.clone().lerp(c.glow, 0.25), transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false })));
      coreGlow.scale.setScalar(size * 0.28);
      holder.add(coreGlow);
      for (let i = 0; i < 9; i++) {
        const cloud = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: nebulaTexture, color: (i % 2 ? c.glow : c.accent).clone().lerp(c.mid, 0.3), transparent: true, opacity: 0.14, blending: THREE.AdditiveBlending, depthWrite: false })));
        const a = rand() * Math.PI * 2, r = size * (0.2 + rand() * 0.7);
        cloud.position.set(Math.cos(a) * r, (rand() - 0.5) * 6, Math.sin(a) * r);
        cloud.scale.setScalar(size * (0.35 + rand() * 0.4));
        holder.add(cloud);
      }
      // the core sits off to one side and below, so you sail over its arms rather than through its glare
      holder.position.set((rand() < 0.5 ? -1 : 1) * (size * 0.35), -8, zc);
      holder.rotation.set(0.32 + rand() * 0.14, 0, (rand() - 0.5) * 0.25); // tilted toward you, so the spiral shows as you approach
      group.add(holder);
      const spin = 0.008 + rand() * 0.01;
      animators.push((t) => { disk.rotation.y = t * spin; });
    }

    // Works float straight ahead in a slim glowing frame, and dissolve into light as you pass through them.
    function hangWork(group, entry, texture, c) {
      const { item } = entry;
      const isText = item.kind !== "photo" || !texture;
      const map = isText ? track(textTexture(item)) : texture;
      const aspect = isText ? 0.75 : (texture.image.width / texture.image.height) || 1;
      const height = 2.1;
      const width = Math.min(3.2, height * aspect);
      const h = width / aspect;
      const holder = new THREE.Group();
      const frame = new THREE.Mesh(track(new THREE.PlaneGeometry(width + 0.08, h + 0.08)), track(new THREE.MeshBasicMaterial({ color: c.glow.clone().lerp(new THREE.Color(0xffffff), 0.4).multiplyScalar(0.8), transparent: true })));
      frame.position.z = -0.01;
      const art = new THREE.Mesh(track(new THREE.PlaneGeometry(width, h)), track(new THREE.MeshBasicMaterial({ map, toneMapped: false, transparent: true })));
      const halo = new THREE.Mesh(track(new THREE.PlaneGeometry(width * 1.9, h * 1.8)), track(new THREE.MeshBasicMaterial({ map: glowTexture, color: c.glow.clone().lerp(new THREE.Color(0xffffff), 0.3), transparent: true, opacity: 0.2, blending: THREE.AdditiveBlending, depthWrite: false })));
      halo.position.z = -0.05;
      holder.add(halo, frame, art);
      holder.position.copy(entry.position);
      group.add(holder);
      art.userData.entry = entry;
      Object.assign(entry, { holder, halo, art, frameMesh: frame, phase: Math.random() * 6.28 });
      picks.push(art);
    }

    const GAP = 8; // distance back between works along the spiral
    const SPIRAL_R = 4.6, SPIRAL_STEP = 1.0; // each work sits a turn further around the spiral
    const VIEW = 4.2; // how far in front of a work you stop to look at it
    async function buildRooms(roomList) {
      if (world) {
        scene.remove(world);
        roomDisposables.forEach((thing) => thing.dispose?.());
      }
      roomDisposables = [];
      animators = [];
      picks = [];
      hovered = null;
      world = new THREE.Group();
      scene.add(world);
      roomEnvs = [];
      path = [];
      const entries = [];
      let z = 0;
      for (const [index, room] of roomList.entries()) {
        const c = paletteFor(room);
        const group = new THREE.Group();
        world.add(group);
        const n = room.works.length;
        const approach = index === 0 ? 10 : 16; // open space before you reach the galaxy's first work
        const zFirst = -(z + approach + VIEW);
        const length = approach + VIEW + (n - 1) * GAP + 8;
        galaxy(group, -(z + length / 2 + 2), Math.max(26, length * 0.6), c, hashString(room.title) + index * 977);
        // the galaxy comes into view: a stop as you arrive, then each work in turn
        path.push({ room: index, dwell: 2.6, z: -(z + approach * 0.35) });
        const roomStart = path.length - 1;
        // The works wind around a spiral that recedes into the distance, like a path of stars: from each one
        // you can see the next, a little further around and back, and you glide beside it rather than through.
        const spin = index % 2 ? -1 : 1, phase = index * 0.7;
        const spiralAt = (u) => new THREE.Vector3(SPIRAL_R * Math.sin(spin * u * SPIRAL_STEP + phase), 1.8 + SPIRAL_R * 0.42 * Math.cos(spin * u * SPIRAL_STEP + phase), zFirst - u * GAP);
        path[path.length - 1].lookAt = spiralAt(0);
        room.works.forEach((item, i) => {
          const position = spiralAt(i);
          const entry = { item, room: index, position, distance: 0 };
          entries.push(entry);
          path.push({ room: index, dwell: 2.8, z: position.z + VIEW, x: position.x, y: position.y + 0.05, lookAt: position.clone(), entry });
        });
        // a faint trail of tiny stars along the spiral, linking the works like a constellation
        {
          const steps = Math.max(1, n - 1) * 60, pos = new Float32Array(steps * 3);
          for (let k = 0; k < steps; k++) {
            const pt = spiralAt((k / steps) * Math.max(1, n - 1));
            pos.set([pt.x + (Math.random() - 0.5) * 0.25, pt.y + (Math.random() - 0.5) * 0.25, pt.z + (Math.random() - 0.5) * 0.25], k * 3);
          }
          const geo = track(new THREE.BufferGeometry());
          geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
          group.add(new THREE.Points(geo, track(new THREE.PointsMaterial({ size: 0.06, map: glowTexture, color: c.glow.clone().lerp(new THREE.Color(0xffffff), 0.5), transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }))));
        }
        for (const entry of entries.filter((e) => e.room === index)) hangWork(group, entry, await loadTexture(entry.item), c);
        roomEnvs.push({ c, roomStart, fog: new THREE.Color(0x03040a).lerp(c.deep, 0.1), bloom: 0.45 + (room.design?.dream ?? 0.5) * 0.2 });
        z += length;
      }
      buildSky(z);
      // scroll distance: each stop takes a little scrolling, the glides between take more
      let unit = 0;
      path.forEach((stop, i) => {
        stop.x = stop.x ?? 0;
        stop.y = stop.y ?? 1.8;
        stop.lookAt = stop.lookAt ?? new THREE.Vector3(stop.x, stop.y, stop.z - 10);
        if (i > 0) unit += Math.max(3, Math.abs(path[i - 1].z - stop.z) * 0.55);
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

    // Where the camera is for a scroll distance: resting at a stop, or gliding forward to the next.
    const tmpPos = new THREE.Vector3(), tmpLook = new THREE.Vector3();
    function pathAt(u) {
      if (!path.length) return null;
      const rest = (stop) => ({ pos: tmpPos.set(stop.x, stop.y, stop.z), look: tmpLook.copy(stop.lookAt), stop, room: stop.room, next: stop.room, k: 0 });
      if (u <= path[0].at) {
        const first = path[0]; // before the first stop, drift in from a little further back
        return { pos: tmpPos.set(first.x, first.y, first.z + Math.min(8, (first.at - u) * 1.2)), look: tmpLook.copy(first.lookAt), stop: first, room: first.room, next: first.room, k: 0 };
      }
      for (let i = 0; i < path.length; i++) {
        const stop = path[i], next = path[i + 1];
        if (u <= stop.at + stop.dwell || !next) return rest(stop);
        if (u < next.at) {
          const t = (u - stop.at - stop.dwell) / (next.at - stop.at - stop.dwell);
          const e = t * t * (3 - 2 * t);
          // glide around the spiral (a curve that swings out past the work you're leaving), turning toward the next
          const swing = Math.sin(e * Math.PI) * 1.4;
          const dx = next.x - stop.x, dy = next.y - stop.y, len = Math.hypot(dx, dy) || 1;
          return {
            pos: tmpPos.set(stop.x + dx * e - (dy / len) * swing * 0.3 + (dx / len) * swing * 0.6, stop.y + dy * e + (dy / len) * swing * 0.3, stop.z + (next.z - stop.z) * e),
            look: tmpLook.copy(stop.lookAt).lerp(next.lookAt, Math.min(1, e * 1.3)),
            stop: e > 0.5 ? next : stop, room: stop.room, next: next.room, k: e,
          };
        }
      }
      return rest(path[path.length - 1]);
    }

    function applyEnv(at) {
      const a = roomEnvs[at.room], b = roomEnvs[at.next] || a, k = at.room === at.next ? 0 : at.k;
      scene.fog.color.copy(a.fog).lerp(b.fog, k);
      scene.background.copy(scene.fog.color);
      if (bloom) bloom.strength = a.bloom + (b.bloom - a.bloom) * k;
      return k > 0.5 ? at.next : at.room;
    }

    const tmpDir = new THREE.Vector3();
    function frame() {
      running = false;
      if (!visible || !path.length || stage.offsetParent === null) return;
      running = true;
      requestAnimationFrame(frame);
      const dt = Math.min(0.25, clock.getDelta());
      const t = clock.elapsedTime;
      const p = scrollProgress();
      const target = p * (layout.total + 6) - 6;
      const ease = reducedMotion.matches ? 1 : 1 - Math.exp(-dt * 2.4);
      cam.distance += (target - cam.distance) * ease;
      if (Math.abs(target - cam.distance) > 20) cam.distance = target - Math.sign(target - cam.distance) * 20;
      const at = pathAt(cam.distance);
      look.yaw += (look.tYaw - look.yaw) * 0.05;
      look.pitch += (look.tPitch - look.pitch) * 0.05;
      camera.position.copy(at.pos);
      if (!reducedMotion.matches) { camera.position.y += Math.sin(t * 0.5) * 0.04; camera.position.x += Math.sin(t * 0.37) * 0.03; }
      const dir = tmpDir.copy(at.look).sub(camera.position);
      camera.rotation.set(Math.atan2(dir.y, Math.hypot(dir.x, dir.z)) + look.pitch * 0.6, Math.atan2(-dir.x, -dir.z) + look.yaw * 0.5, 0);

      const index = applyEnv(at);
      roomIndex = index;
      if (index !== lastRoom && cam.distance > -1) {
        lastRoom = index;
        if (rooms[index]) announceRoom(rooms[index], index);
      }
      updateControls(cam.distance, p);
      if (!reducedMotion.matches) animators.forEach((fn) => fn(t));

      raycaster.setFromCamera(pointer, camera);
      const hit = finePointer ? raycaster.intersectObjects(picks, false)[0] : null;
      const nextHover = hit && hit.distance < 14 ? hit.object.userData.entry : null;
      if (nextHover !== hovered) {
        hovered = nextHover;
        stage.classList.toggle("is-pointing", Boolean(hovered));
      }
      const current = at.stop.entry || null;
      for (const entry of layout.works) {
        if (!entry.holder) continue;
        // ahead: visible; close: it glows; passing through: it dissolves into light
        const ahead = camera.position.z - entry.position.z;
        const fade = Math.max(0, Math.min(1, (ahead + 1.2) / 2.4)); // gently gone once it's behind you
        entry.art.material.opacity = fade;
        entry.frameMesh.material.opacity = fade;
        entry.holder.visible = ahead > -1.2;
        entry.holder.lookAt(camera.position.x, entry.holder.position.y, camera.position.z); // works turn softly to face you
        const lit = entry === hovered ? 0.45 : entry === current ? 0.28 : 0.1;
        entry.halo.material.opacity += (lit * (0.4 + 0.6 * fade) + (1 - fade) * 0.35 * (ahead > 0.2 ? 1 : 0) - entry.halo.material.opacity) * 0.15;
        const s = entry === hovered ? 1.04 : 1;
        entry.holder.scale.setScalar(entry.holder.scale.x + (s - entry.holder.scale.x) * 0.2);
        if (!reducedMotion.matches) entry.holder.position.y = entry.position.y + Math.sin(t * 0.6 + entry.phase) * 0.05;
      }
      setCaption(hovered || current);
      if (composer) {
        if (bokeh?.enabled) {
          const focus = hovered || current;
          const want = focus?.holder ? camera.position.distanceTo(focus.holder.position) : VIEW;
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
      camera.fov = w < 640 ? 68 : 55;
      camera.updateProjectionMatrix();
    }

    new ResizeObserver(() => { resize(); ensureRunning(); }).observe(stage);
    new IntersectionObserver(([entry]) => { visible = entry.isIntersecting && !document.hidden; ensureRunning(); }).observe(stage);
    document.addEventListener("visibilitychange", () => { visible = !document.hidden && stage.offsetParent !== null; ensureRunning(); });

    stage.addEventListener("pointermove", (event) => {
      const bounds = stage.getBoundingClientRect();
      pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
      if (event.pointerType === "mouse" && !reducedMotion.matches) {
        look.tYaw = -pointer.x * 0.22;
        look.tPitch = pointer.y * 0.08;
      }
    });
    stage.addEventListener("pointerleave", () => { pointer.set(9, 9); look.tYaw = 0; look.tPitch = 0; });
    canvas.addEventListener("click", (event) => {
      const bounds = stage.getBoundingClientRect();
      const tap = new THREE.Vector2(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
      raycaster.setFromCamera(tap, camera);
      const hit = raycaster.intersectObjects(picks, false)[0];
      if (hit && hit.distance < 16) showDetails(hit.object.userData.entry.item);
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
