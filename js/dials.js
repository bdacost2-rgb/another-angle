// The twelve CLIP design dials (built ↔ overgrown, daylit ↔ nocturnal, still ↔ turbulent…), shared by
// the inspiration feed, the Uncharted map and the gallery. Each dial is a direction in CLIP space between
// two sets of text prompts, made by tools/build_dials.py and saved in data/dials.json with each medium's
// spread across the archive, so a score is in standard deviations ("how much more nocturnal than usual").
import { dot } from "./data.js";

let cached = null;
export function loadDials(base = "data/") {
  if (!cached) {
    cached = fetch(base + "dials.json")
      .then((r) => { if (!r.ok) throw new Error(`dials.json ${r.status}`); return r.json(); })
      .then((data) => data.dials.map((d) => ({ ...d, vec: Float32Array.from(d.vec) })))
      .catch((error) => { console.warn("Design dials unavailable", error); return []; });
  }
  return cached;
}

// One item's position on every dial, in archive standard deviations. Cached per item.
const scores = new WeakMap();
export function itemDials(item, dials) {
  let z = scores.get(item);
  if (z && z.length === dials.length) return z;
  z = new Float32Array(dials.length);
  dials.forEach((dial, k) => {
    const [mean, std] = dial.stats[item.kind] || dial.stats.photo;
    z[k] = (dot(item.vec, dial.vec) - mean) / (std || 1);
  });
  scores.set(item, z);
  return z;
}

// The word for a dial at a value: its high pole when positive, its low pole when negative.
export const dialWord = (dial, value) => (value >= 0 ? dial.high : dial.low);

// The strongest words in a profile (an array of z scores), strongest first.
export function strongestWords(profile, dials, n = 3, floor = 0.25) {
  return [...profile].map((v, k) => ({ k, v })).filter(({ v }) => Math.abs(v) > floor)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, n).map(({ k, v }) => dialWord(dials[k], v));
}
