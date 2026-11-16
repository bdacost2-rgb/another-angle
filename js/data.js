// Loads the archive: items.json (photos + passages with credits) and emb.bin
// (one CLIP embedding per item, float16). Both are made by tools/build_data.py.

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export async function loadArchive(base = "data/") {
  const meta = await (await fetch(base + "items.json")).json();
  // Embeddings live in emb.bin, or inline as base64 (emb_b64) where .bin files can't be served.
  const buf = meta.emb_b64
    ? Uint8Array.from(atob(meta.emb_b64), (c) => c.charCodeAt(0)).buffer
    : await (await fetch(base + "emb.bin")).arrayBuffer();
  const dim = meta.dim;
  const half = new Uint16Array(buf);
  const items = meta.items;
  if (half.length !== items.length * dim) throw new Error("emb.bin does not match items.json");

  // Raw CLIP vectors (kept for searching with your own photo or words).
  const raw = items.map((_, i) => {
    const v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) v[d] = halfToFloat(half[i * dim + d]);
    return normalize(v);
  });

  // CLIP puts all photos in one corner and all text in another (the "modality gap").
  // Subtracting each kind's average lets a photo and a passage match on meaning.
  const means = {};
  for (const kind of ["photo", "text"]) {
    const m = new Float32Array(dim);
    let n = 0;
    items.forEach((it, i) => {
      if (it.kind === kind) { n++; for (let d = 0; d < dim; d++) m[d] += raw[i][d]; }
    });
    if (n) for (let d = 0; d < dim; d++) m[d] /= n;
    means[kind] = m;
  }
  items.forEach((it, i) => {
    const v = new Float32Array(dim);
    const m = means[it.kind];
    for (let d = 0; d < dim; d++) v[d] = raw[i][d] - m[d];
    it.index = i;
    it.vec = normalize(v);
  });
  return { items, dim, means, model: meta.model, raw, themes: meta.themes || [] };
}

// Turn a new CLIP vector (from your photo or words) into the same centered space.
export function centerQuery(archive, vec, kind) {
  const v = Float32Array.from(normalize(Float32Array.from(vec)));
  const m = archive.means[kind];
  for (let d = 0; d < v.length; d++) v[d] -= m[d];
  return normalize(v);
}
