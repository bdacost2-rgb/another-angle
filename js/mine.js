// "My portfolio": your own photos, read by CLIP in your browser and kept on your
// device (IndexedDB). Nothing is uploaded. They join the archive so they show up in
// your feed, searches, "more like this", Rediscover, and the Uncharted map.
import { centerQuery } from "./data.js";

const DB = "anotherangle-portfolio", STORE = "photos";

function db() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: "id" });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => res(out && "result" in out ? out.result : undefined);
    t.onerror = () => rej(t.error);
  });
}

async function shrink(file, max = 720) {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, max / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.85));
}

async function fingerprint(file) {
  const buf = await file.slice(0, 65536).arrayBuffer();
  const h = await crypto.subtle.digest("SHA-1", buf);
  return "m" + [...new Uint8Array(h)].slice(0, 8).map((b) => b.toString(16).padStart(2, "0")).join("") + file.size.toString(36);
}

// Read and save new photos. embed(blob) -> raw CLIP vector.
export async function addPhotos(files, embed, onProgress = () => {}) {
  const list = [...files].filter((f) => f.type.startsWith("image/"));
  const have = new Set(await tx("readonly", (s) => s.getAllKeys()) || []);
  let done = 0, added = 0;
  for (const f of list) {
    try {
      const id = await fingerprint(f);
      if (!have.has(id)) {
        const blob = await shrink(f);
        const vec = await embed(blob);
        const rec = { id, name: f.name.replace(/\.[^.]+$/, ""), date: f.lastModified ? new Date(f.lastModified).toISOString().slice(0, 10) : "", blob, vec: Array.from(vec) };
        await tx("readwrite", (s) => s.put(rec));
        have.add(id); added++;
      }
    } catch (e) { console.warn("Skipped", f.name, e); }
    onProgress(++done, list.length);
  }
  return added;
}

export async function loadPhotos() {
  try { return (await tx("readonly", (s) => s.getAll())) || []; } catch (e) { return []; }
}

export async function clearPhotos() {
  try { await tx("readwrite", (s) => s.clear()); } catch (e) { /* ignore */ }
}

// Turn saved photos into archive items (same centered space as the archive photos).
export function asItems(archive, recs) {
  return recs.map((r) => ({
    id: r.id, kind: "photo", mine: true, img: URL.createObjectURL(r.blob),
    title: r.name, creator: "You", date: r.date, source: "Your portfolio", link: "",
    vec: centerQuery(archive, Float32Array.from(r.vec), "photo"),
  }));
}

// Add your photos into a loaded archive (each gets an index, like the archive's own items).
export async function mergeInto(archive) {
  const mine = asItems(archive, await loadPhotos());
  for (const it of mine) { it.index = archive.items.length; archive.items.push(it); }
  return mine;
}
