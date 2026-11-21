// History for every work, without inventing any.
// 1. "About this work": the archive's own description and facts (data/history.json, made by
//    tools/build_history.py from the Art Institute, Library of Congress and Met APIs), linked to the record.
// 2. "Across time": works that look or read alike (close in CLIP space) but were made in other decades or
//    centuries, on a small timeline. Dates are real and similarity is measured; we call them echoes, not influences.
// 3. "Your taste through time": when the works you love, and the ones Beyond shows you, were made.
import { dot } from "./data.js";

let history = null;
export async function loadHistory(base = "data/") {
  if (history) return history;
  try { history = await (await fetch(base + "history.json")).json(); } catch { history = {}; }
  return history;
}
export const historyOf = (item) => (history && history[item.id]) || null;

export function yearOf(item) {
  const h = historyOf(item);
  if (h?.year) return h.year;
  const m = String(item.date || "").match(/(1[0-9]{3}|20[0-2][0-9])/);
  return m ? Number(m[1]) : null;
}

const decade = (y) => Math.floor(y / 10) * 10;
const label = (y) => (y < 1000 ? `c. ${y}` : String(y));

// Works that resemble this one but come from other eras: at least 25 years apart, one per decade.
export function acrossTime(item, items, { count = 3, fits = () => true } = {}) {
  const y = yearOf(item);
  if (!y || !item.vec) return [];
  const ranked = items
    .filter((it) => it !== item && it.vec && fits(it) && (it.title || "") !== (item.title || ""))
    .map((it) => [it, yearOf(it)])
    .filter(([, yy]) => yy && Math.abs(yy - y) >= 25)
    .map(([it, yy]) => [dot(item.vec, it.vec), it, yy])
    .sort((a, b) => b[0] - a[0]);
  const used = new Set([decade(y)]);
  const out = [];
  for (const [, it, yy] of ranked) {
    if (used.has(decade(yy))) continue;
    used.add(decade(yy));
    out.push({ item: it, year: yy });
    if (out.length >= count) break;
  }
  return out.sort((a, b) => a.year - b.year);
}

// "Looks older than it is": when how a work looks (CLIP's ancient ↔ futuristic dial) disagrees with its date.
export function eraNote(item, eraScore) {
  const y = yearOf(item);
  if (!y || !Number.isFinite(eraScore)) return null;
  if (eraScore < -1.1 && y >= 1940) return `Looks older than it is: the AI reads it as old-fashioned, but it was made in ${label(y)}.`;
  if (eraScore > 1.1 && y < 1900) return `Looks ahead of its time: the AI reads it as modern, but it was made in ${label(y)}.`;
  return null;
}

const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };

// The history section of a work's details: archive facts and description, then echoes across time.
export function renderHistory(item, { items, fits, eraScore, onOpen }) {
  const wrap = el("div", "history");
  const h = historyOf(item);
  const facts = el("dl", "history-facts");
  const add = (k, v) => { if (v) facts.append(el("dt", "", k), el("dd", "", v)); };
  add(item.kind === "text" ? "Written by" : "Made by", h?.maker);
  add("When", h?.made || item.date);
  add("Where", h?.place);
  add("Style", h?.style);
  add("Medium", h?.medium);
  add("Subjects", h?.subjects?.length ? h.subjects.join(" · ") : null);
  if (facts.children.length) wrap.append(facts);
  if (h?.about) {
    const about = el("blockquote", "history-about", h.about);
    const cite = el("cite", "", `From ${item.source || "the archive"}`);
    about.append(cite);
    wrap.append(el("small", "history-label", "About this work, from the archive"), about);
  }
  const note = eraNote(item, eraScore);
  if (note) wrap.append(el("p", "history-note", note));

  const echoes = acrossTime(item, items, { fits });
  const y = yearOf(item);
  if (echoes.length && y) {
    wrap.append(el("small", "history-label", "Across time: works that look or read alike, from other eras"));
    const all = [...echoes, { item, year: y, self: true }].sort((a, b) => a.year - b.year);
    const lo = all[0].year, hi = all[all.length - 1].year, span = Math.max(1, hi - lo);
    const line = el("div", "history-line");
    const list = el("ol", "history-echoes");
    all.forEach((e) => {
      const dotEl = el("i", e.self ? "is-self" : "");
      dotEl.style.left = `${((e.year - lo) / span) * 100}%`;
      line.append(dotEl);
      const li = el("li", e.self ? "is-self" : "");
      const year = el("b", "", label(e.year));
      if (e.self) { li.append(year, el("span", "", "this work")); }
      else {
        const b = el("button", "", e.item.title || "Untitled");
        b.type = "button";
        b.addEventListener("click", () => onOpen?.(e.item));
        li.append(year, b);
      }
      list.append(li);
    });
    wrap.append(line, list);
  }
  return wrap;
}

// When works were made, by decade, for "your taste through time".
export function decades(items) {
  const counts = new Map();
  for (const it of items) { const y = yearOf(it); if (y && y >= 1700) counts.set(decade(y), (counts.get(decade(y)) || 0) + 1); }
  return counts;
}
export function topDecades(items, n = 2) {
  return [...decades(items)].sort((a, b) => b[1] - a[1]).slice(0, n).map(([d]) => `${d}s`);
}
