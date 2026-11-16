// The cursor as a lens. One idea, two crafts:
//  - Writers: the nearby word gets a quiet underline while the passage stays readable.
//  - Photographers: pick photos lift and gently zoom under the pointer.
// Word focus is limited to fine pointers and disabled for reduced-motion preferences.

const fine = matchMedia("(hover: hover) and (pointer: fine)");
const calm = matchMedia("(prefers-reduced-motion: reduce)");
const enabled = () => fine.matches && !calm.matches;

/** Fill `node` with `text`, one <span class="w"> per word, and mark it as lens-reactive. */
export function setWords(node, text) {
  node.replaceChildren();
  node.dataset.words = "";
  String(text).split(/(\s+)/).forEach((part) => {
    if (!part) return;
    if (/^\s+$/.test(part)) { node.append(" "); return; }
    const s = document.createElement("span");
    s.className = "w";
    s.textContent = part;
    node.append(s);
  });
  return node;
}

// ---- words: nearby focus ----
let active = null, frame = 0, px = 0, py = 0;

function measure(box) {
  const r = box.getBoundingClientRect();
  box._lens = { w: r.width, h: r.height, spans: [...box.querySelectorAll(".w")].map((s) => {
    const q = s.getBoundingClientRect();
    return { s, x: q.left - r.left + q.width / 2, y: q.top - r.top + q.height / 2 };
  }) };
}

function focusWords() {
  frame = 0;
  const box = active;
  if (!box) return;
  const r = box.getBoundingClientRect();
  if (!box._lens || Math.abs(box._lens.w - r.width) > 1 || Math.abs(box._lens.h - r.height) > 1) measure(box);
  const size = parseFloat(getComputedStyle(box).fontSize) || 16;
  const x = px - r.left, y = py - r.top;
  let nearest = null, nearestDistance = Infinity;
  for (const w of box._lens.spans) {
    const d = Math.hypot(w.x - x, (w.y - y) * 1.7);
    if (d < nearestDistance) { nearest = w.s; nearestDistance = d; }
  }
  for (const w of box._lens.spans) w.s.classList.toggle("lens-focus", w.s === nearest && nearestDistance < size * 1.8);
}

function release(box) {
  box.querySelectorAll(".lens-focus").forEach((s) => s.classList.remove("lens-focus"));
}

export function initLens() {
  document.addEventListener("pointermove", (e) => {
    if (!enabled() || e.pointerType === "touch") return;
    const t = e.target instanceof Element ? e.target : null;

    const box = t && t.closest("[data-words]");
    if (box !== active) {
      if (active) release(active);
      active = box;
      if (box) measure(box);
    }
    if (active) { px = e.clientX; py = e.clientY; if (!frame) frame = requestAnimationFrame(focusWords); }
  }, { passive: true });

  const clear = () => {
    if (active) { release(active); active = null; }
  };
  document.addEventListener("pointerleave", clear);
  document.documentElement.addEventListener("mouseleave", clear);
  window.addEventListener("blur", clear);

  // static headings marked up in the HTML
  document.querySelectorAll("[data-lens]").forEach((n) => setWords(n, n.textContent.trim()));
}
