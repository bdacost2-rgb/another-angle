// The cursor as a lens. One idea, two crafts:
//  - Writers: point at a passage and the words under the cursor come into focus while the rest of the
//    paragraph falls softly out of it (depth of field for text).
//  - Photographers: point at a photo and a viewfinder frame follows the cursor, then tightens and
//    "locks focus" when you hold still.
// Everything is at rest (fully sharp, no frame) when the pointer is elsewhere, on touch screens,
// and for people who ask for reduced motion.

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

// ---- words: depth of field ----
let active = null, frame = 0, px = 0, py = 0;

function measure(box) {
  const r = box.getBoundingClientRect();
  box._lens = { w: r.width, spans: [...box.querySelectorAll(".w")].map((s) => {
    const q = s.getBoundingClientRect();
    return { s, x: q.left - r.left + q.width / 2, y: q.top - r.top + q.height / 2 };
  }) };
}

function focusWords() {
  frame = 0;
  const box = active;
  if (!box) return;
  const r = box.getBoundingClientRect();
  if (!box._lens || Math.abs(box._lens.w - r.width) > 1) measure(box);
  const size = parseFloat(getComputedStyle(box).fontSize) || 16;
  const near = size * 2.2, far = size * 9;          // sharp inside `near`, fully soft beyond `far`
  const x = px - r.left, y = py - r.top;
  for (const w of box._lens.spans) {
    const d = Math.hypot(w.x - x, (w.y - y) * 1.7);  // lines above and below fall off faster than words alongside
    const f = Math.max(0, Math.min(1, 1 - (d - near) / (far - near)));
    w.s.style.setProperty("--f", f.toFixed(2));
  }
}

function release(box) {
  box.classList.remove("lensing");
  box.querySelectorAll(".w").forEach((s) => s.style.removeProperty("--f"));
}

// ---- photos: viewfinder ----
let reticle, rx = 0, ry = 0, tx = 0, ty = 0, over = false, lockTimer = 0, raf = 0;

function follow() {
  rx += (tx - rx) * 0.22; ry += (ty - ry) * 0.22;
  reticle.style.transform = `translate(${rx.toFixed(1)}px, ${ry.toFixed(1)}px)`;
  raf = over || Math.abs(tx - rx) + Math.abs(ty - ry) > 0.5 ? requestAnimationFrame(follow) : 0;
}

export function initLens() {
  reticle = document.createElement("div");
  reticle.className = "reticle";
  reticle.setAttribute("aria-hidden", "true");
  reticle.innerHTML = "<i></i><i></i><i></i><i></i><b></b>";
  document.body.append(reticle);

  document.addEventListener("pointermove", (e) => {
    if (!enabled() || e.pointerType === "touch") return;
    const t = e.target instanceof Element ? e.target : null;

    const box = t && t.closest("[data-words]");
    if (box !== active) {
      if (active) release(active);
      active = box;
      if (box) { measure(box); box.classList.add("lensing"); }
    }
    if (active) { px = e.clientX; py = e.clientY; if (!frame) frame = requestAnimationFrame(focusWords); }

    const photo = t && t.closest("img.photo, .pick img, .mini-card img");
    tx = e.clientX; ty = e.clientY;
    if (photo && !over) { rx = tx; ry = ty; }
    over = !!photo;
    reticle.classList.toggle("on", over);
    reticle.classList.remove("locked");
    clearTimeout(lockTimer);
    if (over) { lockTimer = setTimeout(() => reticle.classList.add("locked"), 260); if (!raf) raf = requestAnimationFrame(follow); }
  }, { passive: true });

  const clear = () => {
    if (active) { release(active); active = null; }
    over = false; clearTimeout(lockTimer);
    reticle.classList.remove("on", "locked");
  };
  document.addEventListener("pointerleave", clear);
  document.documentElement.addEventListener("mouseleave", clear);
  window.addEventListener("blur", clear);

  // static headings marked up in the HTML
  document.querySelectorAll("[data-lens]").forEach((n) => setWords(n, n.textContent.trim()));
}
