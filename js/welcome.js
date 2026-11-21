// A guided tour: the page dims, one feature at a time is spotlighted with a short note, and you step
// through with Next / Back or skip it. Shown on a first visit; "how it works" replays it.
// Inspiration is where you find what you love; Uncharted is where you go beyond it.

const TOURS = {
  inspiration: [
    { title: "Find what you love.", text: "Inspiration is where you collect: a feed that learns your taste. Here's how it works." },
    { sel: "#pick-grid", title: "Pick what catches your eye", text: "Tap any photo or passage you're drawn to. Pick at least 8 to start." },
    { sel: "#view-start .showme", title: "Photos, writing, or both", text: "Choose what you want to see. It applies everywhere on the site." },
    { sel: ".pick-bar", title: "Your picks", text: "Picks collect in this tray and carry over when you show a different set." },
    { sel: "#pick-more", title: "See a different set", text: "Nothing speaking to you? Show different ones. Your picks are kept." },
    { sel: "#pick-done", title: "See your inspiration", text: "Once you have 8, this opens a feed built from your taste." },
    { sel: "#search", title: "Search by feeling", text: "Type a mood or a place, like \"quiet desert heat\"." },
    { sel: "label[for='upload']", title: "Start from your photo", text: "Upload a photo to find works that match it. It never leaves your device." },
    { sel: "[data-go='mine']", title: "Your own photos", text: "Add your portfolio. The AI ranks it by your taste, privately." },
    { sel: "[data-go='saved']", title: "Your folder", text: "Everything you ♥ is saved here, and teaches the feed your taste." },
    { sel: "#tab-uncharted", title: "Then go beyond", text: "Uncharted shows your taste as a solar system, then takes you past it." },
    { sel: "#tab-coach", title: "Stuck on writing?", text: "The writing coach helps with writer's block, using prompts from your feed." },
    { sel: "[data-welcome]", title: "Replay any time", text: "Open this tour again from \"how it works\"." },
  ],
  feed: [
    { title: "Your feed.", text: "Built from your picks. It keeps learning as you go." },
    { sel: "#feed .card .actions", title: "♥ and ×", text: "♥ saves a work and shows you more like it. × shows you less." },
    { sel: "#feed .card .pairing", title: "Why they're paired", text: "Each photo is paired with a passage, and this says what they share." },
    { sel: "#feed .card button[title^='Write']", title: "Write from it", text: "Opens the writing coach with this pair and a prompt to start you off." },
    { sel: "#feed .card button[title='More like this']", title: "More like this", text: "Fills the feed with works close to this one." },
    { sel: "#taste", title: "Your taste, in words", text: "What the AI has learned you're drawn to. Steer to push it toward something new." },
    { sel: "#retrain", title: "Pick more", text: "Add picks any time. The more you pick, the closer it gets." },
    { sel: "#tab-uncharted", title: "When you're ready", text: "Step into Uncharted to see your collection and go beyond it." },
  ],
  uncharted: [
    { title: "Go beyond your taste.", text: "Inspiration is where you find what you love. Uncharted is where it leads." },
    { sel: "[data-go-gallery='collection']", title: "Your world", text: "Your collection as a solar system: each planet holds a group of works you love. Scroll to travel round it." },
    { sel: "[data-go-gallery='exploration']", title: "Beyond", text: "Galaxies of works you've never seen, each a step further from your taste. ♥ what surprises you." },
    { sel: "[data-go-gallery='map']", title: "The map", text: "Where your taste could go next. Click a star to head there and see what's out there." },
    { sel: ".uc-showme--gallery", title: "Photos, writing, or both", text: "The same switch as in Inspiration." },
    { sel: ".gallery-room", title: "Each room", text: "Named for what its works share. \"Why this room?\" shows how the AI grouped it." },
    { sel: ".gallery-controls", title: "Move between rooms", text: "Scroll to drift through, or jump room to room. Click any work to learn about it." },
    { sel: "[data-welcome]", title: "Replay any time", text: "Open this tour again from \"how it works\"." },
  ],
};

const CSS = `
.tour { position: fixed; inset: 0; z-index: 90; }
.tour-spot { position: fixed; border-radius: 10px; box-shadow: 0 0 0 9999px rgba(8,8,10,.62); outline: 2px solid rgba(255,229,207,.85); outline-offset: 2px; transition: all .35s cubic-bezier(.22,1,.36,1); pointer-events: none; }
.tour-spot.is-center { left: 50% !important; top: 50% !important; width: 0 !important; height: 0 !important; outline: 0; }
.tour-tip { position: fixed; width: min(340px, calc(100vw - 32px)); padding: 18px 20px 16px; border: 1px solid var(--line); border-radius: 12px; background: var(--card, var(--surface, #1c1c1e)); color: var(--ink); box-shadow: 0 18px 50px rgba(0,0,0,.35); transition: left .35s cubic-bezier(.22,1,.36,1), top .35s cubic-bezier(.22,1,.36,1); }
.tour-tip small { display: block; margin-bottom: 8px; color: var(--muted); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.tour-tip strong { display: block; margin-bottom: 6px; font: italic 400 24px/1.15 var(--serif, Georgia, serif); }
.tour-tip.is-center strong { font-size: 34px; }
.tour-tip p { margin: 0; font-size: 14px; line-height: 1.5; color: var(--ink); opacity: .85; }
.tour-row { display: flex; align-items: center; gap: 8px; margin-top: 16px; }
.tour-row span { flex: 1; }
.tour-tip button { min-height: 34px; padding: 0 14px; border: 1px solid var(--line); border-radius: 999px; background: transparent; color: var(--ink); font: inherit; font-size: 13px; cursor: pointer; }
.tour-tip button.tour-next { border-color: var(--ink); background: var(--ink); color: var(--bg); font-weight: 500; }
.tour-tip button.tour-skip { border: 0; padding: 0 4px; color: var(--muted); text-decoration: underline; text-underline-offset: 3px; }
.tour-tip button[hidden] { display: none; }
.tour-tip button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.howto { padding: 6px 0; border: 0; background: none; color: var(--ink); font: inherit; font-size: 13px; text-transform: lowercase; opacity: .5; cursor: pointer; }
.howto:hover { opacity: 1; }
`;

let styled = false;
function addStyle() {
  if (styled) return;
  styled = true;
  const el = document.createElement("style");
  el.textContent = CSS;
  document.head.append(el);
}

const seenKey = (name) => `anotherangle.tour.${name}`;
export function tourSeen(name) { try { return localStorage.getItem(seenKey(name)) === "1"; } catch { return false; } }
function markSeen(name) { try { localStorage.setItem(seenKey(name), "1"); } catch { /* ignore */ } }

// pick(): which tour "how it works" replays right now (Inspiration has one for picking and one for the feed)
export function setupWelcome(page, { pick = () => page } = {}) {
  addStyle();
  document.querySelectorAll("[data-welcome]").forEach((b) => b.addEventListener("click", () => startTour(pick())));
  if (!tourSeen(page)) setTimeout(() => startTour(page), 400);
}
export const openWelcome = (page) => startTour(page);

const visible = (el) => {
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== "hidden";
};

export function startTour(name) {
  addStyle();
  markSeen(name);
  const steps = (TOURS[name] || []).filter((step) => !step.sel || visible(document.querySelector(step.sel)));
  if (!steps.length) return;
  document.querySelector(".tour")?.remove();
  const box = document.createElement("div");
  box.className = "tour";
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-modal", "true");
  box.setAttribute("aria-label", "How it works");
  const spot = document.createElement("div");
  spot.className = "tour-spot";
  const tip = document.createElement("div");
  tip.className = "tour-tip";
  tip.setAttribute("aria-live", "polite");
  box.append(spot, tip);
  document.body.append(box);
  let i = 0;

  const place = () => {
    const step = steps[i];
    const el = step.sel && document.querySelector(step.sel);
    const center = !el;
    spot.classList.toggle("is-center", center);
    tip.classList.toggle("is-center", center);
    const tw = tip.offsetWidth, th = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    if (center) {
      tip.style.left = `${(vw - tw) / 2}px`;
      tip.style.top = `${(vh - th) / 2}px`;
      return;
    }
    const r = el.getBoundingClientRect(), pad = 6;
    Object.assign(spot.style, { left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + pad * 2}px`, height: `${r.height + pad * 2}px` });
    // below the feature if there's room, otherwise above it, always on screen
    let top = r.bottom + 14;
    if (top + th > vh - 12) top = r.top - th - 14;
    if (top < 12) top = Math.min(vh - th - 12, Math.max(12, r.top + 12));
    tip.style.left = `${Math.min(vw - tw - 16, Math.max(16, r.left + r.width / 2 - tw / 2))}px`;
    tip.style.top = `${top}px`;
  };

  const button = (text, cls, onclick) => {
    const b = document.createElement("button");
    b.type = "button";
    if (cls) b.className = cls;
    b.textContent = text;
    b.onclick = onclick;
    return b;
  };
  const render = () => {
    const step = steps[i], last = i === steps.length - 1;
    const el = step.sel && document.querySelector(step.sel);
    if (el) {
      const r = el.getBoundingClientRect();
      if (r.top < 70 || r.bottom > window.innerHeight - 20) el.scrollIntoView({ block: "center", behavior: "instant" });
    }
    tip.replaceChildren();
    const small = document.createElement("small");
    small.textContent = `${i + 1} of ${steps.length}`;
    const strong = document.createElement("strong");
    strong.textContent = step.title;
    const p = document.createElement("p");
    p.textContent = step.text;
    const row = document.createElement("div");
    row.className = "tour-row";
    const skip = button("Skip tour", "tour-skip", close);
    skip.hidden = last;
    const back = button("Back", "", () => { i -= 1; render(); });
    back.hidden = i === 0;
    const next = button(last ? "Done" : i === 0 ? "Show me" : "Next", "tour-next", () => { if (last) close(); else { i += 1; render(); } });
    row.append(skip, document.createElement("span"), back, next);
    tip.append(small, strong, p, row);
    requestAnimationFrame(place);
    next.focus({ preventScroll: true });
  };

  function close() {
    window.removeEventListener("resize", place);
    window.removeEventListener("scroll", place, true);
    document.removeEventListener("keydown", onKey);
    box.remove();
  }
  const onKey = (e) => {
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") tip.querySelector(".tour-next")?.click();
    else if (e.key === "ArrowLeft" && i > 0) { i -= 1; render(); }
  };
  window.addEventListener("resize", place);
  window.addEventListener("scroll", place, true);
  document.addEventListener("keydown", onKey);
  render();
}
