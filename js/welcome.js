// A short welcome for each page, shown on a first visit and again from "how it works".
// Inspiration is where you find what you love; Uncharted is where you go beyond it.
const PAGES = {
  inspiration: {
    kicker: "Inspiration",
    title: "Find what you love.",
    steps: [
      ["Pick 8", "photos or passages that catch your eye."],
      ["♥ what you love,", "× what you don't. Hearts save to your folder and teach it your taste."],
      ["Steer", "your feed, or tap Write to start making from a pairing."],
    ],
    next: "Then step into Uncharted to go beyond your taste.",
    button: "Start",
  },
  uncharted: {
    kicker: "Uncharted",
    title: "Go beyond your taste.",
    steps: [
      ["Your world:", "circle the planet you found. Everything you love, in orbit around it."],
      ["Beyond:", "drift through galaxies of works you've never seen, reaching outward from your taste. ♥ what surprises you."],
      ["The map", "shows where your taste could go next. Click a star to head there."],
    ],
    next: "Inspiration is where you find what you love. This is where it leads.",
    button: "Drift in",
  },
};

const CSS = `
.welcome { position: fixed; inset: 0; z-index: 80; display: grid; place-items: center; padding: 24px; background: color-mix(in srgb, var(--bg) 88%, transparent); -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px); opacity: 0; transition: opacity .45s ease; }
.welcome.is-on { opacity: 1; }
.welcome-card { width: min(520px, 100%); color: var(--ink); }
.welcome-card small { display: block; margin-bottom: 14px; color: var(--muted); font-size: 12px; letter-spacing: .1em; text-transform: uppercase; }
.welcome-card h2 { margin: 0 0 26px; font: italic 400 clamp(38px, 6vw, 60px)/1 var(--serif, Georgia, serif); letter-spacing: -.02em; }
.welcome-card ol { margin: 0; padding: 0; list-style: none; counter-reset: step; display: grid; gap: 16px; }
.welcome-card li { counter-increment: step; display: grid; grid-template-columns: 28px 1fr; gap: 12px; font-size: 15px; line-height: 1.5; }
.welcome-card li::before { content: counter(step); display: grid; place-items: center; width: 24px; height: 24px; border: 1px solid var(--line); border-radius: 50%; font-size: 12px; color: var(--muted); }
.welcome-card p { margin: 26px 0 0; color: var(--muted); font-size: 14px; }
.welcome-card button { margin-top: 30px; min-height: 44px; padding: 0 26px; border: 1px solid var(--ink); border-radius: 999px; background: var(--ink); color: var(--bg); font: inherit; font-size: 14px; font-weight: 500; cursor: pointer; }
.welcome-card button:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }
.howto { padding: 6px 0; border: 0; background: none; color: var(--ink); font: inherit; font-size: 13px; text-transform: lowercase; opacity: .5; cursor: pointer; }
.howto:hover { opacity: 1; }
`;

export function setupWelcome(page) {
  const copy = PAGES[page];
  if (!copy) return;
  const style = document.createElement("style");
  style.textContent = CSS;
  document.head.append(style);
  document.querySelectorAll("[data-welcome]").forEach((b) => b.addEventListener("click", () => openWelcome(page)));
  const key = `anotherangle.welcome.${page}`;
  let seen = false;
  try { seen = localStorage.getItem(key) === "1"; localStorage.setItem(key, "1"); } catch { /* show it anyway */ }
  if (!seen) openWelcome(page);
}

export function openWelcome(page) {
  const copy = PAGES[page];
  document.querySelector(".welcome")?.remove();
  const box = document.createElement("div");
  box.className = "welcome";
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-modal", "true");
  box.setAttribute("aria-labelledby", "welcome-title");
  const card = document.createElement("div");
  card.className = "welcome-card";
  const small = document.createElement("small");
  small.textContent = copy.kicker;
  const h = document.createElement("h2");
  h.id = "welcome-title";
  h.textContent = copy.title;
  const ol = document.createElement("ol");
  for (const [lead, rest] of copy.steps) {
    const li = document.createElement("li");
    const span = document.createElement("span");
    const b = document.createElement("b");
    b.textContent = lead;
    span.append(b, ` ${rest}`);
    li.append(span);
    ol.append(li);
  }
  const p = document.createElement("p");
  p.textContent = copy.next;
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = copy.button;
  card.append(small, h, ol, p, button);
  box.append(card);
  document.body.append(box);
  requestAnimationFrame(() => box.classList.add("is-on"));
  const close = () => {
    box.classList.remove("is-on");
    document.removeEventListener("keydown", onKey);
    setTimeout(() => box.remove(), 450);
  };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  button.addEventListener("click", close);
  box.addEventListener("click", (e) => { if (e.target === box) close(); });
  document.addEventListener("keydown", onKey);
  button.focus({ preventScroll: true });
}
