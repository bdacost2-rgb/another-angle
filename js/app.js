import { loadArchive, centerQuery } from "./data.js";
import { TasteModel } from "./taste.js";
import { Coach } from "./coach.js";
import * as models from "./models.js";
import { addPhotos, loadPhotos, asItems, mergeInto, clearPhotos } from "./mine.js";
import { initLens, setWords } from "./lens.js";
import { loadDials, itemDials, dialWord, strongestWords } from "./dials.js";
import { setupWelcome, startTour, tourSeen } from "./welcome.js";

const $ = (s) => document.querySelector(s);
const ICON = {
  heart: '<path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"/>',
  less: '<path d="M6 6l12 12M18 6L6 18"/>',
  similar: '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  write: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>',
};

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v; // only used for our own icons
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) n.append(k instanceof Node ? k : document.createTextNode(k));
  return n;
}
const icon = (name) => el("span", { html: `<svg viewBox="0 0 24 24" aria-hidden="true">${ICON[name]}</svg>`, style: "display:inline-flex" });

// ---------- Stable masonry ----------
// CSS columns re-balance every card whenever new ones are added, so the feed jumped as more loaded.
// Here each card is placed once, in whichever column is shortest, and stays there. Columns are only
// rebuilt when the window gets wide or narrow enough to change how many fit.
const COL_MIN = 290, COL_GAP = 28;
const layouts = new WeakMap();
function columnCount(box) {
  if (window.innerWidth <= 640) return 1;
  const width = box.clientWidth || window.innerWidth - 48;
  return Math.max(1, Math.min(4, Math.floor((width + COL_GAP) / (COL_MIN + COL_GAP))));
}
function layoutOf(box) {
  let L = layouts.get(box);
  if (!L) {
    L = { cards: [], cols: [] };
    layouts.set(box, L);
    new ResizeObserver(() => { if (L.cols.length && columnCount(box) !== L.cols.length) relayout(box); }).observe(box);
  }
  return L;
}
function buildColumns(box) {
  const L = layoutOf(box);
  L.cols = Array.from({ length: columnCount(box) }, () => el("div", { class: "masonry-col" }));
  box.replaceChildren(...L.cols);
  box.classList.add("is-js");
}
function placeCard(box, card) {
  const L = layoutOf(box);
  if (!L.cols.length || L.cols[0].parentNode !== box) buildColumns(box);
  L.cards.push(card);
  const visible = box.offsetParent !== null;
  let target = L.cols[(L.cards.length - 1) % L.cols.length]; // hidden views can't be measured: take turns
  if (visible) for (const col of L.cols) if (col.offsetHeight < target.offsetHeight) target = col;
  target.append(card);
}
function relayout(box) {
  const L = layoutOf(box);
  const cards = L.cards.filter((c) => c.isConnected);
  L.cards = [];
  buildColumns(box);
  for (const card of cards) placeCard(box, card);
}
function clearCards(box) {
  const L = layoutOf(box);
  L.cards = [];
  L.cols = [];
  box.replaceChildren();
  box.classList.remove("is-js");
}

let A, T, coach;
let DIALS = []; // the twelve CLIP design dials (js/dials.js)
const shown = new Set();
let mode = { kind: "taste" }; // or {kind:"similar", item} | {kind:"search", q, vec} | {kind:"photo", url, vec}
// 8 picks in total to start, across as many pages as you like ("Show different ones" keeps what you
// picked). 8 is only a start: the counter keeps encouraging more, up to a well-known taste at 16.
const PICKS_NEEDED = 8;

// What you're here for: photos, writing, or both side by side. Remembered on this device.
const FOCUS_KEY = "anotherangle.focus.v1";
let focus = (() => { try { return localStorage.getItem(FOCUS_KEY) || "both"; } catch { return "both"; } })();
if (!["photo", "both", "text"].includes(focus)) focus = "both";
function setFocus(next) {
  focus = next;
  try { localStorage.setItem(FOCUS_KEY, next); } catch { /* ignore */ }
  syncFocus();
  toast(next === "photo" ? "Showing photos only" : next === "text" ? "Showing writing only" : "Showing photos paired with writing");
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (!$("#view-start").hidden) {
    // keep what you picked on this page, without counting the rest as "not this": you only changed the filter
    for (const id of selectedPicks()) T.labels.set(id, { y: 1, w: 1 });
    T.fit(); T.save(); renderPicks();
  }
  else if (!$("#view-feed").hidden) resetFeed();
}
function syncFocus() {
  document.querySelectorAll("[data-focus]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.focus === focus)));
}

function toast(text, ms = 2600) {
  const t = $("#toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toast.t);
  if (ms) toast.t = setTimeout(() => (t.hidden = true), ms);
}

function show(view) {
  for (const v of ["start", "feed", "saved", "mine", "coach"]) $(`#view-${v}`).hidden = v !== view;
  // Tabs: the coach is its own tab; everything else belongs to Inspiration.
  const onCoach = view === "coach";
  for (const [id, on] of [["#tab-coach", onCoach], ["#tab-feed", !onCoach]]) {
    if (on) $(id).setAttribute("aria-current", "page"); else $(id).removeAttribute("aria-current");
  }
  $("#tools").hidden = onCoach;
  const want = onCoach ? "#coach" : "";
  if (location.hash !== want) history.replaceState(null, "", location.pathname + location.search + want);
  window.scrollTo(0, 0);
}

function credit(item) {
  const parts = [item.creator, item.title, item.date].filter(Boolean).join(", ");
  return el("div", { class: item.kind === "photo" ? "photo-credit" : "credit" },
    parts ? parts + " · " : "",
    item.link ? el("a", { href: item.link, target: "_blank", rel: "noopener" }, item.source) : item.source);
}

function passage(item, lead = false) {
  return el("div", { class: "passage" + (lead ? " lead" : "") }, setWords(el("p", {}), item.text));
}

// ---------- Onboarding: pick what moves you ----------
let pickSet = [];
// Archive images sometimes fail to load. A broken photo is quietly replaced (on the pick screen) or
// its card is removed (in the feed and Saved), and it isn't offered again this visit.
const brokenImages = new Set();
function renderPicks() {
  const grid = $("#pick-grid");
  grid.replaceChildren();
  const photos = A.items.filter((i) => i.kind === "photo" && !i.mine && !brokenImages.has(i.id));
  const texts = A.items.filter((i) => i.kind === "text");
  const sample = (arr, n) => [...arr].sort(() => Math.random() - 0.5).slice(0, n);
  // A spread of the archive: spaced apart so the first picks are informative.
  pickSet = focus === "photo" ? spread(sample(photos, 180), 24)
    : focus === "text" ? spread(sample(texts, 90), 14)
    : spread(sample(photos, 120), 16).concat(spread(sample(texts, 60), 8)).sort(() => Math.random() - 0.5);
  const before = [...T.labels.values()].filter((l) => l.y === 1 && l.w >= 1).length;
  $("#pick-intro").textContent = before
    ? `Your ${before} earlier pick${before === 1 ? " is" : "s are"} kept. Add more that feel like you.`
    : "Pick at least 8 that catch your eye. Your picks carry over between pages.";
  const makePick = (it) => {
    const picked = T.labels.get(it.id)?.y === 1;
    let sizeClass = "";
    if (it.kind === "text") {
      const size = it.text.length < 180 ? "short" : it.text.length < 460 ? "medium" : "long";
      sizeClass = ` size-${size}`;
    }
    const b = el("button", { class: `pick pick-${it.kind}${sizeClass}`, "aria-pressed": String(picked), onclick: () => {
      b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
      updatePickCount();
    } },
    it.kind === "photo" ? el("img", { src: it.img, alt: it.title || "Archive photo", loading: "lazy", onerror: () => {
      brokenImages.add(it.id);
      const onScreen = new Set(pickSet.map((p) => p.id));
      const swap = photos.find((p) => !onScreen.has(p.id) && !brokenImages.has(p.id));
      if (!swap) { b.remove(); return; }
      pickSet[pickSet.indexOf(it)] = swap;
      b.replaceWith(makePick(swap));
      updatePickCount();
    } }) : setWords(el("span", { class: "txt" }), it.text));
    b.dataset.id = it.id;
    return b;
  };
  for (const it of pickSet) grid.append(makePick(it));
  updatePickCount();
}
function spread(items, n) {
  // Farthest-point sampling: each next item is the one least like those already chosen.
  if (items.length <= n) return items;
  const out = [items[0]];
  const minD = items.map((it) => 1 - dotv(it.vec, items[0].vec));
  while (out.length < n) {
    let bi = 0;
    for (let i = 1; i < items.length; i++) if (minD[i] > minD[bi]) bi = i;
    out.push(items[bi]);
    for (let i = 0; i < items.length; i++) minD[i] = Math.min(minD[i], 1 - dotv(items[i].vec, items[bi].vec));
  }
  return out;
}
const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

function selectedPicks() { return [...document.querySelectorAll('.pick[aria-pressed="true"]')].map((b) => b.dataset.id); }
function updatePickCount() {
  const earlier = earlierPicks();
  const here = selectedPicks().map((id) => T.byId.get(id)).filter(Boolean);
  const total = earlier.length + here.length;
  $("#pick-count").textContent = total < PICKS_NEEDED
    ? `${total} of ${PICKS_NEEDED} picked · ${PICKS_NEEDED - total} to go`
    : `${total} picked`;
  $("#pick-more").textContent = "Show different ones";
  $("#pick-done").disabled = total < PICKS_NEEDED;
  renderTray([...earlier, ...here]);
}
function earlierPicks() {
  // likes from earlier pages of "Show different ones" (and earlier visits) that are not on screen
  const onScreen = new Set(pickSet.map((i) => i.id));
  return [...T.labels].filter(([id, l]) => l.y === 1 && l.w >= 1 && !onScreen.has(id)).map(([id]) => T.byId.get(id)).filter(Boolean);
}
// Everything picked so far, on this page or earlier ones, as small thumbnails. Select one to remove it.
function renderTray(items) {
  const tray = $("#pick-tray");
  if (!tray) return;
  tray.replaceChildren();
  if (!items.length) return;
  tray.append(el("span", { class: "pick-tray-label" }, "Your picks"));
  const MAX = 10;
  for (const it of items.slice(-MAX)) {
    const name = it.kind === "photo" ? it.title || "a photo" : `a passage from ${it.title || "a book"}`;
    tray.append(el("button", { type: "button", title: `Remove ${name}`, "aria-label": `Remove ${name} from your picks`, onclick: () => {
      const here = document.querySelector(`.pick[data-id="${CSS.escape(it.id)}"]`);
      if (here) here.setAttribute("aria-pressed", "false");
      else T.unlabel(it);
      updatePickCount();
    } }, it.kind === "photo" ? el("img", { src: it.img, alt: "", onerror: (e) => e.target.replaceWith(el("span", {}, "▢")) }) : el("span", {}, "“")));
  }
  if (items.length > MAX) tray.append(el("span", { class: "pick-tray-more" }, `+${items.length - MAX}`));
}
function commitPicks() {
  const chosen = new Set(selectedPicks());
  for (const it of pickSet) {
    if (chosen.has(it.id)) T.labels.set(it.id, { y: 1, w: 1 });
    else if (!T.labels.has(it.id) || T.labels.get(it.id).y === 1) T.labels.set(it.id, { y: 0, w: 0.3 }); // seen but not picked: a weak "not this"
  }
  T.fit(); T.save();
}

// ---------- Your taste, in words, and steering it ----------
// The works you liked, averaged on each CLIP design dial, say what you're drawn to. Each dial is
// also a slider: its dot starts at your taste, and moving it re-ranks the feed toward that pole.
const STEER_KEY = "anotherangle.steer.v1";
let steer = new Float32Array(0); // per dial, in slider units (-100..100) away from your taste
function loadSteer() {
  steer = new Float32Array(DIALS.length);
  try {
    const saved = JSON.parse(localStorage.getItem(STEER_KEY) || "null");
    if (Array.isArray(saved) && saved.length === DIALS.length) steer = Float32Array.from(saved);
  } catch { /* private mode: steering just isn't remembered */ }
}
function saveSteer() { try { localStorage.setItem(STEER_KEY, JSON.stringify([...steer])); } catch { /* ignore */ } }
const steered = () => steer.some((v) => Math.abs(v) >= 4);

function tasteProfile() {
  const liked = T.liked;
  const p = new Float32Array(DIALS.length);
  if (!liked.length) return p;
  for (const it of liked) itemDials(it, DIALS).forEach((z, k) => { p[k] += z / liked.length; });
  return p;
}
const basePosition = (z) => 100 / (1 + Math.exp(-1.3 * z)); // your taste as a slider position, 0..100

// Added to each work's score in the feed: how far it sits along the dials you pushed.
function steerBonus() {
  if (!steered()) return null;
  return (item) => {
    const z = itemDials(item, DIALS);
    let s = 0;
    for (let k = 0; k < z.length; k++) s += (steer[k] / 50) * z[k] * 0.3;
    return s;
  };
}

let steerTimer = 0;
function renderTaste() {
  const box = $("#taste");
  if (!box || !DIALS.length || !T.count) { if (box) box.hidden = true; return; }
  box.hidden = false;
  const profile = tasteProfile();
  const words = strongestWords(profile, DIALS, 3);
  $("#taste-words").textContent = words.length ? words.join(" · ") : "a bit of everything";
  const list = $("#taste-dials");
  if (list.children.length !== DIALS.length + 1) {
    list.replaceChildren();
    DIALS.forEach((dial, k) => {
      const input = el("input", { type: "range", min: "0", max: "100", step: "1", "aria-label": `${dial.low} to ${dial.high}` });
      input.addEventListener("input", () => {
        steer[k] = Number(input.value) - Number(input.dataset.base);
        input.closest(".dial").classList.toggle("is-steered", Math.abs(steer[k]) >= 4);
        clearTimeout(steerTimer);
        steerTimer = setTimeout(() => { saveSteer(); renderMode(); resetFeed(); }, 260);
      });
      list.append(el("label", { class: "dial" }, el("span", {}, dial.low), input, el("span", {}, dial.high)));
    });
    list.append(el("button", { class: "btn ghost small", type: "button", onclick: () => { steer.fill(0); saveSteer(); renderTaste(); resetFeed(); } }, "Back to my taste"));
  }
  [...list.querySelectorAll(".dial")].forEach((row, k) => {
    const base = basePosition(profile[k]);
    const input = row.querySelector("input");
    input.dataset.base = base.toFixed(1);
    input.value = String(Math.round(Math.max(0, Math.min(100, base + steer[k]))));
    row.style.setProperty("--base", `${base.toFixed(1)}%`);
    row.classList.toggle("is-steered", Math.abs(steer[k]) >= 4);
  });
}

// ---------- Reading your taste: the moment after you pick ----------
// Your picks drift in and gather, CLIP reads them on the twelve design dials (each dot settles where your
// picks sit), and your taste lands in words. Then it dissolves into the feed. Skippable; short if you
// prefer reduced motion.
let revealTimers = [];
function playReveal() {
  const box = $("#reveal");
  if (!box || !DIALS.length || !T.liked.length) return;
  revealTimers.forEach(clearTimeout);
  revealTimers = [];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const profile = tasteProfile();
  const words = strongestWords(profile, DIALS, 3);
  const strongest = new Set([...profile].map((v, k) => [Math.abs(v), k]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([, k]) => k));

  const field = box.querySelector(".reveal-picks");
  field.replaceChildren();
  const picks = T.liked.slice(-12);
  picks.forEach((it, i) => {
    const a = (i / picks.length) * Math.PI * 2 - Math.PI / 2;
    const tile = el("div", { class: "reveal-pick" },
      it.kind === "photo" ? el("img", { src: it.img, alt: "", onerror: (e) => e.target.remove() }) : el("span", {}, `${(it.text || "").slice(0, 80)}…`));
    tile.style.setProperty("--sx", `${((Math.random() - 0.5) * 110).toFixed(1)}vw`);
    tile.style.setProperty("--sy", `${((Math.random() - 0.5) * 110).toFixed(1)}vh`);
    tile.style.setProperty("--sr", `${((Math.random() - 0.5) * 40).toFixed(1)}deg`);
    tile.style.setProperty("--cx", Math.cos(a).toFixed(3));
    tile.style.setProperty("--cy", Math.sin(a).toFixed(3));
    tile.style.setProperty("--d", `${i * 70}ms`);
    field.append(tile);
  });

  const dials = box.querySelector(".reveal-dials");
  dials.replaceChildren();
  const rows = DIALS.map((dial, k) => {
    const row = el("div", { class: `reveal-dial${strongest.has(k) ? " is-strong" : ""}` }, el("span", {}, dial.low), el("i"), el("span", {}, dial.high));
    row.style.setProperty("--d", `${k * 90}ms`);
    dials.append(row);
    return [row, basePosition(profile[k]) / 100];
  });
  box.querySelector(".reveal-words strong").textContent = words.length ? words.join(" · ") : "a bit of everything";
  box.querySelector(".reveal-words small").textContent = `Your taste, from your ${T.count} choices`;

  const finish = () => {
    revealTimers.forEach(clearTimeout);
    box.classList.add("is-out");
    revealTimers = [setTimeout(() => { box.hidden = true; box.className = "reveal"; }, 700)];
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e) => { if (e.key === "Escape" || e.key === "Enter" || e.key === " ") { e.preventDefault(); finish(); } };
  box.querySelector(".reveal-skip").onclick = finish;
  box.onclick = (e) => { if (e.target === box) finish(); };
  document.addEventListener("keydown", onKey);

  box.className = "reveal";
  box.hidden = false;
  const at = (ms, fn) => revealTimers.push(setTimeout(fn, ms));
  if (reduce) {
    box.classList.add("is-words");
    at(2200, finish);
    return;
  }
  requestAnimationFrame(() => box.classList.add("is-in"));
  at(500, () => box.classList.add("is-gather"));
  at(1300, () => { box.classList.add("is-reading"); rows.forEach(([row, v]) => row.style.setProperty("--v", v.toFixed(3))); });
  at(3500, () => box.classList.add("is-words"));
  at(6200, finish);
}

// What a photo and passage have in common on the dials: the poles both lean toward.
function sharedWords(a, b, n = 3) {
  if (!DIALS.length || !a || !b) return [];
  const za = itemDials(a, DIALS), zb = itemDials(b, DIALS);
  return DIALS.map((dial, k) => ({ dial, k, v: Math.sign(za[k]) === Math.sign(zb[k]) ? Math.min(Math.abs(za[k]), Math.abs(zb[k])) : 0, s: Math.sign(za[k]) }))
    .filter(({ v }) => v > 0.3).sort((x, y) => y.v - x.v).slice(0, n).map(({ dial, s }) => dialWord(dial, s));
}

// ---------- Feed ----------
// After your first picks: point to the gallery those picks just built (until you go there or dismiss it).
const NEXT_KEY = "anotherangle.nextstep.v1";
function renderNextStep() {
  const box = $("#next-step");
  if (!box) return;
  let done = false;
  try { done = localStorage.getItem(NEXT_KEY) === "done"; } catch { /* ignore */ }
  box.hidden = done || T.count < PICKS_NEEDED || mode.kind !== "taste";
}
function finishNextStep() { try { localStorage.setItem(NEXT_KEY, "done"); } catch { /* ignore */ } renderNextStep(); }

function resetFeed() {
  shown.clear();
  clearCards($("#feed"));
  renderMode();
  renderTaste();
  renderNextStep();
  loadMore(12);
}

function renderMode() {
  const m = $("#mode");
  m.replaceChildren();
  const clear = el("button", { "aria-label": "Back to your feed", onclick: () => { mode = { kind: "taste" }; resetFeed(); } }, icon("close"));
  if (mode.kind === "taste") {
    m.append(el("span", {}, T.count ? `For you · ${T.count} choices` : "Exploring"));
    if (steered()) {
      const pushed = DIALS.map((dial, k) => ({ dial, v: steer[k] })).filter(({ v }) => Math.abs(v) >= 4)
        .sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 3).map(({ dial, v }) => dialWord(dial, v));
      m.append(el("span", { class: "chip" }, `steered toward ${pushed.join(", ")}`,
        el("button", { "aria-label": "Stop steering", onclick: () => { steer.fill(0); saveSteer(); renderTaste(); resetFeed(); } }, icon("close"))));
    }
  } else if (mode.kind === "similar") {
    m.append("More like", el("span", { class: "chip" },
      mode.item.kind === "photo" ? el("img", { src: mode.item.img, alt: "" }) : null,
      mode.item.kind === "photo" ? (mode.item.title || "this photo").slice(0, 40) : `"${mode.item.text.slice(0, 40)}…"`, clear));
  } else if (mode.kind === "search") {
    m.append("Searching", el("span", { class: "chip" }, `"${mode.q}"`, clear));
  } else if (mode.kind === "photo") {
    m.append("Matching", el("span", { class: "chip" }, el("img", { src: mode.url, alt: "" }), "your photo", clear));
  }
}

function anchor() {
  if (mode.kind === "similar") return mode.item.vec;
  if (mode.kind === "search" || mode.kind === "photo") return mode.vec;
  return null;
}

function loadMore(n = 9) {
  const feed = $("#feed");
  const nPhoto = Math.round(n * 0.67);
  const bonus = steerBonus();
  const plan = focus === "photo" ? [["photo", n]] : focus === "text" ? [["text", n]] : [["photo", nPhoto], ["text", n - nPhoto]];
  const skip = new Set([...shown, ...brokenImages]);
  const leads = plan.flatMap(([kind, k]) => T.pick(k, { kind, exclude: skip, anchor: anchor(), bonus }))
    .sort(() => Math.random() - 0.5);
  if (!leads.length) { $("#more").textContent = "That's everything in the archive for now."; return; }
  for (const lead of leads) {
    shown.add(lead.id);
    const partner = focus === "both" ? T.partnerFor(lead, shown) : null; // photos or writing on their own otherwise
    if (partner) shown.add(partner.id);
    placeCard(feed, pairCard(lead, partner));
  }
}

function heartBtn(item) {
  const on = T.labels.get(item.id)?.y === 1 && T.labels.get(item.id)?.w >= 1;
  const b = el("button", { "aria-pressed": String(on), "aria-label": `Save ${item.kind === "photo" ? "photo" : "passage"}`, title: "Save, and show me more like this",
    onclick: () => {
      const now = b.getAttribute("aria-pressed") !== "true";
      b.setAttribute("aria-pressed", String(now));
      if (now) T.label(item, 1); else T.unlabel(item);
      updateSaved();
      renderMode();
      renderTaste();
    } }, icon("heart"));
  return b;
}

function pairCard(lead, partner) {
  const card = el("article", { class: "card" });
  const parts = lead.kind === "photo" ? [lead, partner] : [partner, lead];
  const [photo, text] = parts;
  if (photo) {
    card.append(el("img", { class: "photo", src: photo.img, alt: photo.title || "Archive photo", loading: "lazy", onerror: () => { brokenImages.add(photo.id); card.remove(); } }), credit(photo));
    if (photo.mine) card.append(el("span", { class: "badge" }, "Your photo"));
  }
  if (text) card.append(passage(text, lead.kind === "text"), credit(text));
  const shared = sharedWords(photo, text);
  if (photo && text && DIALS.length) {
    card.append(el("div", { class: "pairing" }, shared.length ? "Both feel " : "Paired for their meaning", shared.length ? el("em", {}, shared.join(" · ")) : null));
  }
  const why = mode.kind === "taste" && T.because(lead);
  if (why) card.append(el("div", { class: "because" }, "Because you liked ", why.kind === "photo" ? `"${(why.title || "a photo").slice(0, 50)}"` : `a passage from ${why.title || why.creator}`));
  card.append(el("div", { class: "actions" },
    photo ? heartBtn(photo) : null,
    text ? heartBtn(text) : null,
    el("span", { class: "grow" }),
    photo && text ? el("button", { title: "Write from this pair in the writing coach", onclick: () => {
      openCoach({ item: photo, partner: text, words: shared.length ? shared : strongestWords(itemDials(photo, DIALS), DIALS, 2) });
    } }, icon("write"), "Write") : null,
    el("button", { title: "More like this", onclick: () => { mode = { kind: "similar", item: lead }; resetFeed(); window.scrollTo(0, 0); } }, icon("similar"), "More like this"),
    el("button", { "aria-label": "Less like this", title: "Less like this", onclick: () => {
      T.label(lead, 0); card.classList.add("fade"); setTimeout(() => card.remove(), 350); toast("Got it. Less like that.");
    } }, icon("less")),
  ));
  return card;
}

// ---------- Saved ----------
function updateSaved() {
  const liked = T.liked;
  $("#saved-count").textContent = liked.length;
  const box = $("#saved");
  clearCards(box);
  if (!liked.length) { box.append(el("p", { class: "empty" }, "Nothing saved yet. Tap the heart on anything in your feed.")); return; }
  for (const it of liked.reverse()) {
    const card = el("article", { class: "card" });
    if (it.kind === "photo") card.append(el("img", { class: "photo", src: it.img, alt: it.title || "", loading: "lazy", onerror: () => card.remove() }), credit(it));
    else card.append(passage(it), credit(it));
    card.append(el("div", { class: "actions" }, heartBtn(it), el("span", { class: "grow" }),
      el("button", { onclick: () => { mode = { kind: "similar", item: it }; show("feed"); resetFeed(); } }, icon("similar"), "More like this")));
    placeCard(box, card);
  }
}

// ---------- Search and your own photo ----------
async function runSearch(q) {
  if (!q) return;
  toast("Reading your words… (the first search loads the AI, about a minute)", 0);
  try {
    const v = await models.clipText(q);
    mode = { kind: "search", q, vec: centerQuery(A, v, "text") };
    $("#toast").hidden = true;
    show("feed"); resetFeed();
  } catch (e) {
    console.error(e);
    toast("Couldn't load the search model. Check your connection and try again.");
  }
}

async function runPhoto(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  toast("Looking at your photo… (the first time loads the AI, about a minute)", 0);
  try {
    const v = await models.clipImage(file);
    mode = { kind: "photo", url, vec: centerQuery(A, v, "photo") };
    $("#toast").hidden = true;
    show("feed"); resetFeed();
  } catch (e) {
    console.error(e);
    toast("Couldn't read that photo. Try a JPG or PNG.");
  }
}

// ---------- My portfolio ----------
async function importMine(files) {
  if (!files || !files.length) return;
  const status = $("#mine-status");
  status.textContent = "Loading the AI (about a minute the first time)…";
  try {
    const added = await addPhotos(files, (blob) => models.clipImage(blob), (n, total) => {
      status.textContent = `Reading your photos: ${n} of ${total}`;
    });
    const known = new Set(A.items.map((i) => i.id));
    const fresh = asItems(A, (await loadPhotos()).filter((r) => !known.has(r.id)));
    for (const it of fresh) { it.index = A.items.length; A.items.push(it); T.byId.set(it.id, it); }
    status.textContent = added ? `Added ${added} photo${added === 1 ? "" : "s"}. They're saved on this device only.` : "Those photos were already here.";
    renderMine();
  } catch (e) {
    console.error(e);
    status.textContent = "Couldn't load the AI to read your photos. Check your connection and try again.";
  }
}

function renderMine() {
  const mine = A.items.filter((i) => i.mine);
  $("#mine-count").textContent = mine.length ? `${mine.length} of your photos` : "";
  $("#mine-clear").hidden = !mine.length;
  const box = $("#mine");
  clearCards(box);
  if (!mine.length) { box.append(el("p", { class: "empty" }, "Your photos will appear here.")); return; }
  // Rediscover: your own photos, ranked by your taste model, so the strong ones you forgot come back first.
  const ranked = T.count ? [...mine].sort((a, b) => T.score(b) - T.score(a)) : [...mine].sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  $("#mine-note").textContent = T.count
    ? "Ranked by your taste: the ones near the top match what you love most. Some you may have forgotten."
    : "Pick some favorites first (Pick more) and these will be ranked by your taste.";
  for (const it of ranked) {
    const why = T.count && T.because(it);
    const card = el("article", { class: "card" },
      el("img", { class: "photo", src: it.img, alt: it.title || "Your photo", loading: "lazy" }),
      el("div", { class: "photo-credit" }, [it.title, it.date].filter(Boolean).join(" · ")));
    const partner = T.partnerFor(it);
    if (partner) card.append(passage(partner), credit(partner));
    if (why && !why.mine) card.append(el("div", { class: "because" }, "Close to ", why.kind === "photo" ? `"${(why.title || "a photo").slice(0, 50)}"` : `a passage from ${why.title}`, ", which you liked"));
    card.append(el("div", { class: "actions" }, heartBtn(it), el("span", { class: "grow" }),
      el("button", { onclick: () => { mode = { kind: "similar", item: it }; show("feed"); resetFeed(); } }, icon("similar"), "More like this")));
    placeCard(box, card);
  }
}

// ---------- Coach ----------
function openCoach(seed = null) {
  show("coach");
  const fresh = !coach;
  if (!coach) {
    coach = new Coach({
      leave: () => {
        if (!T || T.count < PICKS_NEEDED) { renderPicks(); show("start"); return; }
        show("feed");
        if (!$("#feed").querySelector(".card")) { mode = { kind: "taste" }; resetFeed(); } // arrived from Uncharted: the feed hasn't loaded yet
      },
      embed: (list) => models.sentences(list),
      inspiration: () => {
        if (!T) return null; // the coach also works on its own, before the archive has loaded
        const lead = T.pick(1, { kind: "photo", exclude: shown })[0];
        return lead ? { item: lead, partner: T.partnerFor(lead) } : null;
      },
      say: renderMsg,
    });
  }
  if (seed) coach.writeFrom(seed);
  else if (fresh) coach.start();
  setTimeout(() => $("#chat-text").focus({ preventScroll: true }), 50);
}

let pending = null;
function renderMsg(m) {
  const chat = $("#chat");
  chat.querySelectorAll(".opts button").forEach((b) => (b.disabled = true));
  chat.append(el("div", { class: `msg ${m.from}` }, m.text));
  if (m.card) {
    const { item, partner } = m.card;
    chat.append(el("div", { class: "mini-card" },
      el("img", { src: item.img, alt: item.title || "", onerror: (e) => e.target.remove() }),
      partner ? el("p", {}, partner.text.slice(0, 220) + (partner.text.length > 220 ? "…" : "")) : null,
      el("small", {}, [item.title, partner && partner.creator].filter(Boolean).join(" · "))));
  }
  if (m.options) {
    const box = el("div", { class: "opts" });
    const chosen = new Set();
    for (const o of m.options) {
      const b = el("button", { "aria-pressed": "false", onclick: () => {
        if (m.multi) { b.setAttribute("aria-pressed", String(!chosen.has(o))); chosen.has(o) ? chosen.delete(o) : chosen.add(o); return; }
        if (m.restart) { coach.start(); return; }
        coach.reply(o);
      } }, o);
      box.append(b);
    }
    if (m.multi) box.append(el("button", { onclick: () => coach.reply([...chosen].join(" | ")) }, "Done"));
    chat.append(box);
  }
  if (m.skip) chat.append(el("div", { class: "opts" }, el("button", { onclick: () => coach.reply(m.skip) }, m.skip)));
  const ta = $("#chat-text");
  ta.placeholder = m.input || "Type here";
  pending = m;
  chat.scrollTop = chat.scrollHeight;
}

// ---------- Start ----------
async function main() {
  initLens();
  document.querySelectorAll("[data-focus]").forEach((b) => b.addEventListener("click", () => { if (T) setFocus(b.dataset.focus); }));
  syncFocus();
  // The writing coach doesn't need the archive, so its tab is wired up first and works either way.
  $("#tab-coach").addEventListener("click", (e) => { e.preventDefault(); openCoach(); });
  window.addEventListener("hashchange", () => { if (location.hash === "#coach") openCoach(); });
  $("#taste-toggle")?.addEventListener("click", () => {
    const list = $("#taste-dials"), open = list.hidden;
    list.hidden = !open;
    $("#taste-toggle").setAttribute("aria-expanded", String(open));
    $("#taste-toggle").textContent = open ? "Hide dials" : "Steer";
  });
  $("#chat-form").onsubmit = (e) => {
    e.preventDefault();
    const ta = $("#chat-text");
    const v = ta.value.trim();
    if (!v || !coach) return;
    ta.value = "";
    coach.reply(v);
  };
  $("#chat-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); }
  });
  if (location.hash === "#coach") openCoach();

  try {
    A = await loadArchive();
  } catch (e) {
    console.error(e);
    $("#view-start").replaceChildren(el("p", { class: "empty" }, "The archive data is missing. Run tools/build_data.py (see README) to create data/items.json and data/emb.bin."));
    document.querySelectorAll('[data-go="feed"]').forEach((a) => a.addEventListener("click", (ev) => { ev.preventDefault(); show("start"); }));
    if (location.hash !== "#coach") show("start");
    return;
  }
  await mergeInto(A); // your portfolio, if you've added one
  T = new TasteModel(A);
  DIALS = await loadDials();
  loadSteer();
  updateSaved();

  $("#pick-done").onclick = () => { commitPicks(); pickSet = []; updateSaved(); mode = { kind: "taste" }; show("feed"); resetFeed(); playReveal(); tourFeedWhenReady(); };
  $("#pick-more").onclick = () => {
    const kept = selectedPicks().length + earlierPicks().length;
    commitPicks(); renderPicks();
    window.scrollTo({ top: 0, behavior: "smooth" });
    toast(kept ? `Kept your ${kept} pick${kept === 1 ? "" : "s"}.` : "Here are different ones.");
  };
  $("#retrain").onclick = () => { renderPicks(); show("start"); };
  document.querySelectorAll("[data-go]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    const go = a.dataset.go;
    if (go === "saved") updateSaved();
    if (go === "mine") renderMine();
    if (go === "feed" && !T.count) { renderPicks(); show("start"); return; }
    if (go === "feed") { mode = { kind: "taste" }; resetFeed(); }
    show(go);
  }));
  $("#search").onsubmit = (e) => { e.preventDefault(); runSearch($("#q").value.trim()); };
  $("#upload").onchange = (e) => runPhoto(e.target.files[0]);
  $("#mine-files").onchange = (e) => importMine(e.target.files);
  $("#mine-folder").onchange = (e) => importMine(e.target.files);
  $("#mine-clear").onclick = async () => {
    if (!confirm("Remove your photos from this device? (They stay on your computer, this only clears the site's copy.)")) return;
    await clearPhotos(); location.reload();
  };
  const drop = $("#mine-drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); importMine(e.dataTransfer.files); });

  new IntersectionObserver((es) => {
    if (es.some((x) => x.isIntersecting) && !$("#view-feed").hidden) loadMore();
  }, { rootMargin: "600px" }).observe($("#more"));

  $("#next-step-x")?.addEventListener("click", finishNextStep);
  $("#next-step-go")?.addEventListener("click", finishNextStep);

  // Arriving from the end of the Uncharted gallery: write from the room you were standing in.
  if (location.hash === "#write") {
    let handoff = null;
    try { handoff = JSON.parse(localStorage.getItem("anotherangle.handoff.v1") || "null"); localStorage.removeItem("anotherangle.handoff.v1"); } catch { /* ignore */ }
    const photo = handoff?.photo ? T.byId.get(handoff.photo) : null;
    const text = handoff?.text ? T.byId.get(handoff.text) : photo ? T.partnerFor(photo) : null;
    if (T.count < PICKS_NEEDED) renderPicks();
    openCoach({ item: photo, partner: text, words: handoff?.words || [], room: handoff?.room || null });
    return;
  }
  if (location.hash === "#coach") { if (T.count < PICKS_NEEDED) renderPicks(); }
  else if (T.count >= PICKS_NEEDED) { show("feed"); resetFeed(); }
  else { renderPicks(); show("start"); }
  // the guided tour: the picking tour on a first visit, or the feed tour if you arrive with picks already
  const onFeed = () => !$("#view-feed").hidden;
  if (location.hash !== "#coach") setupWelcome(onFeed() ? "feed" : "inspiration", { pick: () => (onFeed() ? "feed" : "inspiration") });
}

// After your first picks, once the "reading your taste" moment has finished, tour the feed's own features.
function tourFeedWhenReady() {
  if (tourSeen("feed")) return;
  const wait = setInterval(() => {
    if (!$("#reveal").hidden || !$("#feed .card")) return;
    clearInterval(wait);
    startTour("feed");
  }, 400);
}

main();
