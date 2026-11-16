import { loadArchive, centerQuery } from "./data.js";
import { TasteModel } from "./taste.js";
import { Coach } from "./coach.js";
import * as models from "./models.js";
import { addPhotos, loadPhotos, asItems, mergeInto, clearPhotos } from "./mine.js";
import { initLens, setWords } from "./lens.js";

const $ = (s) => document.querySelector(s);
const ICON = {
  heart: '<path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"/>',
  less: '<path d="M6 6l12 12M18 6L6 18"/>',
  similar: '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
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

let A, T, coach;
const shown = new Set();
let mode = { kind: "taste" }; // or {kind:"similar", item} | {kind:"search", q, vec} | {kind:"photo", url, vec}
const PICKS_NEEDED = 8;

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
function renderPicks() {
  const grid = $("#pick-grid");
  grid.replaceChildren();
  const photos = A.items.filter((i) => i.kind === "photo" && !i.mine);
  const texts = A.items.filter((i) => i.kind === "text");
  const sample = (arr, n) => [...arr].sort(() => Math.random() - 0.5).slice(0, n);
  // A spread of the archive: spaced apart so the first picks are informative.
  pickSet = spread([...sample(photos, 120)], 16).concat(spread(sample(texts, 60), 8)).sort(() => Math.random() - 0.5);
  for (const it of pickSet) {
    const picked = T.labels.get(it.id)?.y === 1;
    const b = el("button", { class: "pick", "aria-pressed": String(picked), onclick: () => {
      b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
      updatePickCount();
    } },
    it.kind === "photo" ? el("img", { src: it.img, alt: it.title || "Archive photo", loading: "lazy" }) : setWords(el("span", { class: "txt" }), it.text));
    b.dataset.id = it.id;
    grid.append(b);
  }
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
  const total = selectedPicks().length + countEarlier();
  $("#pick-count").textContent = `${total} of ${PICKS_NEEDED} picked`;
  $("#pick-done").disabled = total < PICKS_NEEDED;
}
function countEarlier() {
  // likes from earlier rounds of "Show different ones" that are not on screen
  const onScreen = new Set(pickSet.map((i) => i.id));
  return [...T.labels].filter(([id, l]) => l.y === 1 && l.w >= 1 && !onScreen.has(id)).length;
}
function commitPicks() {
  const chosen = new Set(selectedPicks());
  for (const it of pickSet) {
    if (chosen.has(it.id)) T.labels.set(it.id, { y: 1, w: 1 });
    else if (!T.labels.has(it.id) || T.labels.get(it.id).y === 1) T.labels.set(it.id, { y: 0, w: 0.3 }); // seen but not picked: a weak "not this"
  }
  T.fit(); T.save();
}

// ---------- Feed ----------
function resetFeed() {
  shown.clear();
  $("#feed").replaceChildren();
  renderMode();
  loadMore(12);
}

function renderMode() {
  const m = $("#mode");
  m.replaceChildren();
  const clear = el("button", { "aria-label": "Back to your feed", onclick: () => { mode = { kind: "taste" }; resetFeed(); } }, icon("close"));
  if (mode.kind === "taste") {
    m.append(el("span", {}, T.count ? `For you, learned from ${T.count} choices` : "Exploring"));
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
  const leads = [
    ...T.pick(nPhoto, { kind: "photo", exclude: shown, anchor: anchor() }),
    ...T.pick(n - nPhoto, { kind: "text", exclude: shown, anchor: anchor() }),
  ].sort(() => Math.random() - 0.5);
  if (!leads.length) { $("#more").textContent = "That's everything in the archive for now."; return; }
  for (const lead of leads) {
    shown.add(lead.id);
    const partner = T.partnerFor(lead, shown);
    if (partner) shown.add(partner.id);
    feed.append(pairCard(lead, partner));
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
    } }, icon("heart"));
  return b;
}

function pairCard(lead, partner) {
  const card = el("article", { class: "card" });
  const parts = lead.kind === "photo" ? [lead, partner] : [partner, lead];
  const [photo, text] = parts;
  if (photo) {
    card.append(el("img", { class: "photo", src: photo.img, alt: photo.title || "Archive photo", loading: "lazy" }), credit(photo));
    if (photo.mine) card.append(el("span", { class: "badge" }, "Your photo"));
  }
  if (text) card.append(passage(text, lead.kind === "text"), credit(text));
  const why = mode.kind === "taste" && T.because(lead);
  if (why) card.append(el("div", { class: "because" }, "Because you liked ", why.kind === "photo" ? `"${(why.title || "a photo").slice(0, 50)}"` : `a passage from ${why.title || why.creator}`));
  card.append(el("div", { class: "actions" },
    photo ? heartBtn(photo) : null,
    text ? heartBtn(text) : null,
    el("span", { class: "grow" }),
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
  box.replaceChildren();
  if (!liked.length) { box.append(el("p", { class: "empty" }, "Nothing saved yet. Tap the heart on anything in your feed.")); return; }
  for (const it of liked.reverse()) {
    const card = el("article", { class: "card" });
    if (it.kind === "photo") card.append(el("img", { class: "photo", src: it.img, alt: it.title || "", loading: "lazy" }), credit(it));
    else card.append(passage(it), credit(it));
    card.append(el("div", { class: "actions" }, heartBtn(it), el("span", { class: "grow" }),
      el("button", { onclick: () => { mode = { kind: "similar", item: it }; show("feed"); resetFeed(); } }, icon("similar"), "More like this")));
    box.append(card);
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
  box.replaceChildren();
  if (!mine.length) { box.append(el("p", { class: "empty" }, "Your photos will appear here.")); return; }
  // Rediscover: your own photos, ranked by your taste model, so the strong ones you forgot come back first.
  const ranked = T.count ? [...mine].sort((a, b) => T.score(b) - T.score(a)) : [...mine].sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  $("#mine-note").textContent = T.count
    ? "Ranked by your taste: the ones near the top match what you love most. Some you may have forgotten."
    : "Pick some favorites first (Pick again) and these will be ranked by your taste.";
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
    box.append(card);
  }
}

// ---------- Coach ----------
function openCoach() {
  show("coach");
  if (!coach) {
    coach = new Coach({
      embed: (list) => models.sentences(list),
      inspiration: () => {
        if (!T) return null; // the coach also works on its own, before the archive has loaded
        const lead = T.pick(1, { kind: "photo", exclude: shown })[0];
        return lead ? { item: lead, partner: T.partnerFor(lead) } : null;
      },
      say: renderMsg,
    });
    coach.start();
  }
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
      el("img", { src: item.img, alt: item.title || "" }),
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
  // The writing coach doesn't need the archive, so its tab is wired up first and works either way.
  $("#tab-coach").addEventListener("click", (e) => { e.preventDefault(); openCoach(); });
  window.addEventListener("hashchange", () => { if (location.hash === "#coach") openCoach(); });
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
  updateSaved();

  $("#pick-done").onclick = () => { commitPicks(); pickSet = []; updateSaved(); mode = { kind: "taste" }; show("feed"); resetFeed(); };
  $("#pick-more").onclick = () => { commitPicks(); renderPicks(); };
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

  if (location.hash === "#coach") { if (T.count < PICKS_NEEDED) renderPicks(); }
  else if (T.count >= PICKS_NEEDED) { show("feed"); resetFeed(); }
  else { renderPicks(); show("start"); }
}

main();
