import { dot } from "./data.js";

const $ = (selector) => document.querySelector(selector);

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}

function titleOf(item) {
  return item.title || (item.kind === "photo" ? "Untitled photograph" : "Untitled passage");
}

function creatorOf(item) {
  return item.creator || (item.kind === "text" ? "Author not credited" : "Maker not credited");
}

function workKey(item) {
  return `${item.kind}|${titleOf(item).toLocaleLowerCase()}|${creatorOf(item).toLocaleLowerCase()}`;
}

function excerptOf(text, words = 64) {
  const parts = (text || "").trim().split(/\s+/);
  return parts.length > words ? `${parts.slice(0, words).join(" ")}…` : parts.join(" ");
}

function uniqueWorks(items, count, excluded = new Set()) {
  const seen = new Set(excluded);
  return items.filter((item) => {
    const key = workKey(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, count);
}

function diverse(items, count, scoreOf, excluded = new Set()) {
  const pool = items.filter((item) => !excluded.has(item.id));
  const selected = [];
  while (selected.length < count && pool.length) {
    let bestIndex = 0, bestScore = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      const similarity = selected.length ? Math.max(...selected.map((item) => dot(item.vec, pool[i].vec))) : 0;
      const score = scoreOf(pool[i]) - Math.max(0, similarity) * 0.3;
      if (score > bestScore) { bestScore = score; bestIndex = i; }
    }
    const [item] = pool.splice(bestIndex, 1);
    selected.push(item);
  }
  return selected;
}

export function initGallery(items, model) {
  const track = $("#gallery-track");
  const stage = $("#gallery-stage");
  const journey = $("#gallery-journey");
  const progress = $("#gallery-progress");
  const intro = $("#gallery-intro");
  const tip = $("#gallery-tip");
  const previous = $("#gallery-prev");
  const next = $("#gallery-next");
  const dialog = $("#gallery-detail");
  const detailLayout = $("#gallery-detail-layout");
  if (!track || !stage || !journey || !progress || !intro || !tip || !previous || !next || !dialog || !detailLayout) return;

  if (!model || !items.length) {
    intro.textContent = "The archive couldn't be loaded, so this gallery can't be personalized yet.";
    track.append(node("p", "gallery-empty", "Please reload the page to try again."));
    previous.disabled = true;
    next.disabled = true;
    return;
  }

  let rooms = [];
  let roomIndex = 0;
  let scrollFrame = 0;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function updateIntro() {
    intro.textContent = model.count >= 8
      ? `Built from ${model.count} choices in your inspiration profile. Each room uses the same on-device taste model as your feed.`
      : "This is a preview. Choose at least 8 works in Inspiration to make the gallery reflect your taste.";
  }

  function makeRooms() {
    const liked = [...model.liked].sort((a, b) => model.score(b) - model.score(a));
    const labeled = new Set(model.labels.keys());
    const first = uniqueWorks(liked.length ? liked : model.pick(12), 4);
    const used = new Set([...labeled, ...first.map((item) => item.id)]);
    const usedTitles = new Set(first.map(workKey));
    const echoes = uniqueWorks(model.pick(24, { exclude: used }), 4, usedTitles);
    echoes.forEach((item) => { used.add(item.id); usedTitles.add(workKey(item)); });

    const candidates = items.filter((item) => !used.has(item.id) && !usedTitles.has(workKey(item)));
    const edge = uniqueWorks(diverse(candidates, Math.min(12, candidates.length), (item) => {
      const boundary = 1 - Math.abs(2 * model.prob(item) - 1);
      return model.uncertainty(item) * (0.65 + boundary * 0.35);
    }), 4);

    return [
      {
        title: liked.length ? "The works you brought" : "A first look",
        note: liked.length
          ? "Your picks, returned to you at gallery scale."
          : "Choose works in Inspiration to make this room reflect your taste.",
        works: first,
      },
      {
        title: "Where the echoes lead",
        note: "The Compass finds likely matches, while varying the works so the room doesn't repeat itself.",
        works: echoes,
      },
      {
        title: "At the edge of your taste",
        note: "These works sit near the model's uncertain boundary. Your reaction helps it learn.",
        works: edge,
      },
    ].filter((room) => room.works.length);
  }

  function updateControls() {
    progress.textContent = `room ${String(roomIndex + 1).padStart(2, "0")} / ${String(rooms.length).padStart(2, "0")}`;
    previous.disabled = roomIndex === 0;
    next.disabled = roomIndex >= rooms.length - 1;
  }

  function updateJourney() {
    if (!rooms.length || !stage.offsetHeight) return;
    const stickyTop = parseFloat(getComputedStyle(stage).top) || 67;
    const travel = Math.max(1, journey.offsetHeight - stage.offsetHeight);
    const scrollProgress = Math.max(0, Math.min(1, (stickyTop - journey.getBoundingClientRect().top) / travel));
    const position = scrollProgress * Math.max(0, rooms.length - 1);
    rooms.forEach((room, index) => {
      const element = track.children[index];
      const distance = index - position;
      const proximity = Math.abs(distance);
      const opacity = Math.max(0, 1 - proximity * 1.15);
      const offsetX = Math.max(-140, Math.min(140, distance * 44));
      const depth = -Math.min(proximity * 460, 1100);
      const scale = 1 - Math.min(proximity * 0.12, 0.45);
      element.style.transform = reducedMotion.matches
        ? "none"
        : `translate3d(${offsetX}px, 0, ${depth}px) scale(${scale})`;
      element.style.opacity = String(opacity);
      element.style.visibility = opacity > 0.01 ? "visible" : "hidden";
      element.style.pointerEvents = proximity < 0.5 ? "auto" : "none";
      element.setAttribute("aria-hidden", String(proximity >= 0.5));
      element.querySelectorAll("button").forEach((button) => {
        button.tabIndex = proximity < 0.5 ? 0 : -1;
      });
    });
    roomIndex = Math.round(position);
    updateControls();
  }

  function scheduleJourneyUpdate() {
    if (scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      updateJourney();
    });
  }

  function workButton(item) {
    const title = titleOf(item);
    const button = node("button", `gallery-work${item.kind === "text" ? " is-text" : ""}`);
    button.type = "button";
    button.setAttribute("aria-label", `Learn about ${title}, by ${creatorOf(item)}`);
    const art = node("span", "gallery-art");
    if (item.kind === "photo") {
      const image = node("img");
      image.src = item.img;
      image.alt = title;
      image.loading = "lazy";
      image.referrerPolicy = "no-referrer";
      art.append(image);
    } else {
      art.append(node("span", "gallery-excerpt", excerptOf(item.text, 55)));
    }
    const plaque = node("span", "gallery-plaque");
    plaque.append(node("strong", "", title));
    plaque.append(node("small", "", [creatorOf(item), item.date].filter(Boolean).join(" · ")));
    button.append(art, plaque);
    button.addEventListener("click", () => showDetails(item));
    return button;
  }

  function renderRooms() {
    updateIntro();
    rooms = makeRooms();
    roomIndex = Math.min(roomIndex, Math.max(0, rooms.length - 1));
    journey.style.height = `${Math.max(1, rooms.length) * 140}dvh`;
    track.replaceChildren();
    for (const [index, room] of rooms.entries()) {
      const section = node("section", "gallery-room");
      section.setAttribute("aria-label", `Room ${index + 1}: ${room.title}`);
      section.append(node("span", "gallery-wing left"), node("span", "gallery-wing right"));
      const content = node("div", "gallery-room-content");
      const heading = node("div", "gallery-room-heading");
      heading.append(node("h3", "", room.title), node("p", "", room.note));
      const works = node("div", "gallery-works");
      room.works.forEach((item) => works.append(workButton(item)));
      content.append(heading, works);
      section.append(content);
      track.append(section);
    }
    updateControls();
    scheduleJourneyUpdate();
  }

  function renderFeedback(item, container) {
    const actions = node("div", "gallery-feedback");
    const current = model.labels.get(item.id);
    for (const [label, text] of [[1, "Drawn to this"], [0, "Not for me"]]) {
      const button = node("button", "", text);
      button.type = "button";
      button.setAttribute("aria-pressed", String(current?.y === label && current.w >= 1));
      button.addEventListener("click", () => {
        model.label(item, label);
        const vote = { id: item.id, x: item.x, y: item.y, yv: label, w: 1 };
        const training = window.COMPASS_DATA?.train;
        if (training) {
          const index = training.findIndex((entry) => entry.id === item.id);
          if (index >= 0) training.splice(index, 1, vote);
          else training.push(vote);
        }
        window.dispatchEvent(new CustomEvent("compass:gallery-vote", { detail: vote }));
        tip.textContent = "Your response updated your taste profile on this device.";
        renderRooms();
        showDetails(item);
      });
      actions.append(button);
    }
    container.append(actions);
  }

  function showDetails(item) {
    detailLayout.replaceChildren();
    const layout = node("div", "gallery-detail-layout");
    const artwork = node("div", "gallery-detail-art");
    if (item.kind === "photo") {
      const image = node("img");
      image.src = item.img;
      image.alt = titleOf(item);
      image.referrerPolicy = "no-referrer";
      artwork.append(image);
    } else {
      artwork.append(node("p", "gallery-detail-words", item.text || ""));
    }

    const copy = node("div", "gallery-detail-copy");
    const close = node("button", "gallery-detail-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close artwork details");
    close.addEventListener("click", () => dialog.close());
    copy.append(close, node("div", "tag", item.kind === "photo" ? "photograph" : "writing"));
    copy.append(node("h2", "", titleOf(item)));
    const credits = node("dl", "gallery-credit-list");
    const fields = [
      [item.kind === "text" ? "Author" : "Artist / maker", creatorOf(item)],
      ["Date", item.date || "Date not listed"],
      ["Collection", item.source || "Source not listed"],
    ];
    for (const [label, value] of fields) {
      credits.append(node("dt", "", label), node("dd", "", value));
    }
    copy.append(credits);

    const anchor = model.because(item);
    if (anchor) {
      copy.append(node("p", "", `The Compass places this near “${titleOf(anchor)}” because their image-and-text embeddings are close in its meaning space.`));
    }
    if (item.link) {
      const source = node("a", "", "Open the archive record ↗");
      source.href = item.link;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      copy.append(source);
    } else {
      copy.append(node("p", "", "No external archive record is available for this item."));
    }

    const related = model.partnerFor(item, new Set([item.id]));
    if (related) {
      const section = node("div", "gallery-related");
      section.append(node("small", "", "A meaning-neighbor in the other medium"));
      const button = node("button", "");
      button.type = "button";
      button.append(node("strong", "", titleOf(related)));
      button.append(node("span", "", ` · ${creatorOf(related)}`));
      button.addEventListener("click", () => showDetails(related));
      section.append(button);
      copy.append(section);
    }
    renderFeedback(item, copy);
    layout.append(artwork, copy);
    detailLayout.append(layout);
    if (!dialog.open) dialog.showModal();
  }

  function changeRoom(nextIndex) {
    const targetIndex = Math.max(0, Math.min(rooms.length - 1, nextIndex));
    if (rooms.length < 2) return;
    const stickyTop = parseFloat(getComputedStyle(stage).top) || 67;
    const travel = Math.max(0, journey.offsetHeight - stage.offsetHeight);
    const journeyTop = journey.getBoundingClientRect().top + window.scrollY;
    const start = journeyTop - stickyTop;
    const target = start + travel * targetIndex / (rooms.length - 1);
    window.scrollTo({ top: target, behavior: reducedMotion.matches ? "auto" : "smooth" });
  }

  previous.addEventListener("click", () => changeRoom(roomIndex - 1));
  next.addEventListener("click", () => changeRoom(roomIndex + 1));
  stage.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      changeRoom(roomIndex + 1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      changeRoom(roomIndex - 1);
    }
  });
  let touchStart = null;
  stage.addEventListener("pointerdown", (event) => {
    touchStart = event.target.closest("button") ? null : event.clientX;
  });
  stage.addEventListener("pointerup", (event) => {
    if (touchStart == null) return;
    const delta = event.clientX - touchStart;
    touchStart = null;
    if (Math.abs(delta) > 60) changeRoom(roomIndex + (delta < 0 ? 1 : -1));
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  window.addEventListener("scroll", scheduleJourneyUpdate, { passive: true });
  window.addEventListener("resize", scheduleJourneyUpdate);
  new ResizeObserver(scheduleJourneyUpdate).observe(stage);
  renderRooms();
}
