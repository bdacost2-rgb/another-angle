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

function environmentFor(theme) {
  const name = theme?.n?.toLowerCase() || "";
  if (["pastoral", "cottagecore", "solarpunk", "fairy tale", "folk horror", "southern gothic", "romanticism"].includes(name)) return "meadow";
  if (["maritime", "greek myth"].includes(name)) return "open-water";
  if (["film noir", "cyberpunk", "documentary", "harlem renaissance", "brutalism", "dystopia", "urban decay", "coming of age"].includes(name)) return "city";
  if (["cosmic", "afrofuturism"].includes(name)) return "cosmos";
  if (name === "desert southwest") return "desert";
  if (["surrealism", "magical realism", "vaporwave"].includes(name)) return "dream";
  if (name === "americana") return "open-road";
  return "gallery";
}

const ENVIRONMENT_LABELS = {
  meadow: "MEADOW",
  "open-water": "OPEN WATER",
  city: "CITY",
  cosmos: "COSMOS",
  desert: "DESERT",
  dream: "DREAMSCAPE",
  "open-road": "OPEN ROAD",
  gallery: "GALLERY",
};

export function initGallery(items, model, themes = []) {
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
  let corridorLength = 1;
  let cameraTravel = 1;
  let roomStarts = [];
  let workElements = [];
  let sceneElements = [];
  let lookYaw = 0;
  let lookPitch = 0;
  let activeRoom = -1;
  let positionedWidth = 0;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  tip.textContent = finePointer
    ? "Scroll to walk · move your cursor to look around · select a work."
    : "Scroll to walk · tap a work to learn about it.";

  function updateIntro() {
    if (model.count >= 8 && model.liked.length) {
      intro.textContent = `Built from ${model.count} choices in your inspiration profile. CLIP compares your liked works with 26 curated themes; related interests share a room, and its colors follow the strongest theme.`;
    } else if (model.count >= 8) {
      intro.textContent = `Built from ${model.count} choices in your inspiration profile. Mark a few works as "Drawn to this" to reveal rooms shaped around their closest themes.`;
    } else {
      intro.textContent = "This is a preview. As you choose works in Inspiration, CLIP will compare your picks with 26 curated themes and shape each room around them.";
    }
  }

  function makeRooms() {
    const liked = uniqueWorks([...model.liked].sort((a, b) => model.score(b) - model.score(a)), model.liked.length);
    const labeled = new Set(model.labels.keys());
    const availableThemes = themes.filter((theme) => theme.vec?.length === model.dim);
    const rooms = [];
    let leadTheme = null;

    if (liked.length && availableThemes.length) {
      const environments = new Map();
      for (const item of liked) {
        let closest = null, similarity = -Infinity;
        for (const theme of availableThemes) {
          const value = dot(item.vec, theme.vec);
          if (value > similarity) { closest = theme; similarity = value; }
        }
        if (!closest) continue;
        const environment = environmentFor(closest);
        if (!environments.has(environment)) environments.set(environment, { environment, themes: new Map(), affinity: 0 });
        const group = environments.get(environment);
        if (!group.themes.has(closest.n)) group.themes.set(closest.n, { theme: closest, works: [] });
        group.themes.get(closest.n).works.push(item);
        group.affinity += similarity;
      }

      const themedRooms = [...environments.values()].map((group) => {
        const themeGroups = [...group.themes.values()].sort((a, b) => b.works.length - a.works.length);
        const dominant = themeGroups[0];
        const works = themeGroups.flatMap(({ works: themeWorks }) => themeWorks.sort((a, b) => model.score(b) - model.score(a)));
        const names = themeGroups.map(({ theme }) => theme.n);
        return {
          environment: group.environment,
          theme: dominant.theme,
          title: dominant.theme.n,
          note: `${works.length} ${works.length === 1 ? "work" : "works"} you chose · ${names.join(" / ")}. ${dominant.theme.desc}.`,
          works,
          affinity: group.affinity / works.length,
        };
      }).sort((a, b) => b.works.length - a.works.length || b.affinity - a.affinity);

      leadTheme = themedRooms[0]?.theme || null;
      rooms.push(...themedRooms);
    } else {
      const first = uniqueWorks(liked.length ? liked : model.pick(12), 4);
      rooms.push({
        title: liked.length ? "The works you brought" : "A first look",
        note: liked.length
          ? "Your picks, returned to you at gallery scale."
          : "Choose works in Inspiration to make this room reflect your taste.",
        works: first,
      });
    }

    const used = new Set([...labeled, ...liked.map((item) => item.id)]);
    const usedTitles = new Set(liked.map(workKey));
    const echoes = uniqueWorks(model.pick(24, { exclude: used }), 4, usedTitles);
    echoes.forEach((item) => { used.add(item.id); usedTitles.add(workKey(item)); });
    rooms.push({
      title: "Where the echoes lead",
      note: "The Compass finds likely matches, while varying the works so the room doesn't repeat itself.",
      works: echoes,
      theme: leadTheme,
      environment: leadTheme ? environmentFor(leadTheme) : undefined,
    });

    const candidates = items.filter((item) => !used.has(item.id) && !usedTitles.has(workKey(item)));
    const edge = uniqueWorks(diverse(candidates, Math.min(12, candidates.length), (item) => {
      const boundary = 1 - Math.abs(2 * model.prob(item) - 1);
      return model.uncertainty(item) * (0.65 + boundary * 0.35);
    }), 4);
    rooms.push({
      title: "At the edge of your taste",
      note: "These works sit near the model's uncertain boundary. Your reaction helps it learn.",
      works: edge,
      theme: leadTheme,
      environment: leadTheme ? environmentFor(leadTheme) : undefined,
    });
    return rooms.filter((room) => room.works.length);
  }

  function applyRoomAtmosphere() {
    const room = rooms[roomIndex];
    if (!room || activeRoom === roomIndex) return;
    activeRoom = roomIndex;
    const palette = room.theme?.pal?.filter((color) => /^#[0-9a-f]{6}$/i.test(color)) || [];
    const colors = palette.length >= 4 ? palette : ["#292523", "#51463b", "#c9b48f", "#171619"];
    track.dataset.environment = room.environment || environmentFor(room.theme);
    stage.dataset.environment = track.dataset.environment;
    track.style.setProperty("--scene-primary", colors[0]);
    track.style.setProperty("--scene-secondary", colors[1]);
    track.style.setProperty("--scene-accent", colors[2]);
    track.style.setProperty("--scene-dark", colors[3]);
    const rgba = (color, opacity) => `rgba(${[1, 3, 5].map((index) => parseInt(color.slice(index, index + 2), 16)).join(", ")}, ${opacity})`;
    stage.style.setProperty("--scene-glow", rgba(colors[0], 0.22));
    stage.style.setProperty("--scene-horizon", rgba(colors[2], 0.08));
  }

  function positionWorks() {
    const width = stage.clientWidth;
    if (width === positionedWidth) return;
    positionedWidth = width;
    const sideOffset = width < 820 ? width * 0.15 : Math.min(330, width * 0.24);
    workElements.forEach((element) => {
      const side = Number(element.dataset.side);
      const depth = Number(element.dataset.depth);
      element.style.transform = `translate3d(${side * sideOffset}px, -38px, ${-depth}px) rotateY(${side < 0 ? 56 : -56}deg)`;
    });
  }

  function updateControls() {
    progress.textContent = `room ${String(roomIndex + 1).padStart(2, "0")} / ${String(rooms.length).padStart(2, "0")}`;
    previous.disabled = roomIndex === 0;
    next.disabled = roomIndex >= rooms.length - 1;
  }

  function updateJourney() {
    if (!rooms.length || !stage.offsetHeight) return;
    positionWorks();
    const stickyTop = parseFloat(getComputedStyle(stage).top) || 67;
    const travel = Math.max(1, journey.offsetHeight - stage.offsetHeight);
    const scrollProgress = Math.max(0, Math.min(1, (stickyTop - journey.getBoundingClientRect().top) / travel));
    const cameraDistance = scrollProgress * cameraTravel;
    track.style.transform = `rotateY(${lookYaw}deg) rotateX(${lookPitch}deg) translateZ(${cameraDistance}px)`;
    sceneElements.forEach(({ element, depth, fadeStart, fadeEnd }) => {
      const distance = depth - cameraDistance;
      const fadeIn = Math.max(0, Math.min(1, (fadeStart - distance) / (fadeStart - fadeEnd)));
      const fadeOut = Math.max(0, Math.min(1, (distance + 100) / 400));
      const opacity = Math.min(fadeIn, fadeOut);
      element.style.opacity = String(opacity);
      element.style.visibility = opacity > 0.01 ? "visible" : "hidden";
    });
    workElements.forEach((element) => {
      const distance = Number(element.dataset.depth) - cameraDistance;
      const fadeIn = Math.max(0, Math.min(1, (2100 - distance) / 500));
      const fadeOut = Math.max(0, Math.min(1, (distance + 150) / 150));
      const opacity = Math.min(fadeIn, fadeOut);
      const visible = opacity > 0.1;
      element.style.opacity = String(opacity);
      element.style.visibility = visible ? "visible" : "hidden";
      element.style.pointerEvents = visible && distance > 100 && distance < 2400 ? "auto" : "none";
      element.setAttribute("aria-hidden", String(!visible));
      element.tabIndex = visible && distance < 2400 ? 0 : -1;
    });
    roomIndex = roomStarts.reduce((current, start, index) => cameraDistance + 1 >= start ? index : current, 0);
    applyRoomAtmosphere();
    updateControls();
  }

  function scheduleJourneyUpdate() {
    if (scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      updateJourney();
    });
  }

  function workButton(item, depth, side) {
    const title = titleOf(item);
    const button = node("button", `gallery-work${item.kind === "text" ? " is-text" : ""}`);
    button.type = "button";
    button.setAttribute("aria-label", `Learn about ${title}, by ${creatorOf(item)}`);
    button.dataset.depth = String(depth);
    button.dataset.side = String(side);
    button.tabIndex = -1;
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
    roomStarts = [];
    workElements = [];
    sceneElements = [];
    positionedWidth = 0;
    track.replaceChildren();
    const leftWall = node("div", "gallery-wall left");
    const rightWall = node("div", "gallery-wall right");
    const floor = node("div", "gallery-floor");
    const ceiling = node("div", "gallery-ceiling");
    track.append(leftWall, rightWall, floor, ceiling);

    let depth = 0;
    let workIndex = 0;
    for (const [index, room] of rooms.entries()) {
      const roomStart = depth;
      roomStarts.push(roomStart);
      if (index > 0) {
        const portal = node("div", "gallery-portal");
        portal.style.left = "-330px";
        portal.style.top = "-250px";
        portal.style.transform = `translateZ(${-roomStart}px)`;
        sceneElements.push({ element: portal, depth: roomStart, fadeStart: 3000, fadeEnd: 800 });
        track.append(portal);
      }

      const sign = node("div", "gallery-room-sign");
      sign.style.left = "-210px";
      sign.style.top = "-160px";
      sign.style.transform = `translate3d(0, 0, ${-(roomStart + 220)}px)`;
      sceneElements.push({ element: sign, depth: roomStart + 220, fadeStart: 1500, fadeEnd: 700 });
      sign.append(
        node("small", "", `${ENVIRONMENT_LABELS[room.environment || environmentFor(room.theme)]} · ROOM ${String(index + 1).padStart(2, "0")}`),
        node("strong", "", room.title),
        node("span", "", room.note),
      );
      track.append(sign);

      room.works.forEach((item, itemIndex) => {
        const side = workIndex % 2 === 0 ? -1 : 1;
        const artwork = workButton(item, roomStart + 650 + itemIndex * 500, side);
        track.append(artwork);
        workElements.push(artwork);
        workIndex += 1;
      });
      depth += Math.max(room.works.length * 500 + 550, 2400);
    }
    corridorLength = Math.max(1200, depth + 800);
    cameraTravel = Math.max(1, corridorLength - 1000);
    const surfaceLength = Math.max(8600, corridorLength + 800);
    leftWall.style.width = rightWall.style.width = `${surfaceLength}px`;
    leftWall.style.left = `${-350 - surfaceLength / 2}px`;
    rightWall.style.left = `${350 - surfaceLength / 2}px`;
    leftWall.style.transform = `translateZ(${-(surfaceLength / 2 - 400)}px) rotateY(90deg)`;
    rightWall.style.transform = `translateZ(${-(surfaceLength / 2 - 400)}px) rotateY(-90deg)`;
    floor.style.height = ceiling.style.height = `${surfaceLength}px`;
    const farWall = node("div", "gallery-far-wall");
    farWall.style.transform = `translateZ(${-corridorLength}px)`;
    track.append(farWall);
    journey.style.height = `calc(${cameraTravel}px + 100dvh)`;
    activeRoom = -1;
    applyRoomAtmosphere();
    positionWorks();
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
    const target = start + travel * roomStarts[targetIndex] / cameraTravel;
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
  stage.addEventListener("pointermove", (event) => {
    if (event.pointerType !== "mouse" || !finePointer || reducedMotion.matches) return;
    const bounds = stage.getBoundingClientRect();
    lookYaw = (0.5 - (event.clientX - bounds.left) / bounds.width) * 7;
    lookPitch = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
    scheduleJourneyUpdate();
  });
  stage.addEventListener("pointerleave", () => {
    lookYaw = 0;
    lookPitch = 0;
    scheduleJourneyUpdate();
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  window.addEventListener("scroll", scheduleJourneyUpdate, { passive: true });
  window.addEventListener("resize", scheduleJourneyUpdate);
  new ResizeObserver(scheduleJourneyUpdate).observe(stage);
  renderRooms();
}
