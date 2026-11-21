// What you're here for: photos, writing, or both. Shared by every page (the inspiration feed keeps
// the same setting), remembered on this device. Uncharted's gallery and checkpoints listen for changes.
const KEY = "anotherangle.focus.v1";
const VALID = ["photo", "both", "text"];

export function getFocus() {
  try {
    const v = localStorage.getItem(KEY);
    return VALID.includes(v) ? v : "both";
  } catch {
    return "both";
  }
}

// Does an archive item belong in what you asked to see?
export const fitsFocus = (item, focus = getFocus()) => focus === "both" || item.kind === focus;

export function setFocus(next) {
  if (!VALID.includes(next)) return;
  try { localStorage.setItem(KEY, next); } catch { /* private mode: it just isn't remembered */ }
  syncFocusButtons();
  window.dispatchEvent(new CustomEvent("focus:change", { detail: { focus: next } }));
}

export function syncFocusButtons() {
  const focus = getFocus();
  document.querySelectorAll("[data-focus]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.focus === focus)));
}

export function wireFocusButtons() {
  document.querySelectorAll("[data-focus]").forEach((b) => b.addEventListener("click", () => setFocus(b.dataset.focus)));
  syncFocusButtons();
}
