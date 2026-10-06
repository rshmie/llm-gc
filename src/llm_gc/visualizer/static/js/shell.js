// The page around the views: tabs, the guide, the theme, the live indicator.

import { $, fmt, h, storeSet } from "./dom.js";
import { state, ui } from "./state.js";
import { renderChart } from "./engine.js";

export function showTab(name) {
  ui.tab = name;
  const back = $("return-pill");
  if (back && back.dataset.tab === name) back.remove();
  for (const tab of ["conversation", "engine"]) {
    $(`tab-${tab}`).setAttribute("aria-selected", String(tab === name));
    $(`view-${tab}`).hidden = tab !== name || !state.selected;
  }
  storeSet("llmgc.tab", name);
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "engine") renderChart();
}

export function setGuide(open) {
  const drawer = $("guide");
  drawer.classList.toggle("open", open);
  drawer.inert = !open;
  $("backdrop").hidden = !open;
  if (open) $("close-guide").focus();
}

export const THEMES = { system: ["◐", "Theme: follow the system"], light: ["☀", "Theme: light"], dark: ["☾", "Theme: dark"] };
export function applyTheme(theme) {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $("theme").textContent = THEMES[theme][0];
  $("theme").title = `${THEMES[theme][1]} (click to change)`;
  storeSet("llmgc.theme", theme);
  ui.chartKey = null;
}

export function setLive(stateName, text) {
  $("live").dataset.state = stateName;
  $("live-text").textContent = text;
}

// A brief, self-dismissing notice that a collection just ran. The CSS handles not animating its
// entrance and exit under reduced motion; either way it shows for the same few seconds.
export function showCleanupNotice(text) {
  const stack = $("toast-stack");
  if (!stack) return;
  const toast = h("div", { class: "cleanup-toast", role: "status" }, h("span", { class: "dot good" }), text);
  stack.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => { toast.classList.remove("show"); setTimeout(() => toast.remove(), 300); }, 4000);
}

// After a link moves you to another tab, one click takes you back to where you were.
export function offerReturn(tab, scrollY) {
  $("return-pill")?.remove();
  const label = tab === "engine" ? "Engine" : "Conversation";
  const pill = h("button", { id: "return-pill", class: "return-pill", type: "button", "data-tab": tab,
    onclick: () => { showTab(tab); requestAnimationFrame(() => window.scrollTo({ top: scrollY })); } }, `← Back to ${label}`);
  document.body.append(pill);
}

// The settings quoted in the page's own text, filled from the running proxy's config.
export function fillConfigValues(config) {
  for (const el of document.querySelectorAll("[data-config]")) {
    const value = config[el.dataset.config];
    if (typeof value === "number") el.textContent = Number.isInteger(value) ? fmt(value) : value.toFixed(2);
  }
}
