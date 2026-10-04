// The page around the views: tabs, the guide, the theme, the live indicator.

import { $, storeSet } from "./dom.js";
import { state, ui } from "./state.js";
import { renderChart } from "./engine.js";

export function showTab(name) {
  ui.tab = name;
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
