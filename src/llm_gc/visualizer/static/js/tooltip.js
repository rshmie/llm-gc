// The one tooltip, and the ⓘ explanations that use it.

import { GLOSSARY } from "./vocabulary.js";
import { $, h, setChildren } from "./dom.js";
import { openReference } from "./guide.js";

export function placeTooltip(rect) {
  const tip = $("tooltip");
  tip.hidden = false;
  tip.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - tip.offsetWidth - 8))}px`;
  const below = rect.bottom + 10 + tip.offsetHeight < window.innerHeight;
  tip.style.top = `${below ? rect.bottom + 10 : Math.max(8, rect.top - tip.offsetHeight - 10)}px`;
}
export function tooltipContent(title, rows) {
  setChildren($("tooltip"), h("div", { class: "tip-title" }, title),
    rows.filter((row) => row !== null && row !== undefined && row !== false)
      .map((row) => row instanceof Node ? row : h("div", { class: "tip-row" }, row)));
}
export function showTooltip(event, title, ...rows) {
  tooltipContent(title, rows);
  placeTooltip(event.currentTarget.getBoundingClientRect());
}
export const hideTooltip = () => { $("tooltip").hidden = true; };
export const tipQuote = (text) => text ? h("div", { class: "tip-quote" }, text) : null;
// What clicking does, as the last line of a tooltip.
export const tipAction = (text) => h("div", { class: "tip-action" }, text, h("span", { class: "tip-arrow", "aria-hidden": "true" }, " →"));
export const hoverTip = (title, ...rows) => ({ onpointerenter: (e) => showTooltip(e, title, ...rows), onpointerleave: hideTooltip });
// A phrase on the page with more behind it: underlined with dots, the rest on hover or focus.
export const hint = (text, title, ...rows) => h("span", { class: "hint", tabindex: "0", ...hoverTip(title, ...rows),
  onfocus: (e) => showTooltip(e, title, ...rows), onblur: hideTooltip }, text);

export function info(term) {
  const button = h("button", { class: "info", type: "button", "aria-label": `What is ${GLOSSARY[term][0]}?` }, "i");
  explain(button, `term-${term}`, GLOSSARY[term]);
  return button;
}

// An ⓘ shows the short meaning on hover or focus; a click opens the guide at the full one.
export function explain(button, id, [title, short, more]) {
  const rows = [short, more ? tipAction("Click for the full explanation") : null];
  button.addEventListener("pointerenter", (e) => showTooltip(e, title, ...rows));
  button.addEventListener("pointerleave", hideTooltip);
  button.addEventListener("focus", (e) => showTooltip(e, title, ...rows));
  button.addEventListener("blur", hideTooltip);
  button.addEventListener("click", () => { hideTooltip(); openReference(id); });
}
