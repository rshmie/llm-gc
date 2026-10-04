// The one tooltip, and the ⓘ explanations that use it.

import { GLOSSARY } from "./vocabulary.js";
import { $, h, setChildren } from "./dom.js";

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
export const hoverTip = (title, ...rows) => ({ onpointerenter: (e) => showTooltip(e, title, ...rows), onpointerleave: hideTooltip });

export function info(term) {
  const [title, text] = GLOSSARY[term];
  return h("button", { class: "info", type: "button", "aria-label": `What is ${title}?`,
    onpointerenter: (e) => showTooltip(e, title, text), onpointerleave: hideTooltip,
    onfocus: (e) => showTooltip(e, title, text), onblur: hideTooltip }, "i");
}
