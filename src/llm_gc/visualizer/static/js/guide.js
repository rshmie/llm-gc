// The guide drawer: opened from the ? button, and at one entry from any ⓘ.

import { $ } from "./dom.js";

export function setGuide(open) {
  const drawer = $("guide");
  drawer.classList.toggle("open", open);
  drawer.inert = !open;
  $("backdrop").hidden = !open;
  if (open) $("close-guide").focus();
}

// Opens the guide at one entry of its reference - `term-<key>` or `card-<key>` - and marks it briefly.
export function openReference(id) {
  setGuide(true);
  const entry = $(`ref-${id}`);
  if (!entry) return;
  entry.scrollIntoView({ block: "start" });
  entry.classList.add("flash");
  setTimeout(() => entry.classList.remove("flash"), 1600);
}
