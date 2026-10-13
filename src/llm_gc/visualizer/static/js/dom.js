// Building DOM safely, and formatting numbers and times.
//
// Conversation text is untrusted, so it only ever enters the DOM as text.

// ----------------------------------------------------------------- DOM helpers

// Absent parts (null, undefined, false) are skipped and nested arrays flattened.
// The DOM's own append and replaceChildren would print them as "null" and
// "[object HTMLDivElement]", so children are only ever set through these helpers.
export function nodes(children) {
  return children.flat(Infinity).filter((child) => child !== null && child !== undefined && child !== false)
    .map((child) => child instanceof Node ? child : document.createTextNode(String(child)));
}
export function setChildren(target, ...children) { target.replaceChildren(...nodes(children)); }
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style") el.style.cssText = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  setChildren(el, ...children);
  return el;
}
export const SVG_NS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== null && value !== undefined) el.setAttribute(key, value);
  for (const child of nodes(children)) el.appendChild(child);
  return el;
}
export const $ = (id) => document.getElementById(id);
export const fill = (id, ...children) => setChildren($(id), ...children);

export const fmt = (n) => Number(n).toLocaleString();
export const plural = (n, word, many = `${word}s`) => `${fmt(n)} ${n === 1 ? word : many}`;
export const percent = (part, whole) => whole > 0 ? Math.round((part / whole) * 100) : 0;
// A share to show: a part that is not zero never reads as 0%.
export const share = (part, whole) => (part > 0 && percent(part, whole) === 0 ? "<1" : String(percent(part, whole)));
export function compact(n) {
  if (n >= 1e6) return `${+(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${+(n / 1e3).toFixed(0)}k`;
  if (n >= 1e3) return `${+(n / 1e3).toFixed(1)}k`;
  return String(n);
}
export function ranges(numbers) {
  const sorted = [...new Set(numbers)].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < sorted.length; i++) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    out.push(j > i ? `${sorted[i]}–${sorted[j]}` : `${sorted[i]}`);
    i = j;
  }
  return out.join(", ");
}
export function ago(seconds) {
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}
export const agoTs = (ts) => ago(Date.now() / 1000 - ts);
export const agoIso = (iso) => ago((Date.now() - Date.parse(iso)) / 1000);

export function storeGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
export function storeSet(key, value) { try { localStorage.setItem(key, value); } catch { /* a per-viewer convenience only */ } }

// ----------------------------------------------------------------- motion: numbers that count up, meters that fill

export const reducedMotion = () => { try { return matchMedia("(prefers-reduced-motion: reduce)").matches; } catch { return false; } };

const numPrev = new Map();  // key -> last value shown, so a redrawn number can tween from it instead of jumping
const meterEls = new Map(); // key -> the one <div> that fills a meter, reused so its width transitions

// A <span> holding a number that eases from whatever it last showed (under this key) to `value`,
// instead of jumping, when the value actually changed. First draw and reduced motion skip the tween.
export function countEl(key, value, { format = fmt } = {}) {
  const span = document.createElement("span");
  span.className = "num";
  const prev = numPrev.get(key);
  const animate = prev !== undefined && prev !== value && !reducedMotion();
  span.textContent = format(animate ? prev : value);
  numPrev.set(key, value);
  if (animate) {
    const from = prev, to = value, start = performance.now(), duration = 450;
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - t) ** 3;
      span.textContent = format(Math.round(from + (to - from) * eased));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  return span;
}

// The fill of a meter, as the same element every time it is drawn under this key: changing its
// width, rather than recreating it, is what lets the CSS transition on it actually play.
export function meterBar(key, pct, extraClass = "") {
  let el = meterEls.get(key);
  if (!el) { el = document.createElement("div"); meterEls.set(key, el); }
  const some = pct > 0 ? "some" : "";
  el.className = [some, extraClass].filter(Boolean).join(" ");
  el.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  return el;
}

// Forgets every tracked number and meter: called when the selected conversation changes, so a
// value from the previous one never gets treated as "before" for the newly selected one.
export function clearMotionCaches() { numPrev.clear(); meterEls.clear(); }
