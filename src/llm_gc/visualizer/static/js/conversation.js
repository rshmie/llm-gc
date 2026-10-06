// The conversation view: what the model knows, exchange by exchange and message by message.
//
// Every value comes from an endpoint; this view formats and explains, and derives
// nothing beyond grouping, counting, and sums and differences of numbers it was given.

import { BECOMES, CATS, CLAMP_CHARS, EARLIER_SHOWN, FACTS_SHOWN, GLOSSARY, HEIGHT, MIN_BAR_PX, ORDER, RUN_MIN, categoryOf, describeOutcome, swatch } from "./vocabulary.js";
import { $, agoTs, countEl, fill, fmt, h, meterBar, percent, plural, ranges, reducedMotion, setChildren, storeGet, storeSet, svg } from "./dom.js";
import { collecting, countCats, factsByTurn, has, instructionTokens, marked, newestReply, pendingTurns, preview, query, recapLimit, state, turnMatches, ui, writtenTokens } from "./state.js";
import { hideTooltip, info, showTooltip, tipQuote, hoverTip } from "./tooltip.js";
import { offerReturn, showTab } from "./shell.js";
import { openCard, showWhy, whyText } from "./engine.js";

const smooth = () => (reducedMotion() ? "auto" : "smooth");
export const speakerName = (role) => role === "user" ? "You" : role ? role[0].toUpperCase() + role.slice(1) : "—";
const avatarText = (role) => role === "user" ? "You" : role === "assistant" ? "AI" : (role || "?").slice(0, 2).toUpperCase();
const avatar = (role) => h("div", { class: `av ${role === "user" ? "user" : "assistant"}`, "aria-hidden": "true" }, avatarText(role));
// A turn of tool results is the client reporting what its tools returned, under the user role.
const speakerOf = (turn) => turn.tool_results ? "Tool results" : speakerName(turn.role);
const avatarOf = (turn) => turn.tool_results ? h("div", { class: "av user", "aria-hidden": "true" }, "Tool") : avatar(turn.role);
const toolTip = () => hoverTip(GLOSSARY.tools[0], GLOSSARY.tools[1]);
// The speaker, explained when it is the client reporting tool output; a tag when the assistant calls a tool.
const who = (turn) => h("span", { class: "who", ...(turn.tool_results ? toolTip() : {}) }, speakerOf(turn));
const toolTag = (label) => h("span", { class: "tag", ...toolTip() }, label);
const callsTag = (turn) => turn.tool_calls ? toolTag("calls a tool") : null;
// Instructions the client puts inside the message: never condensed, so their size stands apart.
function instructionsTag(turn) {
  const tokens = instructionTokens(turn.turn_index);
  return tokens ? h("span", { class: "tag", ...hoverTip(GLOSSARY.instructions[0], GLOSSARY.instructions[1]) },
    `+ client instructions · ${plural(tokens, "token")}`) : null;
}
const changes = (n) => n === 1 ? "1 message changes" : `${fmt(n)} messages change`;
const changing = (k) => k.turns.filter((t) => t.becomes).length;
export const messagesLabel = (turns) => turns.length === 1 ? `message ${turns[0]}` : `messages ${ranges(turns)}`;
// The colour of a level; out of view is drawn as a warning outline rather than a fill.
const levelColour = (cat) => cat === "out_of_view" ? "var(--warn)" : `var(--fid-${cat})`;

export function renderHero(k) {
  renderSize(k);
  renderHolds(k);
  renderWindow(k);
  renderMap(k);
}

// ----------------------------------------------------------------- what needs attention

// Only what needs a look gets a banner: what the model can no longer see, and a collection that fell
// short. Everything else is said in the panel it is about. A warning can be dismissed, and comes
// back only when there is more of what it warns about.
const dismissedKey = () => `llmgc.dismissed.${state.selected}`;
function dismissed() {
  try { return JSON.parse(storeGet(dismissedKey()) || "{}"); } catch { return {}; }
}
function dismiss(kind, upTo) {
  storeSet(dismissedKey(), JSON.stringify({ ...dismissed(), [kind]: upTo }));
  renderStatus();
}

export function renderStatus() {
  const k = state.knowledge;
  if (!k) return;
  const unseen = countCats(k).out_of_view;
  const folded = k.spans.filter((s) => s.level === "out_of_view").length;
  const shortfalls = state.activity ? state.activity.recent_degradations : [];
  const latestShortfall = shortfalls.length ? shortfalls.at(-1).occurred_at : 0;
  const hidden = dismissed();
  const toMessages = (filter) => () => { setFilter(filter); $("messages").scrollIntoView({ behavior: smooth(), block: "start" }); };
  const line = (tone, icon, title, text, action, dismissal) => h("div", { class: `status-line tone-${tone}` },
    h("span", { class: "badge", "aria-hidden": "true" }, icon),
    h("span", { class: "msg-text" }, h("b", {}, title), text ? ` ${text}` : null),
    action ? h("button", { class: "link-btn", type: "button", onclick: action[1] }, `${action[0]} →`) : null,
    dismissal ? h("button", { class: "x", type: "button", "aria-label": "Dismiss", title: "Hide until there is more of this",
      onclick: () => dismiss(...dismissal) }, "×") : null);
  const items = [];
  if (unseen > (hidden.out_of_view ?? 0)) items.push(line("warn", "!", `The model can no longer see ${plural(unseen, "message")}.`,
    `The recap reached its budget of ${recapLimit()} with every exchange at its least detail, so the oldest ${plural(folded, "exchange")} were folded into one line saying they exist. Their words are stored, not lost.`,
    ["Show them", toMessages("out_of_view")], ["out_of_view", unseen]));
  if (latestShortfall > (hidden.shortfalls ?? 0)) items.push(line("warn", "!", `${plural(shortfalls.length, "part")} of a collection fell short.`,
    "Each says what the model gets instead. Nothing is lost: an exchange always keeps its words.",
    ["See why", () => { showTab("engine"); openCard("notmoved", true); }], ["shortfalls", latestShortfall]));
  if (collecting()) items.push(line("", "…", "Collecting this conversation.", "The page updates when it finishes."));
  fill("status", items);
}

// ----------------------------------------------------------------- the prompt: cost, holdings, window

export function renderSize(k) {
  const written = writtenTokens(k);
  const sent = k.context_tokens;
  const saved = written - sent;
  const [tone, text] = k.cuts === 0 ? ["", "Same as written: nothing collected yet"]
    : saved >= 0 ? ["good", `↓ ${fmt(saved)} saved · ${percent(saved, written)}% less than as written`]
    : ["", `↑ ${fmt(-saved)} more than as written: the recap outgrew what it replaced`];
  fill("panel-size",
    h("div", { class: "label" }, "Sent with each request", info("size")),
    h("div", { class: "hero-row" }, h("span", { class: "hero" }, countEl("size-sent", sent)), h("span", { class: "hero-unit" }, "tokens"),
      h("span", { class: `delta${tone ? ` ${tone}` : ""}` }, text)),
    h("p", { class: "hero-sub" }, "The recap, then the recent messages as written. Your next message is added on top."));
}

export function renderHolds(k) {
  const counts = countCats(k);
  const next = changing(k);
  const why = h("button", { class: "link-btn", type: "button", onclick: () => { showTab("engine"); openCard("rate", true); } }, "Why?");
  const sofar = `${plural(k.turns.length, "message")} so far, ${!k.cuts ? "all sent as written." : counts.out_of_view ? "some out of view." : "none out of view."}`;
  const rows = ORDER.filter((key) => key !== "out_of_view" || counts.out_of_view).map((key) => h("li",
    { class: key === "out_of_view" ? "alert" : counts[key] ? "" : "zero", ...hoverTip(GLOSSARY[key][0], GLOSSARY[key][1]) },
    h("i", { class: `hb f-${key}`, style: `height:${Math.max(3, Math.round(HEIGHT[key] * 0.13))}px` }),
    h("span", {}, CATS[key].label), h("b", {}, countEl(`holds-${key}`, counts[key]))));
  fill("panel-holds",
    h("div", { class: "label" }, "How each message is sent", info("holds")),
    h("ul", { class: "holds" }, rows),
    h("div", { class: "note" }, sofar,
      next ? h("div", {}, k.next_cut.batch_ready ? `A collection runs after your next reply: ${changes(next)}. ` : `${changes(next)} at the next collection. `, why) : null));
}

export function renderWindow(k) {
  const w = state.config ? state.config.context_window : null;
  const share = w ? k.context_tokens / w : 0;
  fill("panel-window",
    h("div", { class: "label" }, "The model's window", info("window")),
    h("div", { class: "tile-big" }, w ? [countEl("window-pct", share * 100, { format: (v) => (v < 0.1 && v > 0 ? "<0.1" : v.toFixed(v < 10 ? 1 : 0)) }), "%"] : "—", h("small", {}, " used")),
    h("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": String(w || 0), "aria-valuenow": String(k.context_tokens), "aria-label": "Share of the model's window" },
      meterBar("window-share", share * 100)),
    h("div", { class: "note" }, w ? `${fmt(k.context_tokens)} of ${fmt(w)} tokens, before your next message` : "window size not reported"));
}

// ----------------------------------------------------------------- the map: what the model has of each message

// One bar per message, reused across renders and keyed by message: a collection changes a bar's
// colour and height in place, which is what lets the change transition instead of jumping.
const barCache = new Map();
let lastCats = new Map();
let ledgerSegs = null;
export function clearMapCache() { barCache.clear(); lastCats = new Map(); ledgerSegs = null; }

// Appends what is new instead of re-inserting what is already there: an element taken out of the
// page and put back loses its running transition.
function reconcile(host, nodes) {
  const current = [...host.children];
  if (current.length <= nodes.length && current.every((node, i) => node === nodes[i])) {
    for (const node of nodes.slice(current.length)) host.appendChild(node);
  } else setChildren(host, nodes);
}

function ledger() {
  if (!ledgerSegs) {
    ledgerSegs = Object.fromEntries(["block", "saved", "recent", "pending"].map((key) => [key, h("div", { class: `an-s ${key}` }, h("i"))]));
    setChildren($("an-ledger"), Object.values(ledgerSegs));
  }
  return ledgerSegs;
}

export function renderMap(k) {
  const facts = factsByTurn(k);
  const items = [
    ...k.turns.map((t) => ({ turn: t, i: t.turn_index, cat: categoryOf(t), tok: t.tokens, role: t.role, text: t.text })),
    ...pendingTurns(k).map((p) => ({ pending: true, i: p.turn_index, cat: "pending", tok: p.tokens, role: p.role, text: p.text })),
  ];
  const barsHost = $("an-bars");
  // Widths are tokens, with a floor of about MIN_BAR_PX so a one-token message stays visible and clickable.
  const total = items.reduce((n, x) => n + (x.tok ?? 0), 0) || items.length || 1;
  const floor = (total * MIN_BAR_PX) / Math.max(240, barsHost.clientWidth || 900);
  const width = (x) => Math.max(x.tok ?? floor, floor);
  barsHost.classList.toggle("dense", items.length > 80);
  barsHost.classList.toggle("searching", !!query());
  $("map").setAttribute("aria-label", `Every message in order, oldest first: ${items.length} in all`);

  const groupOf = (x) => x.pending ? "newest" : x.cat !== "verbatim" ? "condensed" : x.turn.becomes ? "next" : "tail";
  const runs = [];
  for (const x of items) {
    const g = groupOf(x), r = runs.at(-1);
    if (r && r.g === g) { r.n++; r.w += width(x); } else runs.push({ g, n: 1, w: width(x) });
  }
  const GROUP = { condensed: "Collected", next: "Next", tail: "Recent", newest: "" };
  setChildren($("an-labels"), runs.map((r) => h("div", { class: "an-l", style: `flex-grow:${r.w}` },
    h("span", { title: r.g === "newest" ? "Newest request" : `${GROUP[r.g]} · ${plural(r.n, "message")}` },
      r.g === "newest" ? " " : [h("b", {}, GROUP[r.g]), ` · ${r.n}`]))));

  let changed = 0;
  const cells = items.map((x) => {
    let el = barCache.get(x.i);
    if (!el) {
      el = h("div", { class: "bar", tabindex: "0", role: "button",
        onclick: () => jumpTo(el._i),
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); jumpTo(el._i); } },
        onpointerenter: (e) => { showTooltip(e, ...el._tip); trace([el._i]); },
        onpointerleave: () => { hideTooltip(); trace(null); },
        onfocus: (e) => { showTooltip(e, ...el._tip); trace([el._i]); },
        onblur: () => { hideTooltip(); trace(null); } });
      el._cell = h("div", { class: "an-c" }, el);
      barCache.set(x.i, el);
    }
    el._i = x.i;
    el._cell.style.flexGrow = String(width(x));
    const t = x.turn;
    const span = t && t.span !== null ? k.spans[t.span] : null;
    const hit = query() ? (x.pending ? has(x.text) : turnMatches(t, facts)) : null;
    el.className = `bar${x.pending ? " pending" : ` f-${x.cat}`}${hit === null ? "" : hit ? " hit" : " miss"}`;
    const before = lastCats.get(x.i);
    el.style.setProperty("--d", before !== undefined && before !== x.cat ? `${(changed++ * 0.03).toFixed(2)}s` : "0s");
    lastCats.set(x.i, x.cat);
    el.style.setProperty("--h", `${x.pending ? 100 : HEIGHT[x.cat]}%`);
    setChildren(el, t && t.becomes ? h("span", { class: "q", style: `--qb:${levelColour(t.becomes)}` }) : null);
    const size = plural(x.tok, "token");
    el.setAttribute("aria-label", x.pending ? `Message ${x.i}, newest request, ${size}: not in the conversation yet`
      : `Message ${x.i}, ${speakerOf(x.turn)}, ${size}: ${CATS[x.cat].label}`);
    el._tip = x.pending
      ? [`Message ${x.i} · newest request · ${size}`, GLOSSARY.newest[1], tipQuote(preview(x.text))]
      : [[swatch(x.cat), `Message ${x.i} · ${speakerOf(x.turn)} · ${size}`], `${CATS[x.cat].label}: ${CATS[x.cat].desc}`,
        span ? `Its exchange: ${messagesLabel(span.turns)}` : null,
        t.becomes ? `Next collection: ${BECOMES[t.becomes]}. ${whyText(span || t)}` : null,
        tipQuote(preview(x.text)), h("div", { class: "tip-row muted" }, "Click to open it")];
    return el._cell;
  });
  reconcile(barsHost, cells);

  // The Sent row: the recap and the saving split the width of the condensed messages
  // above them, and recent messages sit under themselves.
  const segs = ledger();
  const cond = items.filter((x) => !x.pending && x.cat !== "verbatim");
  const verb = items.filter((x) => !x.pending && x.cat === "verbatim");
  const pend = items.filter((x) => x.pending);
  const sum = (list) => list.reduce((n, x) => n + width(x), 0);
  const saved = writtenTokens(k) - k.context_tokens;
  const condW = sum(cond);
  const blockW = condW && saved > 0 ? (condW * k.front_tokens) / (k.front_tokens + saved) : condW;
  const recent = k.context_tokens - k.front_tokens;
  const pendTok = pend.reduce((n, x) => n + (x.tok ?? 0), 0);
  const parts = {
    block: [blockW, `The recap · ${plural(k.front_tokens, "token")}`, "Stands in for the collected messages above it.", cond],
    saved: [condW - blockW, `Saved · ${plural(Math.max(0, saved), "token")}`, "What collections took out of every request.", cond],
    recent: [sum(verb), `Recent messages · ${plural(recent, "token")}`, "Not collected yet: each sent as its own message, as written.", verb],
    pending: [sum(pend), `Newest request · ${plural(pendTok, "token")}`, "Its own message, sent on top of the recorded ones; not in the conversation yet.", pend],
  };
  for (const [key, [grow, title, text, list]] of Object.entries(parts)) {
    const seg = segs[key];
    seg.style.flexGrow = String(grow);
    seg.classList.toggle("zero", !grow);
    seg.onpointerenter = (e) => { showTooltip(e, title, text); trace(list.map((x) => x.i), seg); };
    seg.onpointerleave = () => { hideTooltip(); trace(null); };
  }
  renderKey(k, items, { saved, recent, pendTok });
}

// The key mirrors the map's two rows: what a bar means, then what the Sent row is made of.
function renderKey(k, items, { saved, recent, pendTok }) {
  const counts = countCats(k);
  const queued = k.turns.some((t) => t.becomes);
  const pending = items.some((x) => x.pending);
  const item = (mark, ...text) => h("span", {}, mark, h("span", {}, ...text));
  const explained = (key, mark, ...text) => h("span", hoverTip(GLOSSARY[key][0], GLOSSARY[key][1]), mark, h("span", {}, ...text));
  fill("an-key",
    h("div", { class: "an-rowlab" }, "Bars"),
    h("div", { class: "items" },
      ORDER.filter((key) => counts[key]).map((key) => explained(key, h("i", { class: `lg f-${key}`, style: `height:${Math.max(3, Math.round(HEIGHT[key] * 0.14))}px` }), CATS[key].label)),
      queued ? explained("next", h("i", { class: "qdot" }), "Dot: changes at the next collection, in the colour it will become") : null,
      pending ? item(h("i", { class: "key pending" }), "Outline: the newest request, not in the conversation yet") : null),
    h("div", { class: "an-rowlab" }, "Sent"),
    h("div", { class: "items" },
      k.front_tokens ? item(h("i", { class: "key block" }), "Recap ", h("b", {}, fmt(k.front_tokens))) : null,
      saved > 0 && k.front_tokens ? item(h("i", { class: "key saved" }), "Saved ", h("b", {}, fmt(saved))) : null,
      saved < 0 ? item(null, "The recap is ", h("b", {}, fmt(-saved)), " tokens larger than what it replaced") : null,
      item(h("i", { class: "key recent" }), "Recent messages ", h("b", {}, fmt(recent))),
      pending ? item(h("i", { class: "key pending" }), "Newest request ", h("b", {}, fmt(pendTok))) : null));
}

// Hovering a message anywhere traces it: its bar, its Sent segment, its row, and its score.
export function trace(ids, seg = null) {
  for (const el of document.querySelectorAll(".linked")) el.classList.remove("linked");
  $("an-bars").classList.toggle("tracing", !!ids);
  $("an-ledger").classList.toggle("tracing", !!ids);
  if (!ids) return;
  const k = state.knowledge;
  const segs = ledger();
  // Exchange rows and score rows stand for several messages; they carry them in data-turns.
  for (const row of document.querySelectorAll("#timeline [data-turns], #relevance [data-turns]")) {
    if (row.dataset.turns.split(",").map(Number).some((n) => ids.includes(n))) row.classList.add("linked");
  }
  for (const i of ids) {
    barCache.get(i)?.classList.add("linked");
    const t = k && k.turns.find((x) => x.turn_index === i);
    const key = !t ? "pending" : t.fidelity !== "verbatim" ? "block" : "recent";
    segs[key].classList.add("linked");
    if (key === "block") segs.saved.classList.add("linked");
  }
  if (seg) seg.classList.add("linked");
}

// After a collection the collected bars shrink first, then the block slides in beneath them; a brief
// glow marks where they went.
export function celebrateCleanup() {
  if (reducedMotion() || !ledgerSegs) return;
  const block = ledgerSegs.block;
  setTimeout(() => { block.classList.add("landed"); setTimeout(() => block.classList.remove("landed"), 1100); }, 900);
}

// ----------------------------------------------------------------- exchange by exchange

export function setFilter(key) {
  ui.filter = key;
  ui.listOpen = true;
  if (ui.promptView) setPromptView(false);
  renderMessages();
}

export function renderMessages() {
  const k = state.knowledge;
  if (!k) return;
  renderChips(k);
  if (ui.promptView) renderPrompt(k); else renderTimeline(k);
  renderMemory(k);
  renderMap(k);
}

export function renderChips(k) {
  const counts = countCats(k);
  const pending = pendingTurns(k).length + (newestReply() ? 1 : 0);
  const nextCount = changing(k);
  const chip = (key, label, count, mark) => h("button", { class: "chip-btn", type: "button", "aria-pressed": String(ui.filter === key),
    ...(GLOSSARY[key] ? hoverTip(GLOSSARY[key][0], GLOSSARY[key][1]) : {}),
    onclick: () => setFilter(ui.filter === key && key !== "all" ? "all" : key) }, mark, label, h("span", { class: "count" }, countEl(`chip-${key}`, count)));
  const toggle = h("button", { class: "list-toggle", type: "button", "aria-controls": "timeline", "aria-expanded": String(ui.listOpen),
    "aria-label": ui.listOpen ? "Hide the messages" : "Show the messages", title: ui.listOpen ? "Hide the messages" : "Show the messages",
    onclick: () => { ui.listOpen = !ui.listOpen; storeSet("llmgc.messages", ui.listOpen ? "open" : "closed"); renderChips(k); renderTimeline(k); } },
    svg("svg", { width: 14, height: 14, viewBox: "0 0 16 16", "aria-hidden": "true" },
      svg("path", { d: "M4 6l4 4 4-4", fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round" })));
  fill("chips", chip("all", "All", k.turns.length + pending),
    ORDER.filter((key) => counts[key]).map((key) => chip(key, CATS[key].label, counts[key], swatch(key))),
    nextCount ? chip("next", "Changes next", nextCount, h("span", { class: "qdot" })) : null, toggle);
}

// One row per exchange the old generation holds, one per run of folded exchanges (they share a
// line in the recap), then one per recent message, then the newest request.
export function buildRows(k) {
  const byIndex = new Map(k.turns.map((t) => [t.turn_index, t]));
  const rows = [];
  for (const span of k.spans) {
    const turns = span.turns.map((n) => byIndex.get(n));
    const last = rows.at(-1);
    if (span.level === "out_of_view" && last && last.kind === "folded") { last.spans.push(span); last.turns.push(...turns); }
    else if (span.level === "out_of_view") rows.push({ kind: "folded", cat: "out_of_view", spans: [span], turns });
    else rows.push({ kind: "span", cat: span.level, span, turns });
  }
  for (const turn of k.turns) if (turn.fidelity === "verbatim") rows.push({ kind: "turn", cat: "verbatim", turn, turns: [turn] });
  for (const pending of pendingTurns(k)) rows.push({ kind: "pending", cat: "verbatim", pending, turns: [] });
  const reply = newestReply();
  if (reply) rows.push({ kind: "reply", cat: "verbatim", pending: reply, turns: [] });
  return rows;
}

export function rowVisible(row, facts) {
  const f = ui.filter;
  // The newest message and its reply are not recorded yet, so they belong to no category but "all".
  const unrecorded = row.kind === "pending" || row.kind === "reply";
  const passes = f === "all" || (f === "next" ? row.turns.some((t) => t.becomes) : !unrecorded && row.cat === f);
  if (!passes) return false;
  if (!query()) return true;
  if (unrecorded) return has(row.pending.text);
  const line = row.kind === "span" ? row.span.text : row.kind === "folded" ? row.spans[0].text : null;
  return has(line) || row.turns.some((t) => turnMatches(t, facts));
}

// Long stretches of archived exchanges fold into one row, so the page leads with what the model
// reads in detail. Filtering or searching shows every match.
export function foldRuns(rows) {
  if (ui.filter !== "all" || query()) return rows;
  const out = [];
  let run = [];
  const flush = () => { if (run.length >= RUN_MIN) out.push({ kind: "run", rows: run, turns: run.flatMap((r) => r.turns) }); else out.push(...run); run = []; };
  for (const row of rows) {
    if (row.kind === "span" && row.cat === "archived") run.push(row);
    else { flush(); out.push(row); }
  }
  flush();
  return out;
}

// How many original messages a row stands for.
export const rowMessageCount = (row) => row.kind === "pending" || row.kind === "reply" ? 1 : row.turns.length;

const isQueued = (row) => row.kind === "turn" && !!row.turn.becomes;
const RECAP_ROWS_SHOWN = 3;   // recap rows shown before "Show N older messages"
const isCondensed = (row) => row.kind === "run" || row.kind === "span" || row.kind === "folded";

export function renderTimeline(k) {
  const facts = factsByTurn(k);
  const visible = buildRows(k).filter((row) => rowVisible(row, facts));
  // Newest first: what the model sees now leads, and older history reads on below it.
  const folded = foldRuns(visible).reverse();
  const filtered = ui.filter !== "all" || query();
  // Unfiltered, a long conversation shows only its newest rows; the rest sit behind a button at
  // the foot. Filtering, searching or jumping to a message shows everything that matches.
  // Unfolded, the list shows at least the first few rows of the recap: what the model reads in
  // place of everything older is the part a reader most wants to see.
  const firstCondensed = folded.findIndex(isCondensed);
  const keep = Math.max(EARLIER_SHOWN, firstCondensed === -1 ? 0 : firstCondensed + RECAP_ROWS_SHOWN);
  const canFold = !filtered && folded.length - keep >= RUN_MIN;
  const foldOlder = canFold && !ui.earlierOpen;
  const shown = foldOlder ? folded.slice(0, keep) : folded;
  // One line marks where the next collection's queue starts, and one where the recap starts.
  const items = [];
  let pastQueued = false, pastCondensed = false;
  for (const row of shown) {
    if (!filtered && !pastQueued && !pastCondensed && isQueued(row)) {
      pastQueued = true;
      items.push(h("div", { class: "boundary" }, h("span", {}, "↑ Stays recent · ↓ Taken by the next collection")));
    }
    if (!filtered && !pastCondensed && isCondensed(row)) {
      pastCondensed = true;
      items.push(h("div", { class: "boundary" }, h("span", {}, "↑ Recent, sent as written · ↓ In the recap, ", h("b", {}, fmt(k.front_tokens)), " tokens")));
    }
    items.push(rowElement(row, facts));
  }
  const timeline = $("timeline");
  timeline.hidden = !ui.listOpen;
  renderSearchVerdict(k);
  const note = $("filter-note");
  note.hidden = !filtered || !ui.listOpen;
  if (filtered) {
    const shownCount = visible.reduce((n, row) => n + rowMessageCount(row), 0);
    setChildren(note, `Showing ${plural(shownCount, "message")}${query() ? ` matching “${ui.query.trim()}”` : ""}${ui.filter !== "all" ? ` · ${ui.filter === "next" ? "changing at the next collection" : CATS[ui.filter].tag}` : ""}`,
      h("button", { class: "link-btn", type: "button", onclick: () => { ui.query = ""; $("search").value = ""; setFilter("all"); } }, "Show all"));
  }
  const olderCount = folded.slice(keep).reduce((n, row) => n + rowMessageCount(row), 0);
  const setOlder = (open) => {
    ui.earlierOpen = open;
    renderTimeline(state.knowledge);
    // Folding back shortens the list from below; keep the button in view rather than the page's foot.
    if (!open) $("older-toggle")?.scrollIntoView({ block: "nearest" });
  };
  const olderBtn = canFold ? h("div", { class: "earlier-row" },
    h("button", { class: "link-btn", id: "older-toggle", type: "button", "aria-expanded": String(!foldOlder), onclick: () => setOlder(foldOlder) },
      foldOlder ? `Show ${plural(olderCount, "older message")}` : "Show fewer messages")) : null;
  setChildren(timeline, items.length ? items : h("p", { class: "empty", style: "padding:14px 0" },
    filtered ? "No message matches." : "No messages recorded yet."), olderBtn);
  for (const row of timeline.querySelectorAll("[data-turns]")) {
    const ids = row.dataset.turns.split(",").map(Number);
    row.addEventListener("pointerenter", () => trace(ids));
    row.addEventListener("pointerleave", () => trace(null));
  }
}

function rowElement(row, facts) {
  return row.kind === "run" ? runRow(row) : row.kind === "span" ? spanRow(row) : row.kind === "folded" ? foldedRow(row, facts)
    : row.kind === "pending" ? pendingRow(row.pending) : row.kind === "reply" ? replyRow(row.pending) : turnRow(row.turn);
}

export function textBlock(text, { faded = false, key = null } = {}) {
  if (!text) return h("div", { class: "lost" }, "no text: thinking, or blocks with no text form");
  const long = text.length > CLAMP_CHARS || text.split("\n").length > 6 || (faded && text.length > 160);
  const open = !long || (key && ui.open.has(key)) || (query() && has(text));
  if (!long) return h("div", { class: `text${faded ? " faded" : ""}` }, marked(text));
  const toggle = () => {
    const nowOpen = !ui.open.has(key);
    if (nowOpen) ui.open.add(key); else ui.open.delete(key);
    body.classList.toggle("clamped", !nowOpen);
    more.textContent = nowOpen ? "Show less" : "Show more";
  };
  const body = h("div", { class: `text clampable${faded ? " faded" : ""}${open ? "" : " clamped"}`, onclick: toggle }, marked(text));
  const more = h("button", { class: "more", type: "button", onclick: toggle }, open ? "Show less" : "Show more");
  return [body, more];
}

export function fidTag(cat) {
  return h("span", { class: "tag", ...hoverTip(CATS[cat].label, GLOSSARY[cat][1]) }, swatch(cat), CATS[cat].tag);
}

// How a fact reaches the model: on its exchange's line, inside quoted words, or not at all.
const factStatus = (fact) => fact.status === "superseded"
  ? h("span", { class: "fact-status" }, h("span", { class: "st" }, "↺ "), "replaced")
  : fact.sent ? h("span", { class: "fact-status" }, h("span", { class: "st good" }, "✓ "), "on its line")
  : fact.span_level === "kept" ? h("span", { class: "fact-status" }, h("span", { class: "st good" }, "✓ "), "in its words")
  : h("span", { class: "fact-status" }, h("span", { class: "st warn" }, "⚠ "), "not sent");
// Who stated a fact: the user's decisions are theirs to change, an assistant's value is a recommendation.
const WHO = { user: "you", assistant: "AI" };
const said = (turn, speaker) => `message ${turn}${WHO[speaker] ? ` · ${WHO[speaker]}` : ""}`;
const whoTitle = (fact) => fact.speaker === "user" ? `You said this, in message ${fact.turn}.`
  : fact.speaker === "assistant" ? `The assistant said this, in message ${fact.turn}: a recommendation, which never replaces a decision of yours.` : null;
// A fact read from a statement that names no topic, like "we decided to ship on Friday".
const topicOf = (fact) => fact.topic === "general" ? "no topic" : fact.topic;

export function factRow(fact, { onclick = null } = {}) {
  const replaced = fact.status === "superseded";
  return h(onclick ? "button" : "div", { class: `fact s-${fact.status}`, type: onclick ? "button" : null, onclick,
    title: onclick ? "Show the message it came from" : null },
    h("span", { class: "fact-topic", title: topicOf(fact) }, topicOf(fact)),
    h("span", { class: "fact-what" }, replaced
      ? [h("div", { class: "was" }, marked(fact.content)), h("div", { class: "now" }, h("span", { class: "now-label" }, "now: "), marked(fact.superseded_by ?? ""))]
      : marked(fact.content)),
    h("span", { class: "fact-meta", title: replaced ? null : whoTitle(fact) }, h("span", { class: "num" }, replaced && fact.superseded_by_turn !== null
      ? `${said(fact.turn, fact.speaker)} → ${said(fact.superseded_by_turn, fact.superseded_by_speaker)}` : said(fact.turn, fact.speaker)),
      replaced ? null : factStatus(fact)));
}

function factChip(fact) {
  return h("span", { class: `fact-chip s-${fact.status}` }, h("b", {}, topicOf(fact)),
    h("span", { class: "fact-text" }, marked(fact.content)), factStatus(fact));
}

// "next: archived" explains itself: hover for the reason, click for its score on the Engine tab.
// `item` is a recent message, or an exchange already collected.
export function nextTag(item, turnIndex) {
  return h("button", { class: "tag next why", type: "button", "aria-label": `Next collection: ${BECOMES[item.becomes]}. Show why on the Engine tab`,
    onpointerenter: (e) => showTooltip(e, `Why ${BECOMES[item.becomes]}?`, whyText(item), h("div", { class: "tip-row muted" }, "Click to see its score on the Engine tab")),
    onpointerleave: hideTooltip, onclick: () => { hideTooltip(); showWhy(turnIndex); } },
    h("span", { class: `swatch f-${item.becomes}` }), `next: ${BECOMES[item.becomes]}`);
}

export function turnRow(turn) {
  return h("article", { class: "msg", id: `turn-${turn.turn_index}`, "data-turns": String(turn.turn_index) },
    h("div", { class: "msg-no" }, turn.turn_index),
    avatarOf(turn),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" },
        who(turn), callsTag(turn), instructionsTag(turn),
        turn.becomes ? nextTag(turn, turn.turn_index) : null,
        h("span", { class: "tok" }, plural(turn.tokens, "token"))),
      textBlock(turn.text, { key: `t-${turn.turn_index}` })));
}

// The messages of an exchange as they were said, each still reachable from the map.
function originals(turns, faded) {
  return turns.map((t) => h("div", { class: "orig", id: `turn-${t.turn_index}` },
    h("div", { class: "msg-head" }, who(t), callsTag(t), instructionsTag(t), h("span", { class: "num" }, `message ${t.turn_index}`),
      h("span", { class: "tok" }, plural(t.tokens, "token"))),
    textBlock(t.text, { faded, key: `t-${t.turn_index}` })));
}

function originalsToggle(key, turns, open) {
  return h("button", { class: "more orig-toggle", type: "button", "aria-expanded": String(open),
    onclick: () => { if (ui.open.has(key)) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
    `${open ? "Hide" : "Show"} what was said in ${messagesLabel(turns.map((t) => t.turn_index))}`);
}

const rangeNo = (turns) => turns.length === 1 ? turns[0].turn_index : `${turns[0].turn_index}–${turns.at(-1).turn_index}`;

// An exchange in the old generation: what the model reads for it, then what was said.
export function spanRow(row) {
  const { span, turns } = row;
  const first = turns[0].turn_index;
  const written = turns.reduce((n, t) => n + t.tokens, 0);
  const head = h("div", { class: "msg-head" }, h("span", { class: "who" }, plural(turns.length, "message")), fidTag(span.level),
    span.becomes ? nextTag(span, first) : null);
  const article = (...body) => h("article", { class: "msg", id: `span-${first}`, "data-turns": span.turns.join(",") },
    h("div", { class: "msg-no" }, rangeNo(turns)), h("div", { "aria-hidden": "true" }), h("div", { class: "msg-body" }, head, ...body));
  if (span.level === "kept") {
    return article(h("div", { class: "keeps-intro" }, `Quoted word for word in the recap · ${plural(span.tokens, "token")} with its markers, for ${plural(written, "token")} as written`),
      originals(turns, false));
  }
  const key = `orig-${first}`;
  const open = ui.open.has(key) || (query() && turns.some((t) => has(t.text)));
  return article(
    h("div", { class: "summary-box" }, h("div", { class: "label" }, `The model reads this instead · ${plural(span.tokens, "token")}, in place of ${plural(written, "token")} as written`),
      textBlock(span.text, { key: `line-${first}` })),
    originalsToggle(key, turns, open),
    open ? originals(turns, true) : null);
}

// A run of exchanges folded out of view: the one line the model reads for all of them.
export function foldedRow(row, facts) {
  const { spans, turns } = row;
  const first = turns[0].turn_index;
  const key = `orig-${first}`;
  const open = ui.open.has(key) || (query() && turns.some((t) => has(t.text)));
  const stored = turns.flatMap((t) => facts.get(t.turn_index) || []).filter((f) => f.status === "active");
  return h("article", { class: "msg gone", id: `span-${first}`, "data-turns": turns.map((t) => t.turn_index).join(",") },
    h("div", { class: "msg-no" }, rangeNo(turns)), h("div", { "aria-hidden": "true" }),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, `${plural(spans.length, "exchange")} · ${plural(turns.length, "message")}`), fidTag("out_of_view")),
      h("div", { class: "summary-box warn" }, h("div", { class: "label" }, h("span", { class: "st warn" }, "⚠ "), "The model reads only this line for all of them"),
        h("div", { class: "text" }, marked(spans[0].text))),
      stored.length ? h("div", { class: "keeps" }, h("div", { class: "keeps-intro" }, "Facts read from them, stored but not sent:"), stored.map(factChip)) : null,
      originalsToggle(key, turns, open),
      open ? originals(turns, true) : null));
}

export function pendingRow(p) {
  // Named for what the proxy knows, never for who wrote it: a side request a client
  // makes, such as Claude Code asking for a suggested next prompt, looks the same.
  return h("article", { class: "msg pending", id: `turn-${p.turn_index}`, "data-turns": String(p.turn_index) },
    h("div", { class: "msg-no" }, p.turn_index), h("div", { class: "av user", "aria-hidden": "true" }, "New"),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, "Newest request"),
        h("span", { class: "muted" }, "not in the conversation yet"), info("newest"), instructionsTag(p)),
      textBlock(p.text, { key: `t-${p.turn_index}` })));
}

// The reply as it reached the client. Shown, never recorded from here: the next request says whether the client kept it.
function replyRow(r) {
  return h("article", { class: "msg pending", id: `turn-${r.turn_index}`, "data-turns": String(r.turn_index) },
    h("div", { class: "msg-no" }, r.turn_index), h("div", { class: "av assistant", "aria-hidden": "true" }, "AI"),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, "Newest reply"),
        h("span", { class: "muted" }, "not in the conversation yet"), info("reply"),
        h("span", { class: "tok" }, plural(r.tokens, "token"))),
      textBlock(r.text, { key: `t-${r.turn_index}` })));
}

export function runRow(run) {
  const first = run.turns[0].turn_index, last = run.turns.at(-1).turn_index;
  const key = `run-${first}`;
  const open = ui.open.has(key);
  return h("div", { class: `run${open ? " open" : ""}` },
    h("button", { class: "run-head", type: "button", "aria-expanded": String(open),
      onclick: () => { if (open) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
      h("span", { class: "run-range" }, `${first}–${last}`),
      h("span", { class: "run-rail" }),
      h("span", {},
        h("span", { class: "run-title" }, `${plural(run.rows.length, "exchange")} archived: the model reads one line for each`),
        h("span", { class: "run-sub" }, h("span", { class: "minis" }, run.rows.map(() => h("span", { class: "mini f-archived" }))),
          plural(run.turns.length, "message"))),
      h("span", { class: "run-toggle" }, open ? "Hide" : "Show")),
    open ? [...run.rows].reverse().map(spanRow) : null);
}

export function jumpTo(turnIndex) {
  if (ui.tab !== "conversation") offerReturn(ui.tab, window.scrollY);
  showTab("conversation");
  setPromptView(false);
  const k = state.knowledge;
  if (!k) return;
  const facts = factsByTurn(k);
  // Make sure the message is on the page: clear a filter that hides it, open what folds it.
  const row = buildRows(k).find((r) => r.turns.some((t) => t.turn_index === turnIndex) || (r.pending && r.pending.turn_index === turnIndex));
  if (row && !rowVisible(row, facts)) { ui.filter = "all"; ui.query = ""; $("search").value = ""; }
  ui.earlierOpen = true;
  ui.listOpen = true;
  if (row && (row.kind === "span" || row.kind === "folded")) ui.open.add(`orig-${row.turns[0].turn_index}`);
  if (row && row.cat === "archived") {
    const run = foldRuns(buildRows(k).filter((r) => rowVisible(r, facts)))
      .find((r) => r.kind === "run" && r.turns.some((t) => t.turn_index === turnIndex));
    if (run) ui.open.add(`run-${run.turns[0].turn_index}`);
  }
  renderMessages();
  const target = $(`turn-${turnIndex}`);
  if (!target) return;
  target.scrollIntoView({ behavior: smooth(), block: "center" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1600);
}

export function renderPrompt(k) {
  const tail = k.turns.filter((t) => t.fidelity === "verbatim");
  const whole = [k.front_text ? `[user]\n${k.front_text}` : null, ...tail.map((t) => `[${t.role}]\n${t.text || ""}`)].filter(Boolean).join("\n\n");
  const copy = h("button", { class: "btn", type: "button", onclick: async () => {
    try { await navigator.clipboard.writeText(whole); copy.textContent = "Copied"; } catch { copy.textContent = "Could not copy"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1500);
  } }, "Copy");
  // Newest first, as the other view is, so switching views keeps the newest message on top;
  // Copy keeps the order the model reads, since that is the prompt someone would replay.
  setChildren($("prompt"),
    h("div", { class: "prompt-head" }, h("div", { class: "note" }, "The conversation as LLM-GC forwards it, before your next message, newest first. ",
      "The model reads it from the bottom up, and Copy gives it in that order. ",
      tail.some((t) => t.tool_calls || t.tool_results) ? "Recent messages go exactly as your client sent them; tool calls and results are shown in the text form LLM-GC reads them in. " : null,
      "Your client's own instructions are not shown – its system prompt, its tools, and what it puts inside messages – and go word for word with every request; those from collected messages open the recap. ",
      k.front_text ? "The recap stays byte for byte the same until the next collection, so the provider can keep it cached."
        : "Nothing has been collected, so your app's own request is forwarded untouched."), copy),
    [...tail].reverse().map((t) => h("div", { class: "prompt-msg" }, h("div", { class: "msg-head" }, h("span", { class: "who" }, t.role),
      t.tool_results ? toolTag("tool results") : t.tool_calls ? toolTag("tool calls") : null, instructionsTag(t),
      h("span", { class: "num" }, `message ${t.turn_index} · ${plural(t.tokens, "token")}`)), promptText(t.text || "(no text)", `prompt-${t.turn_index}`))),
    k.front_text ? h("div", { class: "prompt-msg front" }, h("div", { class: "msg-head" }, h("span", { class: "who" }, "user"),
      h("span", { class: "tag" }, h("span", { class: "key block" }), `the recap · ${plural(k.front_tokens, "token")}`), info("block")), promptText(k.front_text, "prompt-front")) : null);
}

// A long block starts folded, as messages do elsewhere; Copy still takes the whole prompt.
function promptText(text, key) {
  if (text.length <= CLAMP_CHARS && text.split("\n").length <= 8) return h("pre", {}, marked(text));
  const open = ui.open.has(key) || (query() && has(text));
  const toggle = () => {
    const nowOpen = !ui.open.has(key);
    if (nowOpen) ui.open.add(key); else ui.open.delete(key);
    body.classList.toggle("clamped", !nowOpen);
    more.textContent = nowOpen ? "Show less" : "Show more";
  };
  const body = h("pre", { class: open ? "" : "clamped" }, marked(text));
  const more = h("button", { class: "more", type: "button", onclick: toggle }, open ? "Show less" : "Show more");
  return [body, more];
}

// What the model reads of a message said in an exchange shown at each level below kept.
const BELOW_KEPT = {
  compacted: "now compacted: the model reads its exchange's summary, which may word it differently",
  archived: "now archived: the model reads only its exchange's opening line and facts",
  out_of_view: "now out of view: the model knows its exchange exists, not what was said",
};

// A search answers the question it is usually asked: does the model still receive these words?
// Matching is literal, as the search is, so the answer is about these exact words.
function renderSearchVerdict(k) {
  const box = $("search-verdict");
  const words = ui.query.trim();
  const verbatim = k.turns.filter((t) => t.fidelity === "verbatim" && has(t.text)).map((t) => t.turn_index);
  const newest = pendingTurns(k).filter((p) => has(p.text));
  const inRecap = has(k.front_text);
  const condensed = k.turns.filter((t) => t.fidelity !== "verbatim" && has(t.text));
  box.hidden = !words || (!verbatim.length && !newest.length && !inRecap && !condensed.length);
  if (box.hidden) return;
  if (verbatim.length || inRecap || newest.length) {
    const how = [verbatim.length ? `as written, in recent ${messagesLabel(verbatim)}` : null, inRecap ? "in the recap" : null,
      newest.length ? "in the newest request, which is not in the conversation yet" : null].filter(Boolean).join(", and ");
    box.className = "search-verdict good";
    setChildren(box, h("b", {}, `“${words}”: the model receives it`), ` ${how}.`);
    return;
  }
  const said = Object.entries(BELOW_KEPT).map(([level, how]) => {
    const turns = condensed.filter((t) => t.fidelity === level).map((t) => t.turn_index);
    return turns.length ? `Said in ${messagesLabel(turns)}, ${how}.` : null;
  }).filter(Boolean);
  box.className = "search-verdict warn";
  setChildren(box, h("b", {}, `“${words}”: the model does not receive these exact words.`), ` ${said.join(" ")}`,
    " The words are stored, so a later collection shows them again if the conversation comes back to them.");
}

export function setPromptView(on) {
  ui.promptView = on;
  $("view-prompt").setAttribute("aria-pressed", String(on));
  $("view-timeline").setAttribute("aria-pressed", String(!on));
  $("prompt").hidden = !on;
  $("timeline").hidden = on || !ui.listOpen;
  $("chips").hidden = on;
  $("filter-note").hidden = on || !ui.listOpen || (ui.filter === "all" && !query());
  if (state.knowledge) { if (on) renderPrompt(state.knowledge); else renderTimeline(state.knowledge); }
}

// The facts read from the conversation: what needs a look first - facts the model cannot see,
// and replacements to check - then every fact, folded, since most are already on the page.
export function renderMemory(k) {
  const matches = (f) => !query() || has(f.content) || has(f.topic) || has(f.superseded_by);
  const active = k.facts.filter((f) => f.status === "active");
  // Facts are read when a message is collected, so they reach only as far as the newest collected one.
  const collected = k.turns.filter((t) => t.fidelity !== "verbatim").map((t) => t.turn_index);
  const readTo = collected.length ? Math.max(...collected) : null;
  const groups = [
    { id: "unseen", title: "Not sent: out of view", icon: h("span", { class: "st warn" }, "⚠"),
      facts: active.filter((f) => !f.sent && f.span_level !== "kept"),
      note: "Their exchanges are folded to keep the recap within its budget, so the model can't see these." },
    { id: "replaced", title: "Replaced later", icon: h("span", { class: "st" }, "↺"), facts: k.facts.filter((f) => f.status === "superseded"),
      note: "The recap notes each replacement beside the exchange that said the old value, so the model reads that value as no longer current. Only your own messages replace a decision you made." },
    { id: "all", title: readTo === null ? "Every fact on record" : `On record up to message ${readTo}`, icon: null, facts: active,
      note: `${readTo === null ? "" : `Read from messages 0–${readTo} as they were collected. Newer messages reach the model word for word, so a change made in one is in its words, and is filed here once it is collected. `}`
        + "A compacted or archived exchange shows its facts on its line in the recap; a kept one has them in its words." },
  ];
  // An exchange made while its facts couldn't be read holds none, whatever its messages settled.
  const unreadTurns = k.spans.filter((s) => s.facts_unread).flatMap((s) => s.turns);
  const unread = unreadTurns.length ? h("section", { id: "memory-unread" },
    h("h3", {}, h("span", { class: "st warn" }, "⚠"), `Not read from ${messagesLabel(unreadTurns)}`),
    h("div", { class: "group-note" }, "Their facts couldn't be read when they were collected, so none are on record, and a later collection doesn't read them again. ",
      "The model still reads these messages at their level in the recap: word for word, as a summary, or by their opening line alone.")) : null;
  if (!k.facts.length) {
    fill("memory", unread || h("p", { class: "empty", style: "margin-top:10px" }, collecting()
      ? "A collection is reading this conversation now; the facts it reads appear here when it finishes."
      : "Nothing yet. When a collection makes an exchange, the facts read from it appear here."));
    return;
  }
  const [unseen, replaced] = groups;
  fill("memory",
    replaced.facts.length || unseen.facts.length ? h("div", { class: "memory-counts" }, h("span", {}, h("b", {}, fmt(active.length)), " on record"),
      replaced.facts.length ? h("span", {}, h("b", {}, fmt(replaced.facts.length)), " replaced") : null,
      unseen.facts.length ? h("span", {}, h("b", {}, fmt(unseen.facts.length)), " out of view") : null) : null,
    unread,
    groups.filter((g) => g.facts.some(matches)).map((g) => {
      const shown = g.facts.filter(matches);
      const open = ui.open.has(`facts-${g.id}`) || !!query();
      const toggle = (label) => h("button", { class: "link-btn show-all", type: "button",
        onclick: () => { if (open) ui.open.delete(`facts-${g.id}`); else ui.open.add(`facts-${g.id}`); renderMemory(k); } }, label);
      const list = open ? shown : shown.slice(0, FACTS_SHOWN);
      return h("section", { id: `memory-${g.id}` },
        h("h3", {}, g.icon, g.title, h("span", { class: "count" }, `· ${shown.length}`)),
        g.note ? h("div", { class: "group-note" }, g.note) : null,
        list.length ? h("div", { class: "list" }, list.map((f) => factRow(f, { onclick: () => jumpTo(f.turn) }))) : null,
        shown.length > FACTS_SHOWN && !query() ? toggle(open ? "Show fewer" : `Show all ${shown.length}`) : null);
    }),
    query() && !k.facts.some(matches) ? h("p", { class: "empty" }, "No fact matches.") : null);
}

// The collections beside the facts: when each was made and what it did, then what comes next.
export function renderCleanupLog() {
  const k = state.knowledge;
  if (!k) return;
  const passes = state.activity ? state.activity.recent_passes : [];
  const cuts = passes.filter((p) => p.gc_status === "completed").reverse().slice(0, 3);
  const next = changing(k);
  const whenWhy = h("button", { class: "link-btn", type: "button", onclick: () => { showTab("engine"); openCard("next", true); } }, "When and why");
  fill("cleanup-log",
    cuts.length ? cuts.map((p) => h("li", {}, h("div", {},
      h("b", {}, "Collection"), h("span", { class: "when" }, ` · ${agoTs(p.gc_completed_at)}`),
      h("div", { class: "muted" }, `${p.outcome ? `${describeOutcome(p.outcome)} · ` : ""}${fmt(p.tokens_before)} → ${fmt(p.tokens_before - p.tokens_saved)} tokens`))))
      : k.cuts ? null : h("li", { class: "future" }, h("div", { class: "muted" }, "No collections yet.")),
    // A collection waits on model calls and can take minutes; until it commits, the panels show the one before it.
    collecting() ? h("li", { class: "future" }, h("div", {}, h("b", {}, "Collecting now"),
      h("div", { class: "muted" }, "Summarising and reading facts in the background. This panel and the facts update when it finishes.")))
      : h("li", { class: "future" }, h("div", {}, h("b", {}, "Next collection"),
        h("div", { class: "muted" }, next ? `${next === 1 ? "1 message" : `${next} messages`} will change form. ` : "Nothing is queued to change yet. ", whenWhy))));
}
