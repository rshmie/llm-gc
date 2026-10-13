// The conversation view: what the model is told, fact by fact, exchange by exchange and message by message.
//
// Every value comes from an endpoint; this view formats and explains, and derives
// nothing beyond grouping, counting, and sums and differences of numbers it was given.

import { BECOMES, CATS, CLAMP_CHARS, FAR_JUMP_SCREENS, MAP_PX_PER_MESSAGE, EARLIER_SHOWN, GLOSSARY, HEIGHT, MIN_BAR_PX, NOT_COLLECTED_TEXT, ORDER, RUN_MIN, FACT_STATUS, categoryOf, describeOutcome, swatch } from "./vocabulary.js";
import { $, agoTs, countEl, fill, fmt, h, meterBar, plural, ranges, reducedMotion, setChildren, share, storeGet, storeSet, svg } from "./dom.js";
import { collecting, countCats, credentialRefusal, factsByTurn, has, instructionTokens, marked, newestReply, pendingTurns, preview, query, recapLimit, state, trafficOf, turnMatches, ui, writtenTokens } from "./state.js";
import { hideTooltip, info, showTooltip, tipAction, tipQuote, hoverTip } from "./tooltip.js";
import { offerReturn, showTab } from "./shell.js";
import { openCard, showWhy, whyText } from "./engine.js";

const smooth = () => (reducedMotion() ? "auto" : "smooth");
export const speakerName = (role) => role === "user" ? "You" : role ? role[0].toUpperCase() + role.slice(1) : "—";
const avatarText = (role) => role === "user" ? "You" : role === "assistant" ? "AI" : (role || "?").slice(0, 2).toUpperCase();
const avatar = (role) => h("div", { class: `av ${role === "user" ? "user" : "assistant"}`, "aria-hidden": "true" }, avatarText(role));
// A turn of tool results is the client reporting what its tools returned, and a relayed turn another
// agent's report; both come under the user role, and neither is the user's words.
const speakerOf = (turn) => turn.relayed ? "Subagent report" : turn.tool_results ? "Tool results" : speakerName(turn.role);
const avatarOf = (turn) => turn.relayed ? h("div", { class: "av assistant", "aria-hidden": "true" }, "Agent")
  : turn.tool_results ? h("div", { class: "av user", "aria-hidden": "true" }, "Tool") : avatar(turn.role);
const toolTip = () => hoverTip(GLOSSARY.tools[0], GLOSSARY.tools[1]);
const relayedTip = () => hoverTip(GLOSSARY.relayed[0], GLOSSARY.relayed[1]);
// The speaker, explained when it is the client reporting tool output or relaying a report; a tag when the assistant calls a tool.
const who = (turn) => h("span", { class: "who", ...(turn.relayed ? relayedTip() : turn.tool_results ? toolTip() : {}) }, speakerOf(turn));
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
const capitalised = (text) => text[0].toUpperCase() + text.slice(1);
export const messagesLabel = (turns) => turns.length === 1 ? `message ${turns[0]}` : `messages ${ranges(turns)}`;
// A level as a tooltip names it: a badge in the level's colour, then what the model reads for it.
const levelBadge = (cat) => h("span", { class: `tip-badge f-${cat}` }, CATS[cat].label);
const levelLine = (cat) => h("div", { class: "tip-level" }, levelBadge(cat), ` ${capitalised(CATS[cat].desc)}.`);
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
  // First, and with no way to hide it: unlike the others, it does not pass until someone acts.
  if (credentialRefusal()) items.push(line("critical", "✕", CREDENTIAL_REFUSED_TITLE, CREDENTIAL_REFUSED_TEXT,
    ["See why", () => { showTab("engine"); openCard("notmoved", true); }]));
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
  const less = written - sent;
  // A forecast: what the next request will carry. What requests did carry is measured, below it.
  const [tone, text] = !k.collected ? ["", "Same as written: not collected"]
    : k.cuts === 0 ? ["", "Same as written: nothing collected yet"]
    : less >= 0 ? ["good", `↓ ${fmt(less)} · ${share(less, written)}% less than as written`]
    : ["", `↑ ${fmt(-less)} more than as written: the recap outgrew what it replaced`];
  fill("panel-size",
    h("div", { class: "label" }, "The next request", info("size")),
    h("div", { class: "hero-row" }, h("span", { class: "hero" }, countEl("size-sent", sent)), h("span", { class: "hero-unit" }, "tokens"),
      h("span", { class: `delta${tone ? ` ${tone}` : ""}` }, text)),
    h("p", { class: "hero-sub" }, "Of this conversation: the recap, then the recent messages as written. Your next message is added on top, and the system prompt and tools go with it unchanged."),
    measuredSoFar());
}

// What this conversation's requests carried, as the proxy counted them: the record behind the forecast above.
function measuredSoFar() {
  const c = trafficOf(state.selected);
  if (!c) return null;
  const t = c.traffic;
  const saved = t.tokens_saved ? `saved ${fmt(t.tokens_saved)} (${share(t.tokens_saved, t.tokens_from_client)}% of all it sent)`
    : c.collected === false ? "nothing saved: not collected" : "nothing saved yet";
  return h("p", { class: "hero-sub measured" }, h("b", {}, "So far: "), `${plural(t.requests, "request")} sent ${fmt(t.tokens_from_client)} tokens overall, the system prompt and tools included; ${saved}.`,
    c.collections_not_yet_sent ? h("span", { class: "st warn" }, ` ⚠ The newest collection has not reached the model: no request has carried it yet.`) : null);
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
    h("div", { class: "note" }, sofar, h("div", { id: "last-collection" }),
      !k.collected ? h("div", { ...hoverTip("Not collected", NOT_COLLECTED_TEXT) }, "Not collected: a subagent's conversation is recorded and sent as written.")
        : next ? h("div", {}, k.next_cut.batch_ready ? `A collection runs after your next reply: ${changes(next)}. ` : `${changes(next)} at the next collection. `, why) : null));
  renderLastCollection();
}

// When the last collection ran, and a way to what it moved. Drawn on every refresh, as "ago" moves on;
// a collection running now is said here too, beside the counts it will change.
export function renderLastCollection() {
  const box = $("last-collection");
  if (!box) return;
  const passes = state.activity ? state.activity.recent_passes : [];
  const last = passes.filter((p) => p.gc_status === "completed").at(-1);
  const moved = h("button", { class: "link-btn", type: "button",
    onclick: () => { offerReturn("conversation", window.scrollY); showTab("engine"); openCard("moves", true); } }, "What moved");
  setChildren(box, collecting() ? "Collecting now: these counts update when it finishes."
    : last ? [`Last collection ${agoTs(last.gc_completed_at)}${last.outcome ? `: ${describeOutcome(last.outcome)}` : ""}. `, moved] : null);
}

export function renderWindow(k) {
  const w = k.context_window;
  const measured = trafficOf(state.selected);
  const window = measured && measured.context_window;
  const source = !window ? "the configured window" : window.reported ? `the window of ${measured.model || "this model"}, as the provider reports it`
    : "the configured window, as the provider has not reported this model's";
  const share = w ? k.context_tokens / w : 0;
  fill("panel-window",
    h("div", { class: "label" }, "The model's window", info("window")),
    h("div", { class: "tile-big" }, w ? [countEl("window-pct", share * 100, { format: (v) => (v < 0.1 && v > 0 ? "<0.1" : v.toFixed(v < 10 ? 1 : 0)) }), "%"] : "—", h("small", {}, " used")),
    h("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": String(w || 0), "aria-valuenow": String(k.context_tokens), "aria-label": "Share of the model's window" },
      meterBar("window-share", share * 100)),
    h("div", { class: "note" }, `${fmt(k.context_tokens)} of ${fmt(w)} tokens before your next message, in ${source}.`));
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

// The smallest size, in tokens, a bar is drawn at so that none is narrower than MIN_BAR_PX on a track this
// wide. Raising the floor widens the small bars and so narrows the rest a little, which raises the floor
// needed: the loop settles within a few passes, because the track always holds every message at
// MAP_PX_PER_MESSAGE, which is more than MIN_BAR_PX. On a track too narrow for that (a hidden tab) it
// does not settle and the bars come out equal, which is the honest limit.
function floorTokens(sizes, total, trackPx) {
  let floor = (total * MIN_BAR_PX) / trackPx;
  for (let pass = 0; pass < 24; pass++) {
    const drawn = sizes.reduce((n, size) => n + Math.max(size ?? floor, floor), 0);
    floor = (drawn * MIN_BAR_PX) / trackPx;
  }
  return floor;
}

// Fades the edge the map scrolls on, so bars out of view don't read as the start or end of the conversation.
function markMoreSideways(el) {
  const left = el.scrollLeft > 1;
  const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
  if (left || right) el.dataset.more = left && right ? "both" : left ? "left" : "right";
  else delete el.dataset.more;
}

export function renderMap(k) {
  const facts = factsByTurn(k);
  const items = [
    ...k.turns.map((t) => ({ turn: t, i: t.turn_index, cat: categoryOf(t), tok: t.tokens, role: t.role, text: t.text })),
    ...pendingTurns(k).map((p) => ({ pending: true, i: p.turn_index, cat: "pending", tok: p.tokens, role: p.role, text: p.text })),
  ];
  const barsHost = $("an-bars");
  const scroller = $("map");
  // Read before anything changes: a map the reader had scrolled to its newest end follows new messages.
  const atNewest = scroller.scrollLeft + scroller.clientWidth >= scroller.scrollWidth - 24;
  // Room for every message, however many: past the card's width the map scrolls sideways as one.
  scroller.style.setProperty("--track", `${items.length * MAP_PX_PER_MESSAGE}px`);
  if (!scroller.dataset.wired) {
    scroller.dataset.wired = "1";
    scroller.addEventListener("scroll", () => markMoreSideways(scroller), { passive: true });
  }
  // Widths are tokens, with a floor so a one-token message stays at least MIN_BAR_PX wide and clickable.
  const total = items.reduce((n, x) => n + (x.tok ?? 0), 0) || items.length || 1;
  const floor = floorTokens(items.map((x) => x.tok), total, Math.max(240, barsHost.clientWidth || 900));
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
      : [[swatch(x.cat), `Message ${x.i} · ${speakerOf(x.turn)} · ${size}`], levelLine(x.cat),
        span ? `Its exchange: ${messagesLabel(span.turns)}` : null,
        t.becomes ? h("div", { class: "tip-level" }, "Next collection: ", levelBadge(t.becomes), ` ${whyText(span || t)}`) : null,
        tipQuote(preview(x.text)), tipAction("Click to open")];
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

  // A new conversation opens at its newest end, where the live action is; after that the reader's place is
  // kept, unless they were already at the newest end, so new messages stay in view.
  if (ui.mapSession !== state.selected || atNewest) scroller.scrollLeft = scroller.scrollWidth;
  ui.mapSession = state.selected;
  markMoreSideways(scroller);
  $("an-hint").textContent = scroller.scrollWidth > scroller.clientWidth + 1 ? " · scrolls sideways" : "";
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
  renderFacts(k);
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

// A status on a fact, with its meaning on hover; the dotted underline says there is one.
function statusTag(key) {
  const [glyph, tone, label, meaning] = FACT_STATUS[key];
  return h("span", { class: "fact-status", ...hoverTip(glyph ? `${glyph} ${label}` : label, meaning) },
    glyph ? h("span", { class: `st ${tone}` }, `${glyph} `) : null, h("span", { class: "term" }, label));
}
// How a current fact reaches the model: listed beside its exchange, in its exchange's words, or not at all.
const factStatus = (fact) => statusTag(fact.sent ? "line" : fact.span_level === "kept" ? "words" : "unsent");
// Who stated a fact: the user's decisions are theirs to change, an assistant's value is a recommendation.
const WHO = { user: "you", assistant: "AI" };
const said = (turn, speaker) => `message ${turn}${WHO[speaker] ? ` · ${WHO[speaker]}` : ""}`;
// A fact read from a statement that names no topic, like "we decided to ship on Friday".
const topicOf = (fact) => fact.topic === "general" ? "no topic" : fact.topic;

function factChip(fact) {
  return h("span", { class: `fact-chip s-${fact.status}` }, h("b", {}, topicOf(fact)),
    h("span", { class: "fact-text" }, marked(fact.content)), factStatus(fact));
}

// "next: archived" explains itself: hover for the reason, click for its score on the Engine tab.
// `item` is a recent message, or an exchange already collected.
export function nextTag(item, turnIndex) {
  return h("button", { class: "tag next why", type: "button", "aria-label": `Next collection: ${BECOMES[item.becomes]}. Show why on the Engine tab`,
    onpointerenter: (e) => showTooltip(e, `Why ${BECOMES[item.becomes]}?`, whyText(item), tipAction("Click to see its score on the Engine tab")),
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
  // A long smooth scroll runs for a second or more, and a page that redraws while it runs can leave
  // it part way down the list. Far targets are jumped to; the flash shows where they landed.
  const { top, height } = target.getBoundingClientRect();
  const far = Math.abs(top + height / 2 - innerHeight / 2) > innerHeight * FAR_JUMP_SCREENS;
  target.scrollIntoView({ behavior: far ? "auto" : smooth(), block: "center" });
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

// ----------------------------------------------------------------- what the model is told, fact by fact

// A current fact reaches the model on its exchange's line in the recap, or in its words when that
// exchange is kept; `sent` is set from the facts the recap actually carries.
const reaches = (fact) => fact.sent || fact.span_level === "kept";
// Newest first, as the exchanges below are, with the facts the model can't see ahead of the rest.
const lookFirst = (a, b) => reaches(a) - reaches(b) || b.turn - a.turn;
const changedAt = (fact) => fact.superseded_by_turn ?? fact.turn;
// Whether the engine reads facts from the assistant's turns; without a model it reads only the user's.
const readsAssistant = () => !state.config || !state.config.facts_read_from || state.config.facts_read_from.includes("assistant");

// Every fact read so far by who stated it, with how it reaches the model, then each value replaced.
// The head - the counts, how far the reading reaches, and any warning - stays in view when the card is
// folded. It leads with "as of message N", never "current": facts are read only when messages are
// collected, so a value a recent message has already changed would read as still in force.
export function renderFacts(k) {
  const matches = (f) => !query() || has(f.content) || has(f.topic) || has(f.superseded_by);
  const active = k.facts.filter((f) => f.status === "active");
  const replaced = k.facts.filter((f) => f.status === "superseded");
  const unseen = active.filter((f) => !reaches(f));
  const collected = k.turns.filter((t) => t.fidelity !== "verbatim").map((t) => t.turn_index);
  const recent = k.turns.filter((t) => t.fidelity === "verbatim").map((t) => t.turn_index);
  // An exchange made while its facts couldn't be read holds none, whatever its messages settled.
  const unreadTurns = k.spans.filter((s) => s.facts_unread).flatMap((s) => s.turns);
  const unread = unreadTurns.length ? h("p", { class: "facts-unread" },
    h("b", {}, h("span", { class: "st warn" }, "⚠ "), `Not read from ${messagesLabel(unreadTurns)}. `),
    "Their facts couldn't be read when they were collected, so none are listed here, and a later collection doesn't read them again. ",
    "The model still reads these messages at their level in the recap: word for word, as a summary, or by their opening line alone.") : null;
  if (!k.facts.length) {
    fill("facts");
    fill("facts-head", unread || h("p", { class: "facts-note" }, collecting()
      ? "A collection is reading this conversation now; the facts it reads appear here when it finishes."
      : !k.cuts ? "Nothing has been collected yet, so the model is sent every message word for word. Facts are read from messages when a collection takes them."
      : readsAssistant() ? "The collection read the collected messages and found no settled facts in them. A fact is a topic with a value, such as “database: Postgres”, and appears here only when a message states one; the model still reads each message in the recap, at its level."
      : "No facts found. Facts are read from your messages only here, as they are when LLM-GC runs without a model, by simple "
        + "patterns that catch statements such as “The database is Postgres.”; none of the collected messages has one. "
        + "The model reads them in the recap, at their level."));
    return;
  }
  fill("facts-head",
    h("div", { class: "facts-headline" },
      collected.length ? h("span", {}, "As of ", h("b", {}, `message ${Math.max(...collected)}`)) : null,
      h("span", {}, h("b", {}, fmt(active.length)), ` ${active.length === 1 ? "fact" : "facts"}`),
      reachLine(active.length, unseen.length),
      replaced.length ? h("span", {}, h("b", {}, fmt(replaced.length)), ` earlier ${replaced.length === 1 ? "value" : "values"} replaced`)
        : h("span", {}, "nothing replaced")),
    h("p", { class: "facts-note" },
      collected.length ? `Facts are read from ${messagesLabel(collected)}.` : null,
      recent.length ? ` ${capitalised(messagesLabel(recent))} ${recent.length === 1 ? "is" : "are"} sent word for word and read `
        + "only once collected, so a fact here may be out of date: a value changed there reaches the model now, and shows here then." : null),
    unread);
  const assistantFacts = readsAssistant() ? active.filter((f) => f.speaker === "assistant").sort(lookFirst) : null;
  const scrolled = new Map([...$("facts").querySelectorAll(".facts-col .list")].map((list) => [list.dataset.col, list.scrollTop]));
  fill("facts", h("div", { class: "facts-cols" },
    factColumn(matches, { id: "user", title: "You said", mark: avatar("user"), row: currentFact,
      facts: active.filter((f) => f.speaker !== "assistant").sort(lookFirst),
      empty: ["No facts from you yet", "Choices, names, numbers and dates you state appear here once their messages are collected."] }),
    factColumn(matches, { id: "assistant", title: "The assistant said", mark: avatar("assistant"), row: currentFact, facts: assistantFacts,
      empty: assistantFacts
        ? ["No assistant facts yet", "Values the assistant states, such as a recommended setting, appear here once their messages are collected."]
        : ["The assistant's messages aren't read", "Facts are read from your messages only here, as they are when LLM-GC runs without a model. "
          + "The model still reads the assistant's replies, in the recap and as written."] }),
    factColumn(matches, { id: "replaced", title: "Replaced", mark: h("div", { class: "av replaced", "aria-hidden": "true" }, "↺"),
      row: replacedFact, facts: replaced.sort((a, b) => changedAt(b) - changedAt(a)),
      empty: ["No replacements on record", "When a later message changes a value, such as Elasticsearch → OpenSearch, the old and new values appear here."] })));
  // Redrawn at every update and keystroke: each column stays where its reader scrolled it.
  for (const list of $("facts").querySelectorAll(".facts-col .list")) {
    list.scrollTop = scrolled.get(list.dataset.col) ?? 0;
    markMore(list);
  }
}

// Fades the edge a list scrolls on, so facts out of its box don't read as the end of the column.
function markMore(list) {
  const above = list.scrollTop > 1;
  const below = list.scrollTop + list.clientHeight < list.scrollHeight - 1;
  if (above || below) list.dataset.more = above && below ? "both" : above ? "above" : "below";
  else delete list.dataset.more;
}

// How many of the facts listed reach the model, worded so the parts visibly add up to the whole.
function reachLine(total, unseen) {
  const seen = total - unseen;
  const reach = (n) => n === 1 ? " reaches the model" : " reach the model";
  const missing = (words) => h("span", { class: "alert" }, h("span", { class: "st warn" }, "⚠ "), ...words);
  if (!unseen) return h("span", {}, total === 1 ? "it reaches the model" : ["all ", h("b", {}, fmt(total)), reach(total)]);
  if (!seen) return missing([total === 1 ? "it doesn't" : "none", reach(2)]);
  return [h("span", {}, h("b", {}, fmt(seen)), reach(seen)), missing([h("b", {}, fmt(unseen)), unseen === 1 ? " doesn't" : " don't"])];
}

// One column: every fact under a heading, in a box that scrolls once they outgrow it, so the card
// keeps its height however many pile up; or a card saying why it has none, so an empty column never
// reads as missing content. `facts` is null for a side whose turns are not read.
function factColumn(matches, { id, title, mark, facts, row, empty }) {
  const shown = (facts ?? []).filter(matches);
  const key = `facts-${id}`;
  if (!shown.length) {
    const [what, why] = facts?.length ? ["No match", "None of these facts match your search."] : empty;
    return h("section", { class: "facts-col empty", "aria-label": title }, mark, h("div", {}, h("b", {}, what), h("p", {}, why)));
  }
  return h("section", { class: "facts-col", "aria-labelledby": `${key}-title` },
    h("h3", { id: `${key}-title` }, mark, title, h("span", { class: "count" }, fmt(shown.length))),
    h("div", { class: "list", "data-col": id, onscroll: (e) => markMore(e.currentTarget) }, shown.map(row)));
}

// A current fact, as a link to the message that stated it; the column says who.
function currentFact(fact) {
  return h("button", { class: "fact", type: "button", onclick: () => jumpTo(fact.turn) },
    h("span", { class: "fact-topic" }, topicOf(fact)),
    h("span", { class: "fact-what" }, marked(fact.content)),
    h("span", { class: "fact-meta" }, h("span", { class: "fact-link num" }, `message ${fact.turn}`), factStatus(fact),
      h("span", { class: "fact-go", "aria-hidden": "true" }, "→")));
}

// A replaced value, what replaced it, and a link to each message. The recap notes the replacement
// on the old value's exchange; when that exchange is out of view, the old value isn't sent at all.
function replacedFact(fact) {
  const link = (turn, speaker) => h("button", { class: "link-btn num", type: "button", title: `Show message ${turn}`, onclick: () => jumpTo(turn) },
    said(turn, speaker));
  const noted = fact.span_level !== null && fact.span_level !== "out_of_view";
  return h("div", { class: "fact s-superseded" },
    h("span", { class: "fact-topic" }, topicOf(fact)),
    h("span", { class: "fact-what" }, h("span", { class: "was" }, marked(fact.content)),
      fact.superseded_by !== null ? h("span", { class: "now" }, "→ ", marked(fact.superseded_by)) : null),
    h("span", { class: "fact-meta" }, link(fact.turn, fact.speaker),
      fact.superseded_by_turn !== null ? ["→", link(fact.superseded_by_turn, fact.superseded_by_speaker)] : null,
      statusTag(noted ? "noted" : "oldUnsent")));
}

// "How fact collection works", below the facts: drawn once, and from the same meanings as each status's tooltip.
export function renderFactsGuide() {
  fill("facts-how",
    h("p", {}, "When a collection takes messages, the model that writes their summaries also reads what they settled — choices, names, ",
      "numbers, dates — as short topic: value notes, once. Without a model, simple patterns read your own messages instead, ",
      "catching statements such as “The database is Postgres.”"),
    h("dl", { class: "facts-statuses" }, Object.values(FACT_STATUS).map(([glyph, tone, label, meaning]) => [
      h("dt", {}, glyph ? h("span", { class: `st ${tone}` }, `${glyph} `) : null, label), h("dd", {}, meaning)])),
    h("p", {}, "When a later message changes a value, the new fact is filed under the same topic and the old one moves to Replaced. ",
      "Only you change a decision you made: when the assistant recommends something else, your decision stays, and the ",
      "recommendation stays in its exchange's summary."),
    h("p", {}, "Facts are a small part of what the model is told: summaries and recent messages carry the rest."));
}

