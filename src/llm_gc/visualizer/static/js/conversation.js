// The conversation view: what the model knows, and message by message.
//
// Every value comes from an endpoint; this view formats and explains, and derives
// nothing beyond grouping, counting, and sums and differences of numbers it was given.

import { BECOMES, BECOMES_CAT, CATS, CLAMP_CHARS, EARLIER_SHOWN, FACTS_SHOWN, GLOSSARY, GONE, HEIGHT, MIN_BAR_PX, ORDER, RUN_MIN, categoryOf, describeOutcome, swatch } from "./vocabulary.js";
import { $, agoTs, countEl, fill, fmt, h, meterBar, percent, plural, ranges, reducedMotion, setChildren, storeSet, svg } from "./dom.js";
import { captured, countCats, factsByTurn, has, marked, maxFacts, pendingTurns, preview, query, saidByTurn, state, turnMatches, ui } from "./state.js";
import { hideTooltip, info, showTooltip, tipQuote, hoverTip } from "./tooltip.js";
import { offerReturn, showTab } from "./shell.js";
import { openCard, showWhy, whyText } from "./engine.js";

const smooth = () => (reducedMotion() ? "auto" : "smooth");
export const speakerName = (role) => role === "user" ? "You" : role ? role[0].toUpperCase() + role.slice(1) : "—";
const avatarText = (role) => role === "user" ? "You" : role === "assistant" ? "AI" : (role || "?").slice(0, 2).toUpperCase();
const avatar = (role) => h("div", { class: `av ${role === "user" ? "user" : "assistant"}`, "aria-hidden": "true" }, avatarText(role));
const changeForm = (n) => `${n === 1 ? "1 message changes" : `${fmt(n)} messages change`} form at the next collection.`;

export function renderHero(k) {
  renderSize(k);
  renderHolds(k);
  renderWindow(k);
  renderMap(k);
}

// ----------------------------------------------------------------- what needs attention

// Good news is one quiet line; each warning is its own amber line saying what to do about it.
export function renderStatus() {
  const k = state.knowledge;
  if (!k) return;
  const counts = countCats(k);
  const unsent = k.facts.filter((f) => f.status === "active" && !f.sent).length;
  const contradicted = k.facts.filter((f) => f.status === "superseded").length;
  const problems = state.activity ? state.activity.recent_degradations.length : 0;
  const next = k.turns.filter((t) => t.becomes).length;
  const toMessages = (filter) => () => { setFilter(filter); $("messages").scrollIntoView({ behavior: smooth(), block: "start" }); };
  const line = (tone, icon, title, text, action, then) => h("div", { class: `status-line tone-${tone}` },
    h("span", { class: "badge", "aria-hidden": "true" }, icon),
    h("span", { class: "msg-text" }, h("b", {}, title), text ? ` ${text}` : null),
    then || null,
    action ? h("button", { class: "link-btn", type: "button", onclick: action[1] }, `${action[0]} →`) : null);
  const items = [];
  if (counts.unseen) items.push(line("warn", "!", `The model sees nothing of ${plural(counts.unseen, "message")}.`,
    `Their facts were stored, but the recap carries at most ${maxFacts()} facts and theirs didn't make it.`, ["Show them", toMessages("unseen")]));
  if (unsent) items.push(line("warn", "!", `${plural(unsent, "fact")} stored but not sent.`,
    "Ranked below the facts the recap has room for, so the model can't see them.",
    ["Show the facts", () => $("memory-unsent")?.scrollIntoView({ behavior: smooth(), block: "start" })]));
  if (problems) items.push(line("warn", "!", `${plural(problems, "planned move")} didn't happen.`,
    "Those messages stayed word for word: nothing is lost, the prompt is just bigger than planned.",
    ["See why", () => { showTab("engine"); openCard("notmoved", true); }]));
  if (!items.length) {
    items.push(line("good", "✓", k.cuts ? "Nothing is out of view." : "The model sees the whole conversation.",
      k.cuts ? "Every collected message still reaches the model kept, compacted, or as a sent fact." : "Nothing has been collected yet.", null,
      next && !k.next_cut.batch_ready ? h("span", { class: "then" }, `→ ${changeForm(next)}`,
        h("button", { class: "link-btn", type: "button", onclick: () => { showTab("engine"); openCard("rate", true); } }, "Why these?")) : null));
  }
  if (k.aging_in_flight) items.push(line("", "…", "LLM-GC is collecting this conversation.", "The map updates when it finishes."));
  if (k.next_cut.batch_ready) items.push(line("", "→", "A collection runs after the next reply.", `Messages ${ranges(k.next_cut.turns)} will change form.`, ["Show them", toMessages("next")]));
  if (contradicted) items.push(line("", "↺", `${plural(contradicted, "fact")} contradicted later.`,
    "The newer fact is sent and the old one isn't. Check the replacement is the one you meant.", ["Show them", toMessages("superseded")]));
  if (!captured()) items.push(line("", "i", "What was said wasn't captured.",
    "The dashboard started after this conversation, so it shows what the model receives but not the original words."));
  fill("status", items);
}

// ----------------------------------------------------------------- the prompt: cost, holdings, window

export function renderSize(k) {
  const written = captured() ? state.transcript.recorded_tokens : null;
  const sent = k.context_tokens;
  const saved = written === null ? null : written - sent;
  const [tone, text] = written === null ? ["", "The words weren't captured, so there's nothing to compare with"]
    : k.cuts === 0 ? ["", "Same as written: nothing collected yet"]
    : saved >= 0 ? ["good", `↓ ${fmt(saved)} saved · ${percent(saved, written)}% less than as written`]
    : ["", `↑ ${fmt(-saved)} more than as written: the recap outgrew what it replaced`];
  fill("panel-size",
    h("div", { class: "label" }, "Each request carries, now", info("size")),
    h("div", { class: "hero-row" }, h("span", { class: "hero" }, countEl("size-sent", sent)), h("span", { class: "hero-unit" }, "tokens"),
      h("span", { class: `delta${tone ? ` ${tone}` : ""}` }, text)),
    h("p", { class: "hero-sub" }, "The size of one request today: the conversation as the model gets it, the recap first. Your new message comes on top of this. The running total is on the Engine tab."));
}

export function renderHolds(k) {
  const counts = countCats(k);
  const next = k.turns.filter((t) => t.becomes).length;
  const rows = ORDER.filter((key) => key !== "superseded" || counts.superseded).map((key) => h("li",
    { class: key === "unseen" && counts.unseen ? "alert" : counts[key] ? "" : "zero" },
    h("i", { class: `hb f-${key}`, style: `height:${Math.max(3, Math.round(HEIGHT[key] * 0.13))}px` }),
    h("span", {}, CATS[key].label), h("b", {}, countEl(`holds-${key}`, counts[key]))));
  fill("panel-holds",
    h("div", { class: "label" }, "What the model holds"),
    h("ul", { class: "holds" }, rows),
    h("div", { class: "note" }, `Of ${plural(k.turns.length, "recorded message")}; your newest is counted once its reply arrives.`,
      next ? [` ${changeForm(next)} `, h("button", { class: "link-btn", type: "button", onclick: () => { showTab("engine"); openCard("rate", true); } }, "Why?")] : null));
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
  const said = saidByTurn();
  const facts = factsByTurn(k);
  const items = [
    ...k.turns.map((t) => ({ turn: t, i: t.turn_index, cat: categoryOf(t), tok: said.get(t.turn_index)?.tokens ?? t.tokens ?? null,
      role: said.get(t.turn_index)?.role ?? t.role, text: said.get(t.turn_index)?.text ?? t.text })),
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

  const groupOf = (x) => x.pending ? "newest" : x.cat !== "verbatim" ? "condensed" : x.turn.becomes ? "next" : "kept";
  const runs = [];
  for (const x of items) {
    const g = groupOf(x), r = runs.at(-1);
    if (r && r.g === g) { r.n++; r.w += width(x); } else runs.push({ g, n: 1, w: width(x) });
  }
  const GROUP = { condensed: "Collected", next: "Next collection", kept: "Word for word", newest: "" };
  setChildren($("an-labels"), runs.map((r) => h("div", { class: "an-l", style: `flex-grow:${r.w}` },
    h("span", { title: r.g === "newest" ? "Your newest message" : null }, r.g === "newest" ? " " : [h("b", {}, GROUP[r.g]), ` · ${r.n}`]))));

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
    const hit = query() ? (x.pending ? has(x.text) : turnMatches(t, said, facts)) : null;
    el.className = `bar${x.pending ? " pending" : ` f-${x.cat}`}${hit === null ? "" : hit ? " hit" : " miss"}`;
    const before = lastCats.get(x.i);
    el.style.setProperty("--d", before !== undefined && before !== x.cat ? `${(changed++ * 0.03).toFixed(2)}s` : "0s");
    lastCats.set(x.i, x.cat);
    el.style.setProperty("--h", `${x.pending ? 100 : HEIGHT[x.cat]}%`);
    setChildren(el, t && t.becomes ? h("span", { class: "q", style: `--qb:var(--fid-${BECOMES_CAT[t.becomes]})` }) : null);
    const size = x.tok === null ? "size not captured" : plural(x.tok, "token");
    const keeps = (facts.get(x.i) || []).filter((f) => f.status === "active");
    el.setAttribute("aria-label", `Message ${x.i}, ${speakerName(x.role)}, ${size}: ${x.pending ? "newest, not recorded yet" : CATS[x.cat].label}`);
    el._tip = x.pending
      ? [`Message ${x.i} · ${speakerName(x.role)} · ${size}`, "Your newest message: recorded, with its reply, once the next request arrives.", tipQuote(preview(x.text))]
      : [[swatch(x.cat), `Message ${x.i} · ${speakerName(x.role)} · ${size}`], `${CATS[x.cat].label}: ${CATS[x.cat].desc}`,
        t.becomes ? `Next collection: ${BECOMES[t.becomes]}. ${whyText(t)}` : null,
        GONE.has(x.cat) && keeps.length ? `Keeps ${plural(keeps.length, "fact")}, ${keeps.filter((f) => f.sent).length} sent` : null,
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
  const written = captured() ? state.transcript.recorded_tokens : null;
  const saved = written === null ? null : written - k.context_tokens;
  const condW = sum(cond);
  const blockW = condW && saved > 0 ? (condW * k.front_tokens) / (k.front_tokens + saved) : condW;
  const recent = k.context_tokens - k.front_tokens;
  const pendTok = pend.reduce((n, x) => n + (x.tok ?? 0), 0);
  const parts = {
    block: [blockW, `The recap · ${plural(k.front_tokens, "token")}`, "Stands in for the collected messages above it.", cond],
    saved: [condW - blockW, `Saved · ${plural(saved ?? 0, "token")}`, "What collections took out of every request.", cond],
    recent: [sum(verb), `Recent messages · ${plural(recent, "token")}`, "Sent word for word.", verb],
    pending: [sum(pend), `Your newest · ${plural(pendTok, "token")}`, "Sent with your next request, on top of the rest.", pend],
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
  fill("an-key",
    h("div", { class: "an-rowlab" }, "Bars"),
    h("div", { class: "items" },
      ORDER.filter((key) => counts[key]).map((key) => item(h("i", { class: `lg f-${key}`, style: `height:${Math.max(3, Math.round(HEIGHT[key] * 0.14))}px` }), CATS[key].label)),
      queued ? item(h("i", { class: "qdot" }), "Dot: queued for the next collection, in the colour it will become") : null,
      pending ? item(h("i", { class: "key pending" }), "Outline: your newest message") : null),
    h("div", { class: "an-rowlab" }, "Sent"),
    h("div", { class: "items" },
      k.front_tokens ? item(h("i", { class: "key block" }), "Recap ", h("b", {}, fmt(k.front_tokens))) : null,
      saved > 0 && k.front_tokens ? item(h("i", { class: "key saved" }), "Saved ", h("b", {}, fmt(saved))) : null,
      saved < 0 ? item(null, "The recap is ", h("b", {}, fmt(-saved)), " tokens larger than what it replaced") : null,
      item(h("i", { class: "key recent" }), "Recent messages ", h("b", {}, fmt(recent))),
      pending ? item(h("i", { class: "key pending" }), "Your newest ", h("b", {}, fmt(pendTok)), ", sent next") : null));
}

// Hovering a message anywhere traces it: its bar, its Sent segment, its row, and its rating.
export function trace(ids, seg = null) {
  for (const el of document.querySelectorAll(".linked")) el.classList.remove("linked");
  $("an-bars").classList.toggle("tracing", !!ids);
  $("an-ledger").classList.toggle("tracing", !!ids);
  if (!ids) return;
  const k = state.knowledge;
  const segs = ledger();
  // Summary rows and quotes stand for several messages; they carry them in data-turns.
  for (const row of document.querySelectorAll("#timeline [data-turns]")) {
    if (row.dataset.turns.split(",").map(Number).some((n) => ids.includes(n))) row.classList.add("linked");
  }
  for (const i of ids) {
    barCache.get(i)?.classList.add("linked");
    $(`r-${i}`)?.classList.add("linked");
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

// ----------------------------------------------------------------- message by message

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
  const pending = pendingTurns(k).length;
  const nextCount = k.turns.filter((t) => t.becomes).length;
  const chip = (key, label, count, mark) => h("button", { class: "chip-btn", type: "button", "aria-pressed": String(ui.filter === key),
    onclick: () => setFilter(ui.filter === key && key !== "all" ? "all" : key) }, mark, label, h("span", { class: "count" }, countEl(`chip-${key}`, count)));
  const toggle = h("button", { class: "list-toggle", type: "button", "aria-controls": "timeline", "aria-expanded": String(ui.listOpen),
    "aria-label": ui.listOpen ? "Hide the messages" : "Show the messages", title: ui.listOpen ? "Hide the messages" : "Show the messages",
    onclick: () => { ui.listOpen = !ui.listOpen; storeSet("llmgc.messages", ui.listOpen ? "open" : "closed"); renderChips(k); renderTimeline(k); } },
    svg("svg", { width: 14, height: 14, viewBox: "0 0 16 16", "aria-hidden": "true" },
      svg("path", { d: "M4 6l4 4 4-4", fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round" })));
  fill("chips", chip("all", "All", k.turns.length + pending),
    ORDER.filter((key) => counts[key]).map((key) => chip(key, CATS[key].label, counts[key], swatch(key))),
    nextCount ? chip("next", "Next collection", nextCount, h("span", { class: "qdot" })) : null, toggle);
}

export function buildRows(k) {
  const rows = [];
  for (let i = 0; i < k.turns.length; i++) {
    const turn = k.turns[i];
    if (turn.fidelity === "summarised") {
      // One row per summary: what the model reads, then what was said in its place.
      const group = [turn];
      while (i + 1 < k.turns.length && k.turns[i + 1].fidelity === "summarised" && k.turns[i + 1].front_entry === turn.front_entry) group.push(k.turns[++i]);
      rows.push({ kind: "summary", cat: "summarised", turns: group, entry: k.front_entries[turn.front_entry] });
    } else rows.push({ kind: "turn", cat: categoryOf(turn), turn });
  }
  for (const pending of pendingTurns(k)) rows.push({ kind: "pending", cat: "verbatim", pending });
  return rows;
}

export function rowVisible(row, said, facts) {
  const f = ui.filter;
  // The newest message is not recorded yet, so it belongs to no category but "all".
  const passes = f === "all" || (f === "next" ? row.kind === "turn" && !!row.turn.becomes : row.kind !== "pending" && row.cat === f);
  if (!passes) return false;
  if (!query()) return true;
  if (row.kind === "summary") return has(row.entry.stated_text) || row.turns.some((t) => has(said.get(t.turn_index)?.text));
  if (row.kind === "pending") return has(row.pending.text);
  return turnMatches(row.turn, said, facts);
}

// Long stretches of condensed-away messages fold into one row, so the page leads
// with what the model still reads. Filtering or searching shows every match.
export function foldRuns(rows) {
  if (ui.filter !== "all" || query()) return rows;
  const out = [];
  let run = [];
  const flush = () => { if (run.length >= RUN_MIN) out.push({ kind: "run", rows: run }); else out.push(...run); run = []; };
  for (const row of rows) {
    if (row.kind === "turn" && GONE.has(row.cat)) run.push(row);
    else { flush(); out.push(row); }
  }
  flush();
  return out;
}

// How many original messages a row stands for: a run or a summary speaks for several at once.
export function rowMessageCount(row) {
  return row.kind === "run" ? row.rows.length : row.kind === "summary" ? row.turns.length : 1;
}

const isQueued = (row) => row.kind === "turn" && row.cat === "verbatim" && !!row.turn.becomes;
const isCondensed = (row) => row.kind === "run" || row.kind === "summary" || (row.kind === "turn" && row.cat !== "verbatim");

export function renderTimeline(k) {
  const said = saidByTurn();
  const facts = factsByTurn(k);
  const all = buildRows(k);
  const visible = all.filter((row) => rowVisible(row, said, facts));
  // Newest first: what the model sees now leads, and older history reads on below it.
  const folded = foldRuns(visible).reverse();
  const filtered = ui.filter !== "all" || query();
  // Unfiltered, a long conversation shows only its newest messages; the rest sit behind a button at
  // the foot. Filtering, searching or jumping to a message shows everything that matches.
  const canFold = !filtered && folded.length - EARLIER_SHOWN >= RUN_MIN;
  const foldOlder = canFold && !ui.earlierOpen;
  const shown = foldOlder ? folded.slice(0, EARLIER_SHOWN) : folded;
  // One line marks where the next collection's queue starts, and one where the recap starts.
  const items = [];
  let pastQueued = false, pastCondensed = false;
  for (const row of shown) {
    if (!filtered && !pastQueued && !pastCondensed && isQueued(row)) {
      pastQueued = true;
      items.push(h("div", { class: "boundary" }, h("span", {}, "↑ Stays word for word · ↓ Queued for the next collection")));
    }
    if (!filtered && !pastCondensed && isCondensed(row)) {
      pastCondensed = true;
      items.push(h("div", { class: "boundary" }, h("span", {}, "↑ Sent word for word · ↓ In the recap, ", h("b", {}, fmt(k.front_tokens)), " tokens")));
    }
    items.push(row.kind === "run" ? runRow(row, said, facts) : row.kind === "summary" ? summaryRow(row, said)
      : row.kind === "pending" ? pendingRow(row.pending) : turnRow(row.turn, said, facts));
  }
  const timeline = $("timeline");
  timeline.hidden = !ui.listOpen;
  const note = $("filter-note");
  note.hidden = !filtered || !ui.listOpen;
  if (filtered) {
    const shownCount = visible.reduce((n, row) => n + (row.kind === "summary" ? row.turns.length : 1), 0);
    setChildren(note, `Showing ${plural(shownCount, "message")}${query() ? ` matching “${ui.query.trim()}”` : ""}${ui.filter !== "all" ? ` · ${ui.filter === "next" ? "next collection" : CATS[ui.filter].tag}` : ""}`,
      h("button", { class: "link-btn", type: "button", onclick: () => { ui.query = ""; $("search").value = ""; setFilter("all"); } }, "Show all"));
  }
  const olderCount = folded.slice(EARLIER_SHOWN).reduce((n, row) => n + rowMessageCount(row), 0);
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

export function textBlock(text, { faded = false, key = null } = {}) {
  if (text === null || text === undefined) return h("div", { class: "lost" }, "the original words were not captured (the dashboard started after this message)");
  if (!text) return h("div", { class: "lost" }, "no text: tool calls or attachments only");
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

const factStatus = (fact) => fact.status === "superseded"
  ? h("span", { class: "fact-status" }, h("span", { class: "st" }, "↺ "), fact.superseded_by_turn !== null ? `replaced by #${fact.superseded_by_turn}` : "replaced")
  : fact.sent ? h("span", { class: "fact-status" }, h("span", { class: "st good" }, "✓ "), "sent")
  : h("span", { class: "fact-status" }, h("span", { class: "st warn" }, "⚠ "), "not sent");

// A fact in the sidebar: its topic above what it says, and whether the model is sent it.
// A fact as the model reads it: cut where the recap cuts it, and saying so.
function sentContent(fact) {
  return [marked(fact.sent_content ?? fact.content), fact.shortened
    ? h("span", { class: "cut-note" }, ` ✂ the model gets only the start: ${fmt(fact.sent_content.length)} of ${fmt(fact.content.length)} characters`)
    : null];
}

function excerptLabel(fact) {
  return fact.from_summary ? "excerpt of its summary" : "excerpt";
}

function excerptTitle(fact) {
  return fact.from_summary
    ? "LLM-GC's summary of this message, kept as an excerpt when the summary was archived: not the message's own words"
    : "an excerpt: the words extraction could not turn into a fact";
}

export function factRow(fact, { withFrom = true, onclick = null } = {}) {
  const raw = fact.knowledge_type === "raw";
  return h(onclick ? "button" : "div", { class: `fact s-${fact.status}`, type: onclick ? "button" : null, onclick,
    title: onclick ? "Show the message it came from" : null },
    h("span", { class: "fact-topic", title: raw ? excerptTitle(fact) : fact.topic }, raw ? excerptLabel(fact) : fact.topic),
    h("span", { class: "fact-what" }, sentContent(fact)),
    h("span", { class: "fact-meta" }, withFrom ? h("span", { class: "num" }, `#${ranges(fact.source_turns)}`) : null, factStatus(fact)));
}

// A fact beside the message it came from, as a compact topic-and-value chip.
function factChip(fact) {
  const raw = fact.knowledge_type === "raw";
  // A cut excerpt opens to everything stored, so what the model never gets is visible too.
  const key = `fact-${fact.turn}-${fact.topic}-${fact.content.length}`;
  const open = fact.shortened && ui.open.has(key);
  const toggle = fact.shortened ? h("button", { class: "more", type: "button", "aria-expanded": String(open),
    onclick: () => { if (ui.open.has(key)) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
    open ? "Hide what is stored" : "Show what is stored") : null;
  return h("span", { class: `fact-chip s-${fact.status}${fact.shortened ? " long" : ""}`, title: raw ? excerptTitle(fact) : null },
    h("b", {}, raw ? excerptLabel(fact) : fact.topic),
    h("span", { class: "fact-text" }, sentContent(fact),
      open ? h("span", { class: "stored-full" }, h("span", { class: "stored-label" }, "Stored in full, not sent beyond the start:"), marked(fact.content)) : null,
      toggle),
    factStatus(fact));
}

// "next: summarised" explains itself: hover for the reason, click for its rating on the Engine tab.
export function nextTag(turn) {
  return h("button", { class: "tag next why", type: "button", "aria-label": `Next collection: ${BECOMES[turn.becomes]}. Show why on the Engine tab`,
    onpointerenter: (e) => showTooltip(e, `Why ${BECOMES[turn.becomes]}?`, whyText(turn), h("div", { class: "tip-row muted" }, "Click to see its rating on the Engine tab")),
    onpointerleave: hideTooltip, onclick: () => { hideTooltip(); showWhy(turn.turn_index); } },
    h("span", { class: `swatch f-${BECOMES_CAT[turn.becomes]}` }), `next: ${BECOMES[turn.becomes]}`);
}

export function turnRow(turn, said, facts) {
  const cat = categoryOf(turn);
  const words = said.get(turn.turn_index);
  const role = words ? words.role : turn.role;
  const kept = facts.get(turn.turn_index) || [];
  const key = `t-${turn.turn_index}`;
  const tokens = words?.tokens ?? turn.tokens;
  let body;
  if (cat === "verbatim") body = textBlock(turn.text, { key });
  else if (cat === "quoted") body = textBlock(words ? words.text : state.knowledge.front_entries[turn.front_entry].stated_text, { key });
  else {
    const viaSummary = kept.some((f) => f.from_summary);
    const intro = cat === "unseen" ? [h("span", { class: "st warn" }, "⚠ "), "The model sees nothing of this message. Stored, but not sent:"]
      : cat === "superseded" ? "Contradicted later, so no longer sent:"
      : viaSummary ? "Compacted first; when its summary went cold, the summary was archived. Extraction read the summary and kept:"
      : "Archived: the message left the prompt, and extraction kept:";
    body = [textBlock(words?.text, { faded: true, key }), h("div", { class: "keeps" }, h("div", { class: "keeps-intro" }, intro),
      kept.length ? kept.map(factChip) : h("span", { class: "lost" }, "nothing"))];
  }
  // The common case carries no badge: only a message that changed, or is about to, is tagged.
  return h("article", { class: `msg${GONE.has(cat) ? " gone" : ""}`, id: `turn-${turn.turn_index}`, "data-turns": String(turn.turn_index) },
    h("div", { class: "msg-no" }, turn.turn_index),
    avatar(role),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" },
        h("span", { class: "who" }, speakerName(role)),
        cat !== "verbatim" ? fidTag(cat) : null,
        turn.becomes ? nextTag(turn) : null,
        tokens && (cat === "verbatim" || cat === "quoted") ? h("span", { class: "tok" }, plural(tokens, "token")) : null),
      body));
}

export function summaryRow(row, said) {
  const { turns, entry } = row;
  const first = turns[0].turn_index, last = turns.at(-1).turn_index;
  const key = `orig-${first}`;
  const open = ui.open.has(key) || (query() && turns.some((t) => has(said.get(t.turn_index)?.text)));
  // A message's size as written lives in the transcript: the session stops counting it once it moves.
  const sizes = turns.map((t) => said.get(t.turn_index)?.tokens);
  const written = sizes.every((n) => typeof n === "number") ? sizes.reduce((sum, n) => sum + n, 0) : null;
  return h("article", { class: "msg", id: `entry-${first}`, "data-turns": turns.map((t) => t.turn_index).join(",") },
    h("div", { class: "msg-no" }, first === last ? first : `${first}–${last}`),
    h("div", { "aria-hidden": "true" }),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, plural(turns.length, "message")), fidTag("summarised")),
      h("div", { class: "summary-box" }, h("div", { class: "label" }, `The model reads this instead · ${plural(entry.tokens, "token")}${written === null ? "" : `, in place of ${plural(written, "token")} as written`}`), h("div", { class: "text" }, marked(entry.stated_text))),
      h("button", { class: "more orig-toggle", type: "button", "aria-expanded": String(!!open),
        onclick: () => { if (ui.open.has(key)) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
        `${open ? "Hide" : "Show"} what was said in ${turns.length > 1 ? `messages ${first}–${last}` : `message ${first}`}`),
      open ? turns.map((t) => h("div", { class: "orig", id: `turn-${t.turn_index}` },
        h("div", { class: "msg-head" }, h("span", { class: "who" }, speakerName(said.get(t.turn_index)?.role)), h("span", { class: "num" }, `message ${t.turn_index}`),
          typeof said.get(t.turn_index)?.tokens === "number" ? h("span", { class: "tok" }, plural(said.get(t.turn_index).tokens, "token")) : null),
        textBlock(said.get(t.turn_index)?.text, { faded: true, key: `t-${t.turn_index}` }))) : null));
}

export function pendingRow(p) {
  return h("article", { class: "msg pending", id: `turn-${p.turn_index}`, "data-turns": String(p.turn_index) },
    h("div", { class: "msg-no" }, p.turn_index), avatar(p.role),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, speakerName(p.role)), h("span", { class: "tag" }, "newest"),
        h("span", {}, "recorded, with its reply, once the next request arrives")),
      textBlock(p.text, { key: `t-${p.turn_index}` })));
}

export function runRow(run, said, facts) {
  const first = run.rows[0].turn.turn_index, last = run.rows.at(-1).turn.turn_index;
  const key = `run-${first}`;
  const open = ui.open.has(key);
  const counts = {};
  for (const row of run.rows) counts[row.cat] = (counts[row.cat] || 0) + 1;
  return h("div", { class: `run${open ? " open" : ""}` },
    h("button", { class: "run-head", type: "button", "aria-expanded": String(open),
      onclick: () => { if (open) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
      h("span", { class: "run-range" }, `${first}–${last}`),
      h("span", { class: "run-rail" }),
      h("span", {},
        h("span", { class: "run-title" }, `${plural(run.rows.length, "message")} the model no longer reads word for word`),
        h("span", { class: "run-sub" }, h("span", { class: "minis" }, run.rows.map((row) => h("span", { class: `mini f-${row.cat}` }))),
          ORDER.filter((c) => counts[c]).map((c) => `${counts[c]} ${CATS[c].tag}`).join(" · "))),
      h("span", { class: "run-toggle" }, open ? "Hide" : "Show")),
    open ? [...run.rows].reverse().map((row) => turnRow(row.turn, said, facts)) : null);
}

export function jumpTo(turnIndex) {
  if (ui.tab !== "conversation") offerReturn(ui.tab, window.scrollY);
  showTab("conversation");
  setPromptView(false);
  const k = state.knowledge;
  const turn = k && k.turns.find((t) => t.turn_index === turnIndex);
  // Make sure the message is on the page: clear a filter that hides it, open what folds it.
  const said = saidByTurn();
  const facts = factsByTurn(k);
  const row = turn ? buildRows(k).find((r) => r.kind === "turn" ? r.turn === turn : r.kind === "summary" && r.turns.includes(turn)) : null;
  if (row && !rowVisible(row, said, facts)) { ui.filter = "all"; ui.query = ""; $("search").value = ""; }
  // A message folded away, or in a hidden list, still needs to be reachable from the map.
  ui.earlierOpen = true;
  ui.listOpen = true;
  if (turn && turn.fidelity === "summarised") {
    const entryRow = buildRows(k).find((r) => r.kind === "summary" && r.turns.includes(turn));
    ui.open.add(`orig-${entryRow.turns[0].turn_index}`);
  }
  if (turn && GONE.has(categoryOf(turn))) {
    const run = foldRuns(buildRows(k).filter((r) => rowVisible(r, said, facts))).find((r) => r.kind === "run" && r.rows.some((x) => x.turn === turn));
    if (run) ui.open.add(`run-${run.rows[0].turn.turn_index}`);
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
  setChildren($("prompt"),
    h("div", { class: "prompt-head" }, h("div", { class: "note" }, "The conversation as LLM-GC forwards it, before your next message. ",
      k.front_text ? "The recap stays byte for byte the same until the next collection, so the provider can keep it cached."
        : "Nothing has been collected, so your app's own request is forwarded untouched."), copy),
    k.front_text ? h("div", { class: "prompt-msg front" }, h("div", { class: "msg-head" }, h("span", { class: "who" }, "user"),
      h("span", { class: "tag" }, h("span", { class: "key block" }), `the recap · ${plural(k.front_tokens, "token")}`), info("block")), promptText(k.front_text, "prompt-front")) : null,
    tail.map((t) => h("div", { class: "prompt-msg" }, h("div", { class: "msg-head" }, h("span", { class: "who" }, t.role),
      h("span", { class: "num" }, `message ${t.turn_index} · ${plural(t.tokens, "token")}`)), promptText(t.text || "(no text)", `prompt-${t.turn_index}`))));
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

export function renderMemory(k) {
  const matches = (f) => !query() || has(f.content) || has(f.topic);
  const active = k.facts.filter((f) => f.status === "active");
  const groups = [
    { id: "sent", title: "Sent to the model", icon: h("span", { class: "st good" }, "✓"), facts: active.filter((f) => f.sent) },
    { id: "unsent", title: "Stored, not sent", icon: h("span", { class: "st warn" }, "⚠"), facts: active.filter((f) => !f.sent),
      note: `The recap carries at most ${maxFacts()} facts, most important first. The model cannot see these.` },
    { id: "superseded", title: "Contradicted later", icon: h("span", { class: "st" }, "↺"), facts: k.facts.filter((f) => f.status === "superseded") },
  ];
  if (!k.facts.length) {
    fill("memory", h("p", { class: "empty", style: "margin-top:10px" }, "Nothing yet. When a collection archives a message, the facts it established appear here."));
    return;
  }
  fill("memory",
    h("div", { class: "memory-counts" }, groups.map((g) => h("span", {}, h("b", {}, fmt(g.facts.length)), ` ${g.id === "sent" ? "sent" : g.id === "unsent" ? "not sent" : "contradicted"}`))),
    groups.filter((g) => g.facts.some(matches)).map((g) => {
      const shown = g.facts.filter(matches);
      const all = ui.open.has(`facts-${g.id}`) || query();
      return h("section", { id: `memory-${g.id}` },
        h("h3", {}, g.icon, g.title, h("span", { class: "count" }, `· ${shown.length}`)),
        g.note ? h("div", { class: "group-note" }, g.note) : null,
        h("div", { class: "list" }, (all ? shown : shown.slice(0, FACTS_SHOWN)).map((f) => factRow(f, { onclick: () => jumpTo(f.source_turns[0]) }))),
        shown.length > FACTS_SHOWN && !query() ? h("button", { class: "link-btn show-all", type: "button",
          onclick: () => { if (all) ui.open.delete(`facts-${g.id}`); else ui.open.add(`facts-${g.id}`); renderMemory(k); } },
          all ? "Show fewer" : `Show all ${shown.length}`) : null);
    }),
    query() && !k.facts.some(matches) ? h("p", { class: "empty" }, "No fact matches.") : null);
}

// The collections beside the facts: when each was made and what it did, then what comes next.
export function renderCleanupLog() {
  const k = state.knowledge;
  if (!k) return;
  const passes = state.activity ? state.activity.recent_passes : [];
  const cuts = passes.filter((p) => p.gc_status === "completed" && p.tokens_saved > 0).reverse().slice(0, 3);
  const next = k.turns.filter((t) => t.becomes).length;
  const whenWhy = h("button", { class: "link-btn", type: "button", onclick: () => { showTab("engine"); openCard("next", true); } }, "When and why");
  fill("cleanup-log",
    cuts.length ? cuts.map((p) => h("li", {}, h("div", {},
      h("b", {}, "Collection"), h("span", { class: "when" }, ` · ${agoTs(p.gc_completed_at)}`),
      h("div", { class: "muted" }, `${p.outcome ? `${describeOutcome(p.outcome)} · ` : ""}${fmt(p.tokens_before)} → ${fmt(p.tokens_before - p.tokens_saved)} tokens`))))
      : k.cuts ? null : h("li", { class: "future" }, h("div", { class: "muted" }, "No collections yet.")),
    h("li", { class: "future" }, h("div", {}, h("b", {}, "Next collection"),
      h("div", { class: "muted" }, next ? `${next === 1 ? "1 message" : `${next} messages`} will change form. ` : "Nothing is queued to change yet. ", whenWhy))));
}
