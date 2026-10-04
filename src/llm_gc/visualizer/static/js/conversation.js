// The conversation view: what the model can see, and message by message.
//
// Every value comes from an endpoint; this view formats and explains, and derives
// nothing beyond grouping, counting, and sums and differences of numbers it was given.

import { BECOMES, BECOMES_CAT, CATS, CELL_MAX_PX, CLAMP_CHARS, EARLIER_SHOWN, FACTS_SHOWN, GLOSSARY, GONE, ORDER, RUN_MIN, categoryOf, speaker, swatch } from "./vocabulary.js";
import { $, agoIso, countEl, fill, fmt, h, meterBar, percent, plural, ranges, setChildren, storeSet, svg } from "./dom.js";
import { captured, countCats, factsByTurn, has, keepLast, marked, maxFacts, pendingTurns, preview, query, saidByTurn, state, turnMatches, ui } from "./state.js";
import { hideTooltip, info, showTooltip, tipQuote, hoverTip } from "./tooltip.js";
import { showTab } from "./shell.js";

export function renderHero(k) {
  const counts = countCats(k);
  const inBlock = counts.quoted + counts.summarised;
  fill("hero-summary", k.cuts === 0
    ? [`All ${plural(k.recorded_turns, "message")} reach the model `, h("b", {}, "word for word"), ". ",
       `Nothing has been condensed yet: the first clean-up comes once older messages add up to ${fmt(k.next_cut.min_cut_tokens)} tokens worth condensing.`]
    : [`Of ${plural(k.recorded_turns, "message")}, the model gets `, h("b", {}, `${fmt(counts.verbatim)} word for word`),
       inBlock ? [", ", h("b", {}, `${fmt(inBlock)} through the condensed block`)] : "",
       counts.facts ? [", ", h("b", {}, `${fmt(counts.facts)} only as facts`)] : "",
       counts.unseen ? [" — and ", h("b", {}, `nothing of ${fmt(counts.unseen)}`)] : "", "."]);
  renderHeroMeta(k);
  renderMap(k);
  renderLegend(k, counts);
  renderSize(k);
  renderNext(k);
  renderWindow(k);
}

export function renderHeroMeta(k) {
  const current = state.sessions.find((s) => s.session_id === state.selected);
  fill("hero-meta",
    k.aging_in_flight ? [h("span", { class: "dot warn" }), "cleaning up…"] : null,
    h("span", {}, plural(k.cuts, "clean-up")), "·",
    h("span", {}, current ? `active ${agoIso(current.last_activity)}` : ""));
}

// Cells keyed by turn_index and reused across renders: a message that changes category gets its
// existing element recoloured (and moved to its new group) instead of torn down and rebuilt, which
// is what lets the colour change transition instead of jumping when a clean-up runs.
const mapCellCache = new Map();
export function clearMapCache() { mapCellCache.clear(); }

export function renderMap(k) {
  const said = saidByTurn();
  const facts = factsByTurn(k);
  const condensed = k.turns.filter((t) => t.fidelity !== "verbatim");
  const next = k.turns.filter((t) => t.fidelity === "verbatim" && t.becomes);
  const kept = k.turns.filter((t) => t.fidelity === "verbatim" && !t.becomes);
  const pending = pendingTurns(k);
  const groups = [
    { key: "condensed", label: "Condensed", term: "block", turns: condensed, pending: [] },
    { key: "next", label: "Next clean-up", term: "next", turns: next, pending: [] },
    { key: "kept", label: "Word for word", term: "verbatim", turns: kept, pending },
  ].filter((g) => g.turns.length + g.pending.length > 0);
  const total = k.turns.length + pending.length;
  const map = $("map");
  map.classList.toggle("dense", total > 120);
  map.classList.toggle("searching", !!query());
  map.setAttribute("aria-label", `Every message in order: ${groups.map((g) => `${g.label.toLowerCase()} ${g.turns.length + g.pending.length}`).join(", ")}`);

  const cell = (t) => {
    const cat = categoryOf(t);
    const words = said.get(t.turn_index)?.text ?? t.text;
    const role = said.get(t.turn_index)?.role ?? t.role;
    const keeps = (facts.get(t.turn_index) || []).filter((f) => f.status === "active");
    let el = mapCellCache.get(t.turn_index);
    if (!el) {
      el = h("div", {
        onclick: () => jumpTo(el._tip.turnIndex), onpointerleave: hideTooltip,
        onpointerenter: (e) => { const d = el._tip; showTooltip(e, [swatch(d.cat), `Message ${d.turnIndex} · ${speaker(d.role)}`],
          `${CATS[d.cat].label}: ${CATS[d.cat].desc}`,
          d.becomes ? `Next clean-up: ${BECOMES[d.becomes]}` : null,
          GONE.has(d.cat) && d.keeps.length ? `Keeps ${plural(d.keeps.length, "fact")}, ${d.keeps.filter((f) => f.sent).length} sent` : null,
          tipQuote(preview(d.words)), h("div", { class: "tip-row muted" }, "Click to open it")); },
      });
      mapCellCache.set(t.turn_index, el);
    }
    el.className = `cell f-${cat}${t.becomes ? " will" : ""}${query() ? (turnMatches(t, said, facts) ? " hit" : " miss") : ""}`;
    el.style.cssText = t.becomes ? `--b:var(--fid-${BECOMES_CAT[t.becomes]})` : "";
    el._tip = { cat, turnIndex: t.turn_index, role, keeps, words, becomes: t.becomes };
    return el;
  };
  const pendingCell = (p) => h("div", { class: "cell pending", onclick: () => jumpTo(p.turn_index), onpointerleave: hideTooltip,
    onpointerenter: (e) => showTooltip(e, `Message ${p.turn_index} · ${speaker(p.role)}`, "Your newest message: recorded, with its reply, once the next request arrives.", tipQuote(preview(p.text))) });

  const widest = (n) => n * CELL_MAX_PX + (n - 1) * 2;
  setChildren(map, groups.map((g) => {
    const n = g.turns.length + g.pending.length;
    const recorded = g.turns.length;
    const span = recorded ? (recorded === 1 ? `#${g.turns[0].turn_index}` : `#${g.turns[0].turn_index}–${g.turns.at(-1).turn_index}`) : "";
    return h("div", { class: `map-group g-${g.key}`, style: `flex:${n} 1 0px;max-width:${widest(n)}px` },
      h("div", { class: "map-label", ...hoverTip(`${g.label}: ${plural(recorded, "message")} ${span}`, GLOSSARY[g.term][1],
          g.pending.length ? "The last cell is your newest message, recorded with its reply." : null) },
        h("b", {}, g.label), countEl(`map-${g.key}`, recorded), g.pending.length ? h("span", {}, "+1 newest") : null,
        h("span", { class: "num range" }, span)),
      h("div", { class: "cells" }, g.turns.map(cell), g.pending.map(pendingCell)));
  }));
  // The end labels span the cells, not the card: a short conversation is a short row.
  $("map-ends").style.maxWidth = `${groups.reduce((sum, g) => sum + widest(g.turns.length + g.pending.length), 0) + (groups.length - 1) * 10}px`;
}

export function renderLegend(k, counts) {
  const nextCount = k.turns.filter((t) => t.becomes).length;
  const item = (key, label, count, glossaryKey, mark) => h("button", {
    class: "legend-item", type: "button", "aria-pressed": String(ui.filter === key),
    onclick: () => { setFilter(ui.filter === key ? "all" : key); $("messages").scrollIntoView({ behavior: "smooth", block: "start" }); },
    ...hoverTip(GLOSSARY[glossaryKey][0], GLOSSARY[glossaryKey][1], "Click to list these messages"),
  }, mark, label, h("b", {}, countEl(`legend-${key}`, count)));
  fill("legend",
    ORDER.filter((key) => counts[key]).map((key) => item(key, CATS[key].label, counts[key], key, swatch(key))),
    nextCount ? item("next", "Taken by the next clean-up", nextCount, "next", h("span", { class: "swatch will-swatch" })) : null);
}

export function renderSize(k) {
  const written = captured() ? state.transcript.recorded_tokens : null;
  const sent = k.context_tokens;
  const scale = Math.max(written ?? 0, sent, 1);
  const tail = sent - k.front_tokens;
  const width = (n) => `${Math.max(0, (n / scale) * 100)}%`;
  const saved = written === null ? null : written - sent;
  fill("panel-size",
    h("div", { class: "stat-label" }, "Tokens per request", info("size")),
    h("div", { class: "big" }, countEl("size-sent", sent), h("small", {}, written === null ? " sent to the model" : ` sent, of ${fmt(written)} written`)),
    h("div", { class: "bars" },
      written === null ? null : [h("span", { class: "lab" }, "As written"),
        h("div", { class: "bar-track" }, h("div", { class: "seg written", style: `width:${width(written)};border-radius:0 4px 4px 0` })),
        h("span", { class: "val" }, fmt(written))],
      h("span", { class: "lab" }, "Sent"),
      h("div", { class: "bar-track" },
        k.front_tokens ? h("div", { class: "seg front", style: `width:${width(k.front_tokens)}`, ...hoverTip("The condensed block", `${fmt(k.front_tokens)} tokens standing in for every condensed message`) }) : null,
        h("div", { class: "seg tail", style: `width:${width(tail)}`, ...hoverTip("Recent messages", `${fmt(tail)} tokens, word for word`) })),
      h("span", { class: "val" }, fmt(sent))),
    k.front_tokens ? h("div", { class: "seg-key" },
      h("span", {}, h("span", { class: "key", style: "background:var(--fid-summarised)" }), `condensed block ${fmt(k.front_tokens)}`),
      h("span", {}, h("span", { class: "key", style: "background:var(--fid-verbatim)" }), `recent messages ${fmt(tail)}`)) : null,
    h("div", { class: "note" }, saved === null ? "The words were not captured (the dashboard started after this conversation), so there is nothing to compare with."
      : k.cuts === 0 ? "Nothing condensed yet: the model gets the whole conversation."
      : saved >= 0 ? [h("b", { class: "secondary" }, `${fmt(saved)} tokens (${percent(saved, written)}%) saved`), ` on every request from now on, after ${plural(k.cuts, "clean-up")}.`]
      : `The condensed block is ${fmt(-saved)} tokens larger than what it replaced.`));
}

export function renderNext(k) {
  const cut = k.next_cut;
  const share = cut.min_cut_tokens > 0 ? cut.compactable_tokens / cut.min_cut_tokens : (cut.compactable_tokens > 0 ? 1 : 0);
  const becoming = {};
  for (const t of k.turns) if (t.becomes) becoming[t.becomes] = (becoming[t.becomes] || 0) + 1;
  const plan = Object.entries(becoming).map(([key, n]) => `${n} ${BECOMES[key]}`).join(" · ");
  const why = cut.batch_ready ? [h("span", { class: "st good" }, "✓ "), "Enough: a clean-up runs after the next reply."]
    : cut.turns.length === 0 ? `Nothing yet: the newest ${keepLast()} messages always stay word for word.`
    : cut.compactable_tokens === 0 ? `The ${plural(cut.turns.length, "older message")} queued are all kept whole, so there is nothing to condense yet.`
    : `${fmt(cut.min_cut_tokens - cut.compactable_tokens)} more needed, or press Run GC now.`;
  fill("panel-next",
    h("div", { class: "stat-label" }, "Next clean-up", info("ready")),
    h("div", { class: "big" }, countEl("next-ready", cut.compactable_tokens), h("small", {}, ` of ${fmt(cut.min_cut_tokens)} tokens ready`)),
    h("div", { class: `meter${cut.batch_ready ? " ready" : ""}`, role: "meter", "aria-valuemin": "0", "aria-valuemax": String(cut.min_cut_tokens), "aria-valuenow": String(cut.compactable_tokens), "aria-label": "Tokens ready to condense" },
      meterBar("next-ready", share * 100)),
    plan ? h("div", { class: "note" }, `${plural(cut.turns.length, "message")} queued: ${plan}`) : null,
    h("div", { class: "note" }, why));
}

export function renderWindow(k) {
  const w = state.config ? state.config.context_window : null;
  const share = w ? k.context_tokens / w : 0;
  fill("panel-window",
    h("div", { class: "stat-label" }, "The model's window", info("window")),
    h("div", { class: "big" }, w ? [countEl("window-pct", share * 100, { format: (v) => (v < 0.1 && v > 0 ? "<0.1" : v.toFixed(v < 10 ? 1 : 0)) }), "%"] : "—", h("small", {}, " used")),
    h("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": String(w || 0), "aria-valuenow": String(k.context_tokens), "aria-label": "Share of the model's window" },
      meterBar("window-share", share * 100)),
    h("div", { class: "note" }, w ? `${fmt(k.context_tokens)} of ${fmt(w)} tokens, before your next message` : "window size not reported"));
}

// ----------------------------------------------------------------- the conversation: what needs attention

export function renderAttention() {
  const k = state.knowledge;
  if (!k) return;
  const counts = countCats(k);
  const unsent = k.facts.filter((f) => f.status === "active" && !f.sent).length;
  const contradicted = k.facts.filter((f) => f.status === "superseded").length;
  const problems = state.activity ? state.activity.recent_degradations.length : 0;
  const items = [];
  if (counts.unseen) items.push({ tone: "warn", icon: "!", title: `The model sees nothing of ${plural(counts.unseen, "message")}`,
    text: `Facts were pulled out of them, but the condensed block carries at most ${maxFacts()} facts and theirs did not make it. If a later answer needs one of these, the model does not have it.`,
    action: ["Show these messages", () => { setFilter("unseen"); $("messages").scrollIntoView({ behavior: "smooth", block: "start" }); }] });
  if (unsent) items.push({ tone: "warn", icon: "!", title: `${plural(unsent, "fact")} stored but not sent`,
    text: "Kept by LLM-GC, ranked below the facts the condensed block has room for.",
    action: ["Show the facts", () => $("memory-unsent")?.scrollIntoView({ behavior: "smooth", block: "start" })] });
  if (problems) items.push({ tone: "warn", icon: "!", title: `${plural(problems, "clean-up problem")}`,
    text: "Part of a clean-up could not be done, so those messages stayed word for word. Nothing is lost; the prompt is bigger than planned.",
    action: ["See the problems", () => { showTab("engine"); $("problems").scrollIntoView({ behavior: "smooth", block: "center" }); }] });
  if (!items.length) items.push({ tone: "good", icon: "✓", title: k.cuts ? "Nothing is out of view" : "The model sees the whole conversation",
    text: k.cuts ? "Every condensed message reaches the model quoted, summarised, or through a fact that is sent." : "Every message is sent word for word; nothing has been condensed yet." });
  if (contradicted) items.push({ tone: "", icon: "↺", title: `${plural(contradicted, "fact")} contradicted later`,
    text: "A later message changed them, so the newer fact is sent and the old one is not. Check the replacement is the one you meant.",
    action: ["Show them", () => { setFilter("superseded"); $("messages").scrollIntoView({ behavior: "smooth", block: "start" }); }] });
  if (k.next_cut.batch_ready) items.push({ tone: "", icon: "→", title: "A clean-up runs after the next reply",
    text: `Messages ${ranges(k.next_cut.turns)} will be condensed. Each one shows what it will become.`,
    action: ["Show them", () => { setFilter("next"); $("messages").scrollIntoView({ behavior: "smooth", block: "start" }); }] });
  if (!captured()) items.push({ tone: "", icon: "i", title: "What was said was not captured",
    text: "The dashboard started after this conversation, so it shows what the model receives but not the original words." });
  fill("attention", items.map((item) => h("div", { class: `callout${item.tone ? ` tone-${item.tone}` : ""}` },
    h("span", { class: "icon", "aria-hidden": "true" }, item.icon), h("div", { class: "title" }, item.title),
    h("div", { class: "text" }, item.text),
    item.action ? h("button", { class: "link-btn", type: "button", onclick: item.action[1] }, `${item.action[0]} →`) : null)));
}

// ----------------------------------------------------------------- the conversation: message by message

export function setFilter(key) {
  ui.filter = key;
  ui.listOpen = true;
  if (ui.promptView) setPromptView(false);
  renderMessages();
  if (state.knowledge) renderLegend(state.knowledge, countCats(state.knowledge));
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
    nextCount ? chip("next", "Next clean-up", nextCount, h("span", { class: "swatch will-swatch" })) : null, toggle);
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
  const items = shown.map((row) => row.kind === "run" ? runRow(row, said, facts)
    : row.kind === "summary" ? summaryRow(row, said) : row.kind === "pending" ? pendingRow(row.pending) : turnRow(row.turn, said, facts));
  const timeline = $("timeline");
  timeline.hidden = !ui.listOpen;
  const note = $("filter-note");
  note.hidden = !filtered || !ui.listOpen;
  if (filtered) {
    const shownCount = visible.reduce((n, row) => n + (row.kind === "summary" ? row.turns.length : 1), 0);
    setChildren(note, `Showing ${plural(shownCount, "message")}${query() ? ` matching “${ui.query.trim()}”` : ""}${ui.filter !== "all" ? ` · ${ui.filter === "next" ? "next clean-up" : CATS[ui.filter].tag}` : ""}`,
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

export function factRow(fact, { withFrom = true, onclick = null, besideItsTurn = false } = {}) {
  const raw = fact.knowledge_type === "raw";
  // Beside its own message, an excerpt of that whole message would only repeat the words above it.
  const wholeTurn = besideItsTurn && raw && fact.source_turns.length === 1;
  const status = fact.status === "superseded"
    ? h("span", {}, h("span", { class: "st" }, "↺ "), fact.superseded_by_turn !== null ? `replaced by #${fact.superseded_by_turn}` : "replaced")
    : fact.sent ? h("span", {}, h("span", { class: "st good" }, "✓ "), "sent") : h("span", {}, h("span", { class: "st warn" }, "⚠ "), "not sent");
  return h(onclick ? "button" : "div", { class: `fact s-${fact.status}`, type: onclick ? "button" : null, onclick,
    title: onclick ? "Show the message it came from" : null },
    h("span", { class: "fact-topic", title: raw ? "an excerpt: nothing more specific could be pulled out" : fact.topic }, raw ? "excerpt" : fact.topic),
    h("span", { class: "fact-what" }, wholeTurn ? h("span", { class: "muted" }, "the message itself, kept as an excerpt (nothing more specific could be pulled out)") : marked(fact.content)),
    h("span", { class: "fact-meta" }, withFrom ? h("span", { class: "num" }, `#${ranges(fact.source_turns)}`) : null, status));
}

export function nextTag(becomes) {
  return h("span", { class: "tag next", ...hoverTip("Next clean-up", `This message will be ${BECOMES[becomes]} when the next clean-up runs.`) },
    h("span", { class: `swatch f-${BECOMES_CAT[becomes]}` }), `next: ${BECOMES[becomes]}`);
}

export function turnRow(turn, said, facts) {
  const cat = categoryOf(turn);
  const words = said.get(turn.turn_index);
  const kept = facts.get(turn.turn_index) || [];
  const key = `t-${turn.turn_index}`;
  let body;
  if (cat === "verbatim") body = textBlock(turn.text, { key });
  else if (cat === "quoted") body = textBlock(words ? words.text : state.knowledge.front_entries[turn.front_entry].stated_text, { key });
  else {
    const intro = cat === "unseen" ? [h("span", { class: "st warn" }, "⚠ "), "The model sees nothing of this message. Stored, but not sent:"]
      : cat === "superseded" ? "Contradicted later, so no longer sent:" : "The model no longer sees this message. It keeps:";
    body = [textBlock(words?.text, { faded: true, key }), h("div", { class: "keeps" }, h("div", { class: "keeps-intro" }, intro),
      kept.length ? kept.map((f) => factRow(f, { withFrom: false, besideItsTurn: true })) : h("div", { class: "lost" }, "nothing"))];
  }
  return h("article", { class: `msg f-${cat}`, id: `turn-${turn.turn_index}` },
    h("div", { class: "msg-no" }, turn.turn_index),
    h("div", { class: "msg-rail" }),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" },
        h("span", { class: "who" }, speaker(words ? words.role : turn.role)),
        fidTag(cat),
        turn.becomes ? nextTag(turn.becomes) : null,
        turn.tokens ? h("span", { class: "num" }, plural(turn.tokens, "token")) : null),
      body));
}

export function summaryRow(row, said) {
  const { turns, entry } = row;
  const first = turns[0].turn_index, last = turns.at(-1).turn_index;
  const key = `orig-${first}`;
  const open = ui.open.has(key) || (query() && turns.some((t) => has(said.get(t.turn_index)?.text)));
  return h("article", { class: "msg f-summarised", id: `entry-${first}` },
    h("div", { class: "msg-no" }, first === last ? first : `${first}–${last}`),
    h("div", { class: "msg-rail" }),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, plural(turns.length, "message")), fidTag("summarised"),
        h("span", {}, "the model reads this summary instead of what was said")),
      h("div", { class: "summary-box" }, h("div", { class: "label" }, `Summary · ${plural(entry.tokens, "token")}`), h("div", { class: "text" }, marked(entry.stated_text))),
      h("button", { class: "more orig-toggle", type: "button", "aria-expanded": String(!!open),
        onclick: () => { if (ui.open.has(key)) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
        `${open ? "Hide" : "Show"} what was said in ${turns.length > 1 ? `messages ${first}–${last}` : `message ${first}`}`),
      open ? turns.map((t) => h("div", { class: "orig", id: `turn-${t.turn_index}` },
        h("div", { class: "msg-head" }, h("span", { class: "who" }, speaker(said.get(t.turn_index)?.role)), h("span", { class: "num" }, `message ${t.turn_index}`)),
        textBlock(said.get(t.turn_index)?.text, { faded: true, key: `t-${t.turn_index}` }))) : null));
}

export function pendingRow(p) {
  return h("article", { class: "msg f-verbatim pending", id: `turn-${p.turn_index}` },
    h("div", { class: "msg-no" }, p.turn_index), h("div", { class: "msg-rail" }),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, speaker(p.role)), h("span", { class: "tag" }, "newest message"),
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
  target.scrollIntoView({ behavior: "smooth", block: "center" });
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
      k.front_text ? "The condensed block stays byte for byte the same until the next clean-up, so the provider can keep it cached."
        : "Nothing has been condensed, so your app's own request is forwarded untouched."), copy),
    k.front_text ? h("div", { class: "prompt-msg front" }, h("div", { class: "msg-head" }, h("span", { class: "who" }, "user"),
      h("span", { class: "tag" }, swatch("summarised"), `the condensed block · ${plural(k.front_tokens, "token")}`), info("block")), h("pre", {}, marked(k.front_text))) : null,
    tail.map((t) => h("div", { class: "prompt-msg" }, h("div", { class: "msg-head" }, h("span", { class: "who" }, t.role),
      h("span", { class: "num" }, `message ${t.turn_index} · ${plural(t.tokens, "token")}`)), h("pre", {}, marked(t.text || "(no text)")))));
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
      note: `The condensed block carries at most ${maxFacts()} facts, most important first. The model cannot see these.` },
    { id: "superseded", title: "Contradicted later", icon: h("span", { class: "st" }, "↺"), facts: k.facts.filter((f) => f.status === "superseded") },
  ];
  if (!k.facts.length) {
    fill("memory", h("p", { class: "empty" }, "Nothing has been reduced to facts yet. Facts appear when a clean-up condenses a message that has gone cold."));
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
