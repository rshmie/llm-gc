// The engine view: what LLM-GC is doing to this conversation, and why. When the next collection
// runs and why each message gets what it gets, what collections did and couldn't do, every check,
// and last the proxy's totals and settings, which apply to every conversation.

import { BECOMES, BECOMES_CAT, CARD_INFO, CHART_HEIGHT, CHECKS_SHOWN, MOVES_SHOWN, MOVE_CAT, MOVE_KIND, NOT_MOVED, STATUS, TRANSITION, describeOutcome, swatch } from "./vocabulary.js";
import { $, agoTs, compact, countEl, fill, fmt, h, meterBar, plural, reducedMotion, setChildren, storeGet, storeSet, svg } from "./dom.js";
import { captured, keepLast, preview, state, ui } from "./state.js";
import { hideTooltip, info, placeTooltip, showTooltip, tipQuote, tooltipContent, hoverTip } from "./tooltip.js";
import { jumpTo, speakerName, trace } from "./conversation.js";
import { showTab } from "./shell.js";

const smooth = () => (reducedMotion() ? "auto" : "smooth");
const chevron = () => svg("svg", { width: 14, height: 14, viewBox: "0 0 16 16", "aria-hidden": "true" },
  svg("path", { d: "M4 6l4 4 4-4", fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round" }));

export function statusChip(status) {
  const s = STATUS[status] || { tone: "", icon: "?", text: status };
  return h("span", { class: "status-chip" }, h("span", { class: `dot ${s.tone}` }), `${s.icon} ${s.text}`);
}

// Each message number links to it in the conversation; a run of them is one link to its first.
function messageLinks(numbers) {
  const sorted = [...new Set(numbers)].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < sorted.length; i++) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    const first = sorted[i];
    out.push(out.length ? ", " : "", h("button", { class: "link-btn", type: "button", onclick: () => jumpTo(first) }, j > i ? `${first}–${sorted[j]}` : String(first)));
    i = j;
  }
  return [sorted.length === 1 ? "message " : "messages ", out];
}

// ----------------------------------------------------------------- cards: collapse and ⓘ

const CARDS_KEY = "llmgc.cards";
const COLLAPSED_BY_DEFAULT = { settings: true };
let cardState = {};

function setCard(card, collapsed) {
  const button = card.querySelector(".card-toggle");
  const title = card.querySelector("h2").firstChild.textContent;
  card.querySelector(".card-body").hidden = collapsed;
  card.classList.toggle("collapsed", collapsed);
  button.setAttribute("aria-expanded", String(!collapsed));
  button.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${title}`);
  button.title = collapsed ? "Expand" : "Collapse";
  if (card.dataset.card === "chart" && !collapsed) { ui.chartKey = null; renderChart(); }
}

export function initCards() {
  try { cardState = JSON.parse(storeGet(CARDS_KEY) || "{}"); } catch { cardState = {}; }
  for (const card of document.querySelectorAll("[data-card]")) {
    const key = card.dataset.card;
    setCard(card, cardState[key] ?? !!COLLAPSED_BY_DEFAULT[key]);
    card.querySelector(".card-toggle").addEventListener("click", () => {
      const collapsed = !card.classList.contains("collapsed");
      setCard(card, collapsed);
      cardState[key] = collapsed;
      storeSet(CARDS_KEY, JSON.stringify(cardState));
    });
  }
  for (const button of document.querySelectorAll("[data-info]")) {
    const [title, text] = CARD_INFO[button.dataset.info];
    button.addEventListener("pointerenter", (e) => showTooltip(e, title, text));
    button.addEventListener("pointerleave", hideTooltip);
    button.addEventListener("focus", (e) => showTooltip(e, title, text));
    button.addEventListener("blur", hideTooltip);
  }
}

export function openCard(key, scroll = false) {
  const card = document.querySelector(`[data-card="${key}"]`);
  if (!card) return;
  if (card.classList.contains("collapsed")) setCard(card, false);
  if (scroll) card.scrollIntoView({ behavior: smooth(), block: "start" });
}

// ----------------------------------------------------------------- all conversations, and health

export function renderProxy() {
  const p = state.proxy;
  $("proxy-details").hidden = !p;
  if (!p) {
    fill("proxy-line", h("span", { class: "label" }, "All conversations"), "No proxy beside this dashboard, so there are no totals.");
    fill("tiles");
    return;
  }
  fill("proxy-line", h("span", { class: "label" }, "All conversations"),
    h("span", {}, h("b", {}, fmt(p.requests)), " requests"), h("span", { class: "sep" }, "·"),
    h("span", {}, h("b", {}, fmt(p.tokens_forwarded)), ` tokens sent in total${p.ratio !== null ? `, ${Math.round(p.ratio * 100)}% of what your apps sent` : ""}`), h("span", { class: "sep" }, "·"),
    p.gc_bypassed ? h("span", {}, h("span", { class: "st warn" }, "⚠ "), `${fmt(p.gc_bypassed)} passed on untouched after an error`) : h("span", {}, "none passed on after an error"),
    h("button", { class: "link-btn", type: "button", onclick: () => $("proxy-details").scrollIntoView({ behavior: smooth(), block: "start" }) }, "Details ↓"));
  const tile = (label, term, value, caption, extra) => h("section", { class: "card tile" },
    h("div", { class: "label" }, label, term ? info(term) : null), value === null ? null : h("div", { class: "value" }, value), extra || null,
    caption ? h("div", { class: "caption" }, caption) : null);
  fill("tiles",
    tile("Requests through the proxy", null, countEl("tile-requests", p.requests),
      [`${fmt(p.gc_applied)} handled by LLM-GC`, p.gc_bypassed ? [" · ", h("span", { class: "st warn" }, "⚠ "), `${fmt(p.gc_bypassed)} passed on untouched after an error`] : " · none passed on after an error"]),
    tile("Total sent since the proxy started", "forwarded", countEl("tile-forwarded", p.tokens_forwarded),
      `${p.ratio !== null ? `${Math.round(p.ratio * 100)}% of` : "of"} the ${fmt(p.tokens_in_client_history)} your apps sent, all conversations`,
      p.ratio !== null ? h("div", { class: "meter", "aria-hidden": "true" }, meterBar("tile-forwarded-ratio", p.ratio * 100)) : null),
    tile("Conversation tracking", "tracking", null, null,
      h("div", { class: "triple" }, [[p.sessions_seeded, "started fresh"], [p.history_rewinds, "rewound after an edit"], [p.history_restarts, "restarted"]]
        .map(([n, label]) => h("div", {}, h("b", {}, fmt(n)), h("span", {}, label))))));
}

// One line answers "is LLM-GC healthy?": the last check, and any move a collection couldn't make.
export function renderEngineStatus() {
  const a = state.activity;
  const last = a && a.gc_breakdown;
  const problems = a ? a.recent_degradations.length : 0;
  $("view-engine").classList.toggle("has-problems", problems > 0);
  const failed = !!last && last.gc_status === "bypassed_on_error";
  const tone = failed ? "critical" : problems ? "warn" : "good";
  const lastText = last ? `Last check ${agoTs(last.gc_completed_at)}: ${STATUS[last.gc_status]?.text || last.gc_status}, ${last.duration_ms.toFixed(0)} ms.`
    : "No checks yet. LLM-GC checks after every reply.";
  fill("eng-status", h("div", { class: `status-line tone-${tone}` },
    h("span", { class: "badge", "aria-hidden": "true" }, failed ? "✕" : problems ? "!" : "✓"),
    h("span", { class: "msg-text" }, h("b", {}, failed ? "The last check failed, and nothing changed." : problems ? `${plural(problems, "planned move")} didn't happen.` : "LLM-GC is working normally."),
      ` ${lastText}`, failed && last.failure_reason ? ` ${last.failure_stage}: ${last.failure_reason}` : null),
    problems ? h("button", { class: "link-btn", type: "button", onclick: () => openCard("notmoved", true) }, "See why →")
      : h("span", { class: "then" }, "Every move a collection planned was made.")));
}

// ----------------------------------------------------------------- what happens next, and why

// The newest pass that planned a cut, whether it was made or refused.
function lastAttempt() {
  const passes = state.activity ? state.activity.recent_passes : [];
  return passes.findLast((p) => p.abandoned_cut || (p.gc_status === "completed" && p.outcome)) || null;
}

// A collection has two conditions to meet, so they are shown as a checklist, each with its state,
// beside what the collection would do to which messages - and which of those count toward the first.
export function renderNext(k) {
  const cut = k.next_cut;
  const taking = k.turns.filter((t) => t.becomes);
  const tokensOf = (turns) => turns.reduce((sum, t) => sum + (t.tokens || 0), 0);
  const refused = lastAttempt()?.abandoned_cut ? lastAttempt() : null;
  const share = cut.min_cut_tokens > 0 ? Math.min(1, cut.compactable_tokens / cut.min_cut_tokens) : (cut.compactable_tokens > 0 ? 1 : 0);
  const newest = k.turns.filter((t) => t.fidelity === "verbatim" && !t.becomes).length;

  const lede = !taking.length ? "Nothing to collect yet."
    : cut.batch_ready ? "Ready: it runs after the next reply, if the result is smaller."
    : `Waiting for ${fmt(cut.min_cut_tokens - cut.compactable_tokens)} more tokens of messages to compact or archive.`;
  const check = (tone, icon, title, figure, ...body) => h("li", { class: `ck tone-${tone}` },
    h("span", { class: "ck-badge", "aria-hidden": "true" }, icon),
    h("div", {}, h("div", { class: "ck-head" }, h("b", {}, title), figure), body));

  const row = (key, label, note) => {
    const turns = taking.filter((t) => t.becomes === key);
    return turns.length ? h("li", {}, swatch(BECOMES_CAT[key]),
      h("span", {}, h("b", {}, plural(turns.length, "message")), ` ${label}`, note ? h("span", { class: "would-note" }, ` (${note})`) : null),
      h("span", { class: "num" }, `${fmt(tokensOf(turns))} tokens`)) : null;
  };
  const counted = [row("summarised", "compacted"), row("facts_only", "archived")].filter(Boolean);
  const notCounted = [row("quoted", "kept whole", "short or rated important"),
    newest ? h("li", {}, swatch("verbatim"), h("span", {}, h("b", {}, `The newest ${plural(newest, "message")}`), " stay word for word"), h("span")) : null].filter(Boolean);

  fill("panel-next",
    h("p", { class: "next-lede" }, lede),
    h("div", { class: "next-grid" },
      h("div", {},
        h("div", { class: "label" }, "It runs when both hold"),
        h("ul", { class: "ck-list" },
          check(cut.batch_ready ? "good" : "wait", cut.batch_ready ? "✓" : "…", "Enough to collect",
            h("span", { class: "num" }, countEl("next-ready", cut.compactable_tokens), ` of ${fmt(cut.min_cut_tokens)} tokens`),
            h("div", { class: `meter${cut.batch_ready ? " ready" : ""}`, role: "meter", "aria-valuemin": "0", "aria-valuemax": String(cut.min_cut_tokens),
              "aria-valuenow": String(cut.compactable_tokens), "aria-label": "Tokens of messages to compact or archive" }, meterBar("next-ready", share * 100)),
            h("p", {}, "Only messages to be compacted or archived count. Keeping a message whole saves nothing.")),
          refused
            ? check("warn", "!", "Makes the prompt smaller", h("span", { class: "num" }, `${fmt(refused.tokens_before)} → ${fmt(refused.abandoned_cut.context_tokens_after)}`),
              h("p", {}, `The last try, ${agoTs(refused.gc_completed_at)}, would have made it bigger, so nothing changed. Each message in the recap carries a short label, so collecting a few short messages costs more than it saves.`))
            : check("wait", "…", "Makes the prompt smaller", null,
              h("p", {}, "Checked as it runs, because the summaries aren't written until then. If the result would be bigger, nothing changes.")))),
      h("div", {},
        h("div", { class: "label" }, "What it would do now"),
        counted.length ? [h("div", { class: "would-head" }, `Counts toward the ${fmt(cut.min_cut_tokens)}`), h("ul", { class: "would" }, counted)] : null,
        notCounted.length ? [h("div", { class: "would-head" }, "Doesn't count"), h("ul", { class: "would" }, notCounted)] : null,
        !counted.length && !notCounted.length ? h("p", { class: "note" }, "Nothing yet.") : null,
        taking.length ? h("button", { class: "link-btn", type: "button", onclick: () => openCard("rate", true) }, "Why each message gets this ↓") : null)));
}

// Why a message gets what it gets: the engine's own classification, worded with your thresholds.
export function whyText(t) {
  const c = state.config;
  const low = c ? c.archive_threshold : 0.3, high = c ? c.keep_threshold : 0.7;
  if (t.score === null || t.score === undefined) return "Not rated.";
  const s = t.score.toFixed(2);
  if (t.forced_keep) {
    if (t.forced_keep.includes("too short")) return `Kept whole: too short to compact (${c ? `${c.min_compactable_tokens} tokens or fewer` : "very short"}), whatever its relevance of ${s}.`;
    if (t.forced_keep.includes("last_n")) {
      const later = t.score >= high ? "kept (quoted whole)" : t.score < low ? "archived (reduced to facts)" : "compacted (summarised)";
      return `Kept word for word: among the newest ${keepLast()}, whatever its relevance. At ${s} it would be ${later} once older.`;
    }
    return `Kept whole by a rule (relevance ${s}).`;
  }
  if (t.classification === "keep") return `Relevance ${s}: at or above ${high.toFixed(2)}, so kept (quoted whole).`;
  if (t.classification === "archive") return `Relevance ${s}: below ${low.toFixed(2)}, so archived (reduced to facts).`;
  return `Relevance ${s}: between ${low.toFixed(2)} and ${high.toFixed(2)}, so compacted (summarised).`;
}
const DOT = { keep: "quoted", compact: "summarised", archive: "facts" };

// The ratings lead with the messages the next collection takes; the rest fold under one line, and a
// long list scrolls inside the card with the threshold labels pinned.
export function renderRelevance(k) {
  const c = state.config;
  const low = c ? c.archive_threshold : 0.3, high = c ? c.keep_threshold : 0.7;
  const plot = $("relevance");
  plot.style.setProperty("--a", `${low * 100}%`);
  plot.style.setProperty("--k", `${high * 100}%`);
  const rated = k.turns.filter((t) => t.fidelity === "verbatim" && t.score !== null);
  const condensed = k.turns.filter((t) => t.fidelity !== "verbatim");
  if (!rated.length && !condensed.length) { setChildren(plot, h("p", { class: "empty" }, "No messages to rate yet.")); return; }
  const taking = rated.filter((t) => t.becomes), staying = rated.filter((t) => !t.becomes);
  const stayOpen = ui.stayOpen || !taking.length;
  const row = (t) => h("div", { class: "rate-row", id: `r-${t.turn_index}`, onpointerenter: () => trace([t.turn_index]), onpointerleave: () => trace(null) },
    h("div", { class: "rate-lab" }, `#${t.turn_index} · ${t.role === "user" ? "You" : "AI"}`),
    h("div", { class: "rate-track" }, h("span", { class: `rate-dot${t.forced_keep ? " forced" : ` f-${DOT[t.classification]}`}`, style: `left:${t.score * 100}%`,
      ...hoverTip(`Message ${t.turn_index} · ${speakerName(t.role)}`, whyText(t), tipQuote(preview(t.text, 120))) })),
    h("div", { class: "rate-why" }, whyText(t), t.becomes ? h("div", { class: "then" }, `→ next collection: ${BECOMES[t.becomes]}`) : null));
  const head = (label, n, open, toggles) => toggles
    ? h("button", { class: "rate-group", type: "button", "aria-expanded": String(open), onclick: () => { ui.stayOpen = !ui.stayOpen; renderRelevance(state.knowledge); } },
      chevron(), label, h("span", { class: "n" }, `· ${n}`))
    : h("div", { class: "rate-group" }, label, h("span", { class: "n" }, `· ${n}`));
  setChildren(plot,
    h("div", { class: "rate-list" },
      h("div", { class: "rate-row rate-axis", "aria-hidden": "true" }, h("div"),
        h("div", { class: "rate-track" },
          h("span", { class: "zone", style: `left:${(low / 2) * 100}%` }, `archive · below ${low}`),
          h("span", { class: "zone", style: `left:${((low + high) / 2) * 100}%` }, "compact"),
          h("span", { class: "zone", style: `left:${((high + 1) / 2) * 100}%` }, `keep · ${high}+`)), h("div")),
      taking.length ? [head("In the next collection", taking.length, true, false), taking.map(row)] : null,
      staying.length ? [head("Staying word for word", staying.length, stayOpen, taking.length > 0), stayOpen ? staying.map(row) : null] : null,
      condensed.length ? h("div", { class: "rate-gone" },
        condensed.length === 1 ? `Message ${condensed[0].turn_index} left the young generation in an earlier collection, so it is no longer rated. `
          : `${plural(condensed.length, "message")} left the young generation in earlier collections, so they are no longer rated. `,
        h("button", { class: "link-btn", type: "button", onclick: () => openCard("moves", true) }, "See what moved")) : null),
    h("div", { class: "rate-legend" },
      h("span", {}, h("span", { class: "rate-dot f-facts" }), "archive"),
      h("span", {}, h("span", { class: "rate-dot f-summarised" }), "compact"),
      h("span", {}, h("span", { class: "rate-dot f-quoted" }), "keep"),
      h("span", {}, h("span", { class: "rate-dot forced" }), "kept whole by a rule")));
}

// A "next:" tag in the conversation lands here, on its message's rating.
export function showWhy(turnIndex) {
  showTab("engine");
  openCard("rate");
  if (!$(`r-${turnIndex}`)) { ui.stayOpen = true; renderRelevance(state.knowledge); }
  const row = $(`r-${turnIndex}`);
  if (!row) return;
  row.scrollIntoView({ behavior: smooth(), block: "center" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1600);
}

// ----------------------------------------------------------------- what has happened

export function renderHistory() {
  const transitions = state.activity ? state.activity.generation_lifecycle.recent_transitions : [];
  // Moves made by one collection arrive together; group them by time.
  const groups = [];
  for (const t of transitions) {
    const lastGroup = groups.at(-1);
    if (lastGroup && t.occurred_at - lastGroup.at < 2) lastGroup.items.push(t);
    else groups.push({ at: t.occurred_at, items: [t] });
  }
  groups.reverse();
  const shown = ui.allMoves ? groups : groups.slice(0, MOVES_SHOWN);
  fill("history", groups.length ? [
    h("ul", { class: "events" }, shown.map((g) => {
      const byType = {};
      for (const t of g.items) (byType[t.transition_type] ||= []).push(t);
      const archived = new Set((byType.archived || []).map((t) => t.turn_index));
      const twice = (byType.promoted || []).filter((t) => archived.has(t.turn_index));
      if (twice.length) {
        const turns = new Set(twice.map((t) => t.turn_index));
        byType.promoted = byType.promoted.filter((t) => !turns.has(t.turn_index));
        byType.archived = byType.archived.filter((t) => !turns.has(t.turn_index));
        byType.promoted_then_archived = twice;
        for (const type of ["promoted", "archived"]) if (!byType[type].length) delete byType[type];
      }
      return h("li", {},
        h("div", { class: "event-head" }, h("b", {}, "Collection"), h("span", { class: "when" }, agoTs(g.at))),
        h("div", { class: "event-lines" }, Object.entries(byType).map(([type, items]) => h("div", {},
          h("span", { class: "k" }, swatch(MOVE_CAT[type] || "verbatim"), TRANSITION[type] || type, MOVE_KIND[type] ? h("span", { class: "kind" }, MOVE_KIND[type]) : null),
          h("span", {}, type === "superseded"
            ? items.map((t, n) => [n ? ", " : "", `“${t.topic_label}” from `, h("button", { class: "link-btn", type: "button", onclick: () => jumpTo(t.turn_index) }, `#${t.turn_index}`)])
            : messageLinks(items.map((t) => t.turn_index)))))));
    })),
    groups.length > MOVES_SHOWN ? h("button", { class: "link-btn show-all", type: "button", onclick: () => { ui.allMoves = !ui.allMoves; renderHistory(); } },
      ui.allMoves ? "Show fewer" : `Show all ${groups.length} collections`) : null,
  ] : h("p", { class: "empty" }, "Nothing has moved yet."));
}

export function renderProblems() {
  const problems = state.activity ? state.activity.recent_degradations.slice().reverse() : [];
  const shown = ui.allProblems ? problems : problems.slice(0, MOVES_SHOWN);
  fill("problems", problems.length ? [
    h("ul", { class: "events" }, shown.map((r) => {
      const [label, kind] = NOT_MOVED[r.degradation.reason] || [r.degradation.reason.replaceAll("_", " "), null];
      return h("li", {},
        h("div", { class: "event-head" }, h("span", { class: "st warn" }, "⚠"), h("b", {}, label), kind ? h("span", { class: "muted" }, kind) : null, h("span", { class: "when" }, agoTs(r.occurred_at))),
        h("div", { class: "event-lines" },
          h("div", {}, h("span", { class: "k" }, "Stayed word for word"), h("span", {}, messageLinks(r.degradation.turn_indices))),
          r.degradation.detail ? h("div", {}, h("span", { class: "k" }, "Detail"), h("span", {}, r.degradation.detail)) : null));
    })),
    problems.length > MOVES_SHOWN ? h("button", { class: "link-btn show-all", type: "button", onclick: () => { ui.allProblems = !ui.allProblems; renderProblems(); } },
      ui.allProblems ? "Show fewer" : `Show all ${problems.length}`) : null,
  ] : h("p", { class: "ok-line" }, h("span", { class: "st good" }, "✓ "), "None so far."));
}

// The conversation as written up to a turn: a running sum of the transcript's own counts.
export function writtenUpTo() {
  if (!captured()) return null;
  const turns = [...state.transcript.turns].sort((a, b) => a.turn_index - b.turn_index);
  const sums = new Map();
  let total = 0;
  for (const t of turns) { total += t.tokens; sums.set(t.turn_index, total); }
  return (turnIndex) => sums.has(turnIndex) ? sums.get(turnIndex) : null;
}

export function niceStep(max, count) {
  const raw = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  return Math.max(1, (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * magnitude);
}

export function renderChart() {
  const host = $("chart");
  const passes = state.activity ? state.activity.recent_passes : [];
  const cutsDone = passes.filter((p) => p.gc_status === "completed" && p.tokens_saved > 0);
  const lastCut = cutsDone.at(-1);
  fill("cuts-stat", h("span", { class: "label", style: "justify-content:flex-end" }, "Collections"),
    h("span", { class: "n" }, state.knowledge ? fmt(state.knowledge.cuts) : "—"),
    lastCut ? `last ${agoTs(lastCut.gc_completed_at)}: ${fmt(lastCut.tokens_before)} → ${fmt(lastCut.tokens_before - lastCut.tokens_saved)}` : "none in the recent checks");
  const width = Math.round(host.clientWidth);
  const key = `${width}|${passes.length}|${passes.at(-1)?.gc_completed_at}|${state.version}|${Math.floor(Date.now() / 30000)}`;
  if (!width || key === ui.chartKey) return;
  ui.chartKey = key;
  if (!passes.length) {
    fill("chart-legend");
    setChildren(host, h("p", { class: "empty" }, "No checks yet. LLM-GC checks after every reply."));
    return;
  }
  const writtenAt = writtenUpTo();
  const pts = passes.map((p, i) => ({ i, p, before: p.tokens_before,
    after: p.gc_status === "completed" ? p.tokens_before - p.tokens_saved : p.tokens_before,
    written: writtenAt && p.newest_turn_index !== null && p.newest_turn_index !== undefined ? writtenAt(p.newest_turn_index) : null }));
  const withWritten = pts.some((d) => d.written !== null);
  const cuts = pts.filter((d) => d.after < d.before);
  const skipped = pts.filter((d) => d.p.abandoned_cut);
  const failed = pts.filter((d) => d.p.gc_status === "bypassed_on_error");

  fill("chart-legend",
    h("span", {}, h("span", { class: "line-key", style: "background:var(--accent)" }), "Sent to the model"),
    withWritten ? h("span", {}, h("span", { class: "line-key", style: "background:var(--written)" }), "The conversation as written") : null,
    // Each mark in the legend is the chart's own marker.
    cuts.length ? h("span", {}, svg("svg", { width: 12, height: 12, viewBox: "0 0 12 12" }, svg("circle", { cx: 6, cy: 6, r: 4.5, fill: "var(--accent)" })), "collection") : null,
    skipped.length ? h("span", {}, svg("svg", { width: 12, height: 12, viewBox: "0 0 12 12" }, svg("circle", { cx: 6, cy: 6, r: 4, fill: "var(--surface)", stroke: "var(--ink-3)", "stroke-width": 1.5 })), "skipped: would not have shrunk the prompt") : null,
    failed.length ? h("span", {}, h("span", { class: "dot critical" }), "failed") : null);

  const narrow = width < 520;
  const M = { l: 48, r: narrow ? 14 : 112, t: 16, b: 30 };
  const pw = Math.max(40, width - M.l - M.r), ph = CHART_HEIGHT - M.t - M.b;
  const maxValue = Math.max(1, ...pts.map((d) => Math.max(d.before, d.written ?? 0)));
  const step = niceStep(maxValue, 4);
  const top = Math.ceil(maxValue / step) * step;
  const x = (i) => M.l + (pts.length === 1 ? pw / 2 : (i / (pts.length - 1)) * pw);
  const y = (v) => M.t + ph - (v / top) * ph;

  const grid = [];
  for (let v = 0; v <= top + 1e-9; v += step) {
    grid.push(svg("line", { class: v === 0 ? "axis" : "grid", x1: M.l, x2: M.l + pw, y1: y(v), y2: y(v) }),
      svg("text", { class: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end" }, compact(v)));
  }
  // Stepped: the prompt holds its size until the next message arrives, then jumps; a collection drops it.
  let sentPath = `M${x(0)},${y(pts[0].before)}`;
  pts.forEach((d, n) => { if (n) sentPath += `H${x(d.i)}V${y(d.before)}`; if (d.after < d.before) sentPath += `V${y(d.after)}`; });
  const areaPath = `${sentPath}H${x(pts.at(-1).i)}V${y(0)}H${x(0)}Z`;
  const writtenPts = pts.filter((d) => d.written !== null);
  const writtenPath = writtenPts.map((d, n) => n ? `H${x(d.i)}V${y(d.written)}` : `M${x(d.i)},${y(d.written)}`).join("");
  const bands = cuts.map((d) => {
    const x0 = Math.max(M.l, x(d.i) - 10), x1 = Math.min(M.l + pw, x(d.i) + 10);
    return svg("rect", { class: "band", x: x0, y: M.t, width: Math.max(0, x1 - x0), height: ph });
  });

  const last = pts.at(-1);
  const ends = [];
  if (!narrow) {
    ends.push(svg("text", { class: "end", x: M.l + pw + 10, y: y(last.after) + 4 }, `sent ${fmt(last.after)}`));
    if (last.written !== null && Math.abs(y(last.written) - y(last.after)) >= 16) {
      ends.push(svg("text", { class: "end", x: M.l + pw + 10, y: y(last.written) + 4, style: "fill:var(--ink-3)" }, `written ${fmt(last.written)}`));
    }
  }
  const cross = svg("line", { class: "cross", x1: 0, x2: 0, y1: M.t, y2: M.t + ph, style: "display:none" });
  const root = svg("svg", { height: CHART_HEIGHT, viewBox: `0 0 ${width} ${CHART_HEIGHT}`, role: "img",
    "aria-label": `Prompt size over the last ${passes.length} checks: now ${fmt(last.after)} tokens sent${last.written !== null ? `, ${fmt(last.written)} as written` : ""}, ${plural(cuts.length, "collection")}.` },
    svg("defs", {}, svg("linearGradient", { id: "g-area", x1: 0, y1: 0, x2: 0, y2: 1 },
      svg("stop", { offset: 0, style: "stop-color:var(--accent);stop-opacity:.28" }), svg("stop", { offset: 1, style: "stop-color:var(--accent);stop-opacity:0" }))),
    grid, bands,
    svg("path", { class: "a-sent", d: areaPath }),
    writtenPath ? svg("path", { class: "l-written", d: writtenPath }) : null,
    svg("path", { class: "l-sent", d: sentPath }),
    pts.length <= 40 ? pts.filter((d) => !(d.after < d.before) && !d.p.abandoned_cut && d.p.gc_status !== "bypassed_on_error")
      .map((d) => svg("circle", { class: "mk-check", cx: x(d.i), cy: y(d.before), r: 3.5 })) : null,
    cuts.map((d) => [svg("circle", { class: "mk-cut", cx: x(d.i), cy: y(d.after), r: 5 }),
      cuts.length <= 8 ? svg("text", { class: "cut-label", x: x(d.i) - 8, y: y(d.after) + 16, "text-anchor": "end" }, `−${fmt(d.before - d.after)}`) : null]),
    skipped.map((d) => svg("circle", { class: "mk-skip", cx: x(d.i), cy: y(d.before), r: 4.5 })),
    failed.map((d) => svg("circle", { class: "mk-error", cx: x(d.i), cy: y(d.before), r: 5 })),
    ends,
    svg("text", { class: "tick", x: M.l, y: CHART_HEIGHT - 8 }, agoTs(pts[0].p.gc_completed_at)),
    svg("text", { class: "tick", x: M.l + pw, y: CHART_HEIGHT - 8, "text-anchor": "end" }, `latest · ${plural(passes.length, "check")}`),
    cross);
  // One hit layer for the whole plot: the pointer snaps to the nearest check.
  const hit = svg("rect", { class: "hit", x: M.l - 8, y: M.t, width: pw + 16, height: ph });
  hit.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const scaleX = box.width / width;
    const at = (event.clientX - box.left) / scaleX;
    const i = Math.max(0, Math.min(pts.length - 1, Math.round(pts.length === 1 ? 0 : ((at - M.l) / pw) * (pts.length - 1))));
    const d = pts[i];
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.style.display = "";
    const figure = (label, value) => h("div", { class: "tip-figure" }, h("span", {}, label), h("b", {}, value));
    tooltipContent(`Check ${i + 1} of ${pts.length} · ${agoTs(d.p.gc_completed_at)}`, [
      statusChip(d.p.gc_status),
      figure("sent to the model", d.after < d.before ? `${fmt(d.before)} → ${fmt(d.after)}` : fmt(d.before)),
      d.written !== null ? figure("as written", fmt(d.written)) : null,
      d.p.abandoned_cut ? figure("a collection would have made it", fmt(d.p.abandoned_cut.context_tokens_after)) : null,
      d.after < d.before && d.p.outcome ? figure("messages", describeOutcome(d.p.outcome)) : null,
      d.p.degradations.length ? h("div", { class: "tip-row" }, h("span", { class: "st warn" }, "⚠ "), plural(d.p.degradations.length, "move"), " didn't happen") : null,
    ]);
    const cx = box.left + x(i) * scaleX;
    placeTooltip({ left: cx + 14, right: cx + 14, top: box.top + M.t, bottom: box.top + M.t + 40 });
  });
  hit.addEventListener("pointerleave", () => { cross.style.display = "none"; hideTooltip(); });
  root.appendChild(hit);
  setChildren(host, root);
}

export function renderChecks() {
  const passes = state.activity ? state.activity.recent_passes.slice().reverse() : [];
  // Consecutive checks that found nothing to do are one row: they are the normal case.
  const groups = [];
  for (const p of passes) {
    const quiet = p.gc_status !== "completed" && p.gc_status !== "bypassed_on_error" && !p.abandoned_cut && !p.degradations.length;
    const lastGroup = groups.at(-1);
    if (quiet && lastGroup && lastGroup.quiet && lastGroup.status === p.gc_status) lastGroup.items.push(p);
    else groups.push({ quiet, status: p.gc_status, items: [p] });
  }
  const shown = ui.allChecks ? groups : groups.slice(0, CHECKS_SHOWN);
  setChildren($("checks"),
    h("thead", {}, h("tr", {}, ["When", "Result", "Prompt, tokens", "Took"].map((title) => h("th", {}, title)))),
    h("tbody", {}, groups.length ? shown.map((g) => {
      const p = g.items[0], oldest = g.items.at(-1);
      const cut = p.gc_status === "completed" && p.tokens_saved > 0;
      const durations = g.items.map((x) => x.duration_ms);
      const span = (a, b) => a === b ? a : `${a} – ${b}`;
      const took = span(`${Math.min(...durations).toFixed(0)}`, `${Math.max(...durations).toFixed(0)}`);
      return h("tr", { class: cut ? "cut" : p.gc_status === "completed" ? "" : "quiet" },
        h("td", { class: "num", style: "white-space:nowrap" }, span(agoTs(oldest.gc_completed_at), agoTs(p.gc_completed_at))),
        h("td", {}, statusChip(p.gc_status), g.items.length > 1 ? h("span", { class: "times" }, `×${g.items.length}`) : null,
          cut && p.outcome ? h("div", { class: "muted", style: "font-weight:400;font-size:12.5px" }, describeOutcome(p.outcome)) : null,
          p.degradations.length ? h("div", { style: "font-weight:400" }, h("span", { class: "st warn" }, "⚠ "), plural(p.degradations.length, "move"), " didn't happen") : null),
        h("td", { class: "num", style: "white-space:nowrap" }, cut ? `${fmt(p.tokens_before)} → ${fmt(p.tokens_before - p.tokens_saved)}`
          : p.abandoned_cut ? [fmt(p.tokens_before), h("div", { class: "muted", style: "font-size:12px" }, `would have been ${fmt(p.abandoned_cut.context_tokens_after)}`)]
          : span(fmt(oldest.tokens_before), fmt(p.tokens_before))),
        h("td", { class: "num", style: "white-space:nowrap" }, `${took} ms`));
    }) : h("tr", {}, h("td", { colspan: 4, class: "empty" }, "No checks yet."))),
    groups.length > CHECKS_SHOWN ? h("tfoot", {}, h("tr", {}, h("td", { colspan: 4 },
      h("button", { class: "link-btn", type: "button", onclick: () => { ui.allChecks = !ui.allChecks; renderChecks(); } },
        ui.allChecks ? "Show fewer" : `Show all ${plural(passes.length, "check")}`)))) : null);
}

// ----------------------------------------------------------------- settings

export function renderSettings() {
  const c = state.config;
  if (!c) return;
  // The flag is how to change a setting while watching here; the rest are GCConfig fields.
  const rows = [
    ["Collection size", "min_cut_tokens", c.min_cut_tokens, "tokens of messages to compact or archive before a collection runs", "--min-cut-tokens"],
    ["Always word for word", "last_n_turns_to_keep", c.last_n_turns_to_keep, "newest messages never collected", "--keep-last"],
    ["Keep from", "keep_threshold", c.keep_threshold, "relevance at or above this is kept, quoted whole"],
    ["Archive below", "archive_threshold", c.archive_threshold, "relevance below this is archived, reduced to facts"],
    ["Too short to compact", "min_compactable_tokens", c.min_compactable_tokens, "messages this short, in tokens, are kept whole"],
    ["Old generation budget", "max_old_gen_tokens", c.max_old_gen_tokens, "most tokens of quotes and summaries in the recap; above it, the coldest are archived"],
    ["Fact budget", "max_memory_tokens", c.max_memory_tokens, "most tokens of facts in the recap"],
    ["The model's window", "context_window", c.context_window, "its capacity, in tokens", "--context-window"],
  ];
  fill("settings", rows.map(([label, name, value, desc, flag]) => h("div", { class: "setting" },
    h("div", { class: "name" }, label, h("code", {}, name)), h("div", { class: "value" }, fmt(value)),
    h("div", { class: "desc" }, desc, flag ? h("div", {}, "change with ", h("code", { class: "flag" }, `run_proxy.py ${flag}`)) : null))));
}

// --------------------------------------------------------------- the three generations

// Young, old, permanent: where each message lives, from the session's own state.
export function renderGenerations(k) {
  const young = k.turns.filter((t) => t.fidelity === "verbatim");
  const quotes = k.front_entries.filter((e) => e.kind === "kept");
  const summaries = k.front_entries.filter((e) => e.kind === "compacted");
  const active = k.facts.filter((f) => f.status === "active");
  const sent = active.filter((f) => f.sent).length;
  const replaced = k.facts.length - active.length;
  const sum = (items) => items.reduce((total, item) => total + item.tokens, 0);
  const box = (key, title, gloss, count, detail, which) => h("div", { class: `gen-box gen-${key}` },
    h("div", { class: "gen-title" }, title), h("div", { class: "gen-gloss" }, gloss),
    h("div", { class: "gen-count" }, count), h("div", { class: "gen-detail" }, detail),
    h("div", { class: "gen-links" }, which));
  const arrow = (verb, how) => h("div", { class: "gen-arrow", "aria-hidden": "true" },
    h("span", { class: "gen-verb" }, verb), h("span", { class: "gen-line" }), h("span", { class: "gen-how" }, how));
  // Which messages each generation holds, each a link to the message.
  const links = (label, numbers) => numbers.length ? h("div", {}, `${label}: `, messageLinks(numbers)[1]) : null;
  const covered = (items, turnsOf) => items.flatMap(turnsOf);
  fill("generations", h("div", { class: "gen-flow" },
    box("young", "Young generation", "sent word for word", plural(young.length, "message"),
      `${plural(sum(young), "token")} · the newest ${keepLast()} always stay`,
      links("messages", young.map((t) => t.turn_index))),
    arrow(`collection${k.cuts ? ` ×${fmt(k.cuts)}` : ""}`, "keep → quote · compact → summary"),
    box("old", "Old generation", "in the recap: kept or compacted", plural(k.front_entries.length, "entry", "entries"),
      `${plural(quotes.length, "quote")} · ${plural(summaries.length, "summary", "summaries")} · ${fmt(sum(k.front_entries))} of ${fmt(state.config ? state.config.max_old_gen_tokens : 0)} tokens`,
      [links("kept", covered(quotes, (e) => e.source_turns)), links("compacted", covered(summaries, (e) => e.source_turns))]),
    arrow("over budget", "archive coldest → facts"),
    box("permanent", "Permanent generation", "archived facts and excerpts, sent in the recap", plural(active.length, "entry", "entries"),
      `${fmt(sent)} sent · ${fmt(active.length - sent)} stored only${replaced ? ` · ${fmt(replaced)} contradicted` : ""}`,
      links("from", covered(active, (f) => f.source_turns)))),
    h("p", { class: "gen-note" }, "Archive inside a collection goes straight from young to permanent. Messages are counted once their reply arrives."));
}
