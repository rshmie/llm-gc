// The engine view: what LLM-GC is doing to this conversation, and why. When the next collection
// runs and why each message gets what it gets, what collections did and couldn't do, every check,
// and last the proxy's totals and settings, which apply to every conversation.

import { BECOMES, CARD_INFO, CATS, CHART_HEIGHT, CHECKS_SHOWN, CREDENTIAL_REFUSED_TEXT, CREDENTIAL_REFUSED_TITLE, MAX_EXCHANGE_LINKS, MOVES_SHOWN, NOT_COLLECTED_TEXT, NOT_MOVED, OUTCOME_LINES, PAUSED_TEXT, STATUS, SWITCH_REASON, TOOL_ROUND_TEXT, describeOutcome, swatch } from "./vocabulary.js";
import { $, agoTs, compact, countEl, fill, fmt, h, meterBar, plural, reducedMotion, setChildren, storeGet, storeSet, svg } from "./dom.js";
import { credentialRefusal, keepLast, preview, state, ui } from "./state.js";
import { explain, hideTooltip, hint, info, placeTooltip, showTooltip, tipQuote, tooltipContent, hoverTip } from "./tooltip.js";
import { jumpTo, messagesLabel, trace } from "./conversation.js";
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
    out.push(out.length ? ", " : "", rangeLink(sorted[i], sorted[j]));
    i = j;
  }
  return [sorted.length === 1 ? "message " : "messages ", out];
}

const rangeLink = (first, last) =>
  h("button", { class: "link-btn", type: "button", onclick: () => jumpTo(first) }, last > first ? `${first}–${last}` : String(first));

// One link per exchange, never merged: neighbouring exchanges at one level are still separate
// exchanges, and a link can only open the first message of the range it names.
function exchangeLinks(exchanges) {
  const shown = exchanges.slice(0, MAX_EXCHANGE_LINKS);
  const out = shown.flatMap((turns, i) => [i ? ", " : "", rangeLink(turns[0], turns[turns.length - 1])]);
  if (exchanges.length > shown.length) out.push(`, and ${fmt(exchanges.length - shown.length)} more`);
  return out;
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
  for (const button of document.querySelectorAll("[data-info]")) explain(button, `card-${button.dataset.info}`, CARD_INFO[button.dataset.info]);
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
  const u = p.provider_usage;
  const s = p.summaries;
  fill("proxy-line", h("span", { class: "label" }, "All conversations"),
    h("span", {}, h("b", {}, fmt(p.requests)), " requests"), h("span", { class: "sep" }, "·"),
    h("span", {}, h("b", {}, fmt(p.tokens_forwarded)), ` tokens forwarded since the proxy started${p.ratio !== null ? `, ${Math.round(p.ratio * 100)}% of what your clients sent${p.gc_applied ? "" : " - nothing collected yet"}` : ""}`), h("span", { class: "sep" }, "·"),
    p.gc_bypassed ? h("span", {}, h("span", { class: "st warn" }, "⚠ "), `${fmt(p.gc_bypassed)} passed on untouched after an error`) : h("span", {}, "none passed on after an error"),
    h("button", { class: "link-btn", type: "button", onclick: () => $("proxy-details").scrollIntoView({ behavior: smooth(), block: "start" }) }, "Details ↓"));
  const tile = (label, term, value, caption, extra) => h("section", { class: "card tile" },
    h("div", { class: "label" }, label, term ? info(term) : null), value === null ? null : h("div", { class: "value" }, value), extra || null,
    caption ? h("div", { class: "caption" }, caption) : null);
  fill("tiles",
    tile("Requests through the proxy", null, countEl("tile-requests", p.requests),
      [`${fmt(p.gc_applied)} handled by LLM-GC`, p.gc_bypassed ? [" · ", h("span", { class: "st warn" }, "⚠ "), `${fmt(p.gc_bypassed)} passed on untouched after an error`] : " · none passed on after an error",
        p.rewrites_refused ? [" · ", h("span", { class: "st warn" }, "⚠ "), `${fmt(p.rewrites_refused)} refused as LLM-GC rewrote ${p.rewrites_refused === 1 ? "it" : "them"}, then sent as written`] : null]),
    tile("Total sent since the proxy started", "forwarded", countEl("tile-forwarded", p.tokens_forwarded),
      `${p.ratio !== null ? `${Math.round(p.ratio * 100)}% of` : "of"} the ${fmt(p.tokens_from_client)} your clients sent, whole requests, all conversations`,
      p.ratio !== null ? h("div", { class: "meter", "aria-hidden": "true" }, meterBar("tile-forwarded-ratio", p.ratio * 100)) : null),
    tile("Conversation tracking", "tracking", null, null,
      h("div", { class: "triple" }, [[p.sessions_seeded, "started fresh"], [p.history_rewinds, "rewound after an edit"], [p.history_restarts, "restarted"]]
        .map(([n, label]) => h("div", {}, h("b", {}, fmt(n)), h("span", {}, label))))),
    tile("What the provider reported", "billed", null,
      u.replies ? [`On ${plural(u.replies, "reply", "replies")}, by the provider's own count: what it bills.`,
        p.collection_cache_write_tokens ? ` ${compact(p.collection_cache_write_tokens)} of the cache writes were on requests that carried a collection: the cost to set a saving against.` : ""]
        : "No reply has reported its usage yet.",
      u.replies ? h("div", { class: "triple" }, [[u.cache_read_input_tokens, "cache read"], [u.input_tokens + u.cache_creation_input_tokens, "input and cache write"], [u.output_tokens, "output"]]
        .map(([n, label]) => h("div", {}, h("b", {}, compact(n)), h("span", {}, label)))) : null),
    tile("LLM-GC's own model calls", null, s.calls ? countEl("tile-summaries", s.calls) : null,
      s.calls ? `To summarise and read facts: ${fmt(s.input_tokens)} tokens in, ${fmt(s.output_tokens)} out, by the provider's count. Not part of any conversation's request, and not taken from its saving.`
        : "None yet: nothing has been summarised."));
}

// One line answers "is LLM-GC healthy?": the last check, and any move a collection couldn't make.
export function renderEngineStatus() {
  const a = state.activity;
  const last = a && a.gc_breakdown;
  const problems = a ? a.recent_degradations.length : 0;
  $("view-engine").classList.toggle("has-problems", problems > 0);
  const failed = !!last && last.gc_status === "bypassed_on_error";
  const refusal = credentialRefusal();
  const tone = failed || refusal ? "critical" : problems || state.pause ? "warn" : "good";
  const lastText = state.pause ? `Its own model calls are paused until ${new Date(state.pause.until).toLocaleTimeString()}: ${state.pause.reason}. Your requests go through as normal.`
    : state.knowledge && !state.knowledge.collected ? "This conversation is a subagent's: recorded and sent as written, never checked for a collection."
    : last ? `Last check ${agoTs(last.gc_completed_at)}: ${STATUS[last.gc_status]?.text || last.gc_status}, ${last.duration_ms.toFixed(0)} ms.`
    : "No checks yet. LLM-GC checks after every reply.";
  fill("eng-status", h("div", { class: `status-line tone-${tone}` },
    h("span", { class: "badge", "aria-hidden": "true" }, failed || refusal ? "✕" : problems ? "!" : state.pause ? "⏸" : "✓"),
    h("span", { class: "msg-text", ...(state.pause ? hoverTip("Paused", PAUSED_TEXT) : {}) }, h("b", {}, refusal ? CREDENTIAL_REFUSED_TITLE : failed ? "The last check failed, and nothing changed." : problems ? `${plural(problems, "part")} of a collection fell short.` : state.pause ? "LLM-GC is yielding to a rate limit." : "LLM-GC is working normally."),
      refusal ? ` ${CREDENTIAL_REFUSED_TEXT} The upstream said: ${refusal.degradation.detail}` : ` ${lastText}`,
      !refusal && failed && last.failure_reason ? ` ${last.failure_stage}: ${last.failure_reason}` : null),
    problems ? h("button", { class: "link-btn", type: "button", onclick: () => openCard("notmoved", true) }, "See why →")
      : h("span", { class: "then" }, "Every collection did all it planned.")));
}

// ----------------------------------------------------------------- what happens next, and why

// The newest pass that planned a cut, whether it was made or refused.
function lastAttempt() {
  const passes = state.activity ? state.activity.recent_passes : [];
  return passes.findLast((p) => p.abandoned_cut || (p.gc_status === "completed" && p.outcome)) || null;
}

// A collection has two halves, so the card shows the one the conversation is in. Before a cut is
// prepared: the two conditions for preparing one. Once one is prepared: the three ways it is
// switched in, each with its state. Beside either, what the collection would do to which messages.
export function renderNext(k) {
  // A conversation that is never collected has no next collection to show the conditions of.
  if (!k.collected) {
    fill("panel-next", h("p", { class: "next-lede" }, "Not collected: a subagent's conversation is recorded and sent as written."),
      h("p", { class: "secondary" }, NOT_COLLECTED_TEXT));
    return;
  }
  if (k.staged) renderStaged(k);
  else renderPreparing(k);
}

const checkItem = (tone, icon, title, figure, ...body) => h("li", { class: `ck tone-${tone}` },
  h("span", { class: "ck-badge", "aria-hidden": "true" }, icon),
  h("div", {}, h("div", { class: "ck-head" }, h("b", {}, title), figure), body));

const meterOf = (key, value, max, label) => h("div", { class: `meter${value >= max ? " ready" : ""}`, role: "meter",
  "aria-valuemin": "0", "aria-valuemax": String(max), "aria-valuenow": String(value), "aria-label": label },
  meterBar(key, max > 0 ? Math.min(1, value / max) * 100 : (value > 0 ? 100 : 0)));

const duration = (seconds) => seconds < 90 ? `${Math.round(seconds)} s`
  : seconds < 5400 ? `${Math.round(seconds / 60)} min` : `${+(seconds / 3600).toFixed(1)} h`;

function renderPreparing(k) {
  const cut = k.next_cut;
  const taking = k.turns.filter((t) => t.fidelity === "verbatim" && t.becomes);
  const refused = lastAttempt()?.abandoned_cut ? lastAttempt() : null;
  const saving = cut.estimated_saving;
  const lede = state.pause ? `Paused until ${new Date(state.pause.until).toLocaleTimeString()}: LLM-GC is yielding to the upstream's rate limit, so no cut is prepared.`
    : !taking.length ? "Nothing to collect yet."
    : !cut.batch_ready ? `Waiting for ${fmt(cut.min_cut_tokens - cut.compactable_tokens)} more tokens of messages to summarise or archive.`
    : saving > 0 ? "Ready: a cut is prepared after the next reply."
    : "Enough to collect, but collecting now would make the prompt bigger, so it waits.";

  fill("panel-next",
    h("p", { class: "next-lede" }, lede),
    h("div", { class: "next-grid" },
      h("div", {},
        h("div", { class: "label" }, "A cut is prepared when both hold"),
        h("ul", { class: "ck-list" },
          checkItem(cut.batch_ready ? "good" : "wait", cut.batch_ready ? "✓" : "…", "Enough to collect",
            h("span", { class: "num" }, countEl("next-ready", cut.compactable_tokens), ` tokens · ${fmt(cut.min_cut_tokens)} needed`),
            meterOf("next-ready", cut.compactable_tokens, cut.min_cut_tokens, "Tokens of messages to summarise or archive"),
            h("p", {}, "Only messages to be summarised or archived count. Quoting a message word for word saves nothing.")),
          !taking.length
            ? checkItem("wait", "…", "Makes the prompt smaller", null, h("p", {}, "Nothing to estimate: no older messages are waiting."))
            : saving > 0
              ? checkItem("good", "✓", "Makes the prompt smaller", h("span", { class: "num" }, `about ${fmt(saving)} tokens smaller`),
                h("p", {}, "Estimated before any summary is paid for, and checked again on the real ones."),
                refused ? h("p", {}, `The last try, ${agoTs(refused.gc_completed_at)}, was skipped on its real summaries: ${fmt(refused.tokens_before)} → ${fmt(refused.abandoned_cut.context_tokens_after)} tokens.`) : null)
              : checkItem("warn", "!", "Makes the prompt smaller", h("span", { class: "num" }, `about ${fmt(-saving)} tokens bigger`),
                h("p", {}, "The recap's markers and opening lines would cost more than these short messages save, so it waits for more.")))),
      wouldDo(k, taking, "What it would do now", [`Counts toward the ${fmt(k.next_cut.min_cut_tokens)}`, "Doesn't count"])));
}

function renderStaged(k) {
  const s = k.staged;
  const cache = k.cache;
  const needed = Math.ceil(s.min_saving_share * s.rewritten_tokens);
  const repays = s.saving_tokens >= needed;
  const overdue = s.stale_tokens >= s.stale_bound_tokens;
  // With no proxy reading the cache beside this dashboard, the verdict was made for a cold one.
  const cold = cache ? cache.state === "cold" : true;
  const lede = s.switches
    ? `Prepared: switched in with the next request, because ${SWITCH_REASON[s.reason] || s.reason}.`
    : s.reason === "tool_round_open"
    ? `Prepared, waiting for the tool round to end. ${TOOL_ROUND_TEXT}`
    : `Prepared, waiting for the right request: switching now would make the provider cache ${fmt(s.rewritten_tokens)} tokens again to save ${fmt(s.saving_tokens)}. Meanwhile the model gets these messages word for word.`;
  const cacheSays = !cache ? "No proxy reads the cache beside this dashboard, so it counts as cold."
    : cache.uses_cache === false ? "This conversation's replies use no prompt cache: every request is read in full, so a smaller prompt is always cheaper."
    : cache.uses_cache === null ? "No request has been answered yet, so nothing says the cache is cold. Until something does, it counts as warm."
    : cold ? `Idle ${duration(cache.idle_s)}, longer than the ${duration(cache.lifetime_s)} the provider keeps this conversation cached.`
    : `${duration(cache.idle_s)} since the last request; the provider keeps this conversation cached for ${duration(cache.lifetime_s)} after each one.`;
  const taking = k.turns.filter((t) => t.fidelity === "verbatim" && t.becomes);
  const prepared = new Set(s.turns);
  const unprepared = taking.filter((t) => !prepared.has(t.turn_index)).length;

  fill("panel-next",
    h("p", { class: "next-lede" }, lede),
    h("div", { class: "next-grid" },
      h("div", {},
        h("div", { class: "label" }, "Switched in when any one holds"),
        h("ul", { class: "ck-list" },
          checkItem(cold ? "good" : "wait", cold ? "✓" : "…", "The cache is cold", null, h("p", {}, cacheSays)),
          checkItem(repays ? "good" : "wait", repays ? "✓" : "…", "It repays its rewrite",
            h("span", { class: "num" }, countEl("next-repays", s.saving_tokens), ` tokens saved · ${fmt(needed)} needed`),
            meterOf("next-repays", s.saving_tokens, needed, "Tokens the switch saves, against the tokens it must save"),
            h("p", {}, `On a warm cache it must save ${fmt(s.min_saving_share)}× the ${fmt(s.rewritten_tokens)} tokens the provider would cache again.`)),
          checkItem(overdue ? "good" : "wait", overdue ? "✓" : "…", "Stale weight at its bound",
            h("span", { class: "num" }, countEl("next-stale", s.stale_tokens), ` of ${fmt(s.stale_bound_tokens)} tokens, by LLM-GC's count`),
            meterOf("next-stale", s.stale_tokens, s.stale_bound_tokens, "Stale weight, against its bound"),
            h("p", {}, "Gone-quiet material the model still gets word for word. At the ",
              boundFrom(k) ? hint("bound", "Where the bound comes from", boundFrom(k)) : "bound", ", it is switched in whatever the cache costs.")))),
      wouldDo(k, taking, "What switching it in would do", ["Condensed", "Sent as they are"],
        unprepared ? h("p", { class: "note" }, `${plural(unprepared, "message")} below ${unprepared === 1 ? "is" : "are"} not prepared yet: the next pass condenses ${unprepared === 1 ? "it" : "them"} first.`) : null)));
}

// Where the stale bound comes from: a share of the model's window, which is in the model's tokens, converted by the
// engine to LLM-GC's count at the rate the conversation's replies measure, so the bound and the stale weight are
// counted alike. Shown as the engine reports it, never recomputed here.
function boundFrom(k) {
  const c = state.config;
  if (!c) return null;
  const share = `${Math.round(c.max_stale_ratio * 100)}% of the model's ${fmt(k.context_window)}-token window`;
  return k.token_rate
    ? `The bound is ${share}: ${fmt(k.window_in_own_count)} in LLM-GC's count, since the model counts ${k.token_rate.toFixed(2)} tokens for each of LLM-GC's.`
    : `The bound is ${share}, taken as LLM-GC's count until a reply measures how the model's count runs against it.`;
}

// What the collection would do: the messages it would condense, by what each becomes, then the rest.
// `heads` names the two lists: while a cut is not prepared, which messages count toward the batch.
function wouldDo(k, taking, title, heads, note = null) {
  const moving = k.spans.filter((s) => s.becomes);
  const tokensOf = (turns) => turns.reduce((sum, t) => sum + t.tokens, 0);
  const newest = k.turns.filter((t) => t.fidelity === "verbatim" && !t.becomes).length;
  const row = (key, label, rowNote) => {
    const turns = taking.filter((t) => t.becomes === key);
    return turns.length ? h("li", {}, swatch(key),
      h("span", {}, h("b", {}, plural(turns.length, "message")), ` ${label}`, rowNote ? h("span", { class: "would-note" }, ` (${rowNote})`) : null),
      h("span", { class: "num" }, `${fmt(tokensOf(turns))} tokens`)) : null;
  };
  const counted = [row("compacted", "summarised"), row("archived", "archived"), row("out_of_view", "folded out of view")].filter(Boolean);
  const notCounted = [row("kept", "quoted word for word", "short, or it matters now"),
    newest ? h("li", {}, swatch("verbatim"), h("span", {}, h("b", {}, `The newest ${plural(newest, "message")}`), " stay recent, sent as written"), h("span")) : null,
    moving.length ? h("li", {}, h("span", { class: "qdot" }), h("span", {}, h("b", {}, plural(moving.length, "earlier exchange")), " shown at another level"), h("span")) : null].filter(Boolean);
  return h("div", {},
    h("div", { class: "label" }, title),
    note,
    counted.length ? [h("div", { class: "would-head" }, heads[0]), h("ul", { class: "would" }, counted)] : null,
    notCounted.length ? [h("div", { class: "would-head" }, heads[1]), h("ul", { class: "would" }, notCounted)] : null,
    !counted.length && !notCounted.length ? h("p", { class: "note" }, "Nothing yet.") : null,
    taking.length || moving.length ? h("button", { class: "link-btn", type: "button", onclick: () => openCard("rate", true) }, "Why each exchange gets this ↓") : null);
}

const RANK = { out_of_view: 0, archived: 1, compacted: 2, kept: 3 };

// Why an exchange is shown at its level, in the planner's own steps: the band its score earns,
// the margin that holds it near a line, a level no cheaper than a richer one, and the budget.
// `item` is a span, or a message the next collection would take (scored as its exchange).
export function whyText(item) {
  const c = state.config;
  const low = c ? c.archive_threshold : 0.3, high = c ? c.keep_threshold : 0.7, margin = c ? c.level_margin : 0.05;
  if (item.score === null || item.score === undefined) return `Not scored: among the newest ${keepLast()} messages, which always stay recent, sent as written.`;
  const s = item.score.toFixed(2);
  const band = item.score >= high ? "kept" : item.score >= low ? "compacted" : "archived";
  const earned = `Score ${s}: ${band === "kept" ? `at or above ${high.toFixed(2)}, so kept` : band === "compacted"
    ? `between ${low.toFixed(2)} and ${high.toFixed(2)}, so summarised` : `below ${low.toFixed(2)}, so archived`}.`;
  const to = item.becomes ?? item.level;
  if (!to || to === band) return earned;
  if (to === "out_of_view") return `${earned} Folded out of view instead: the recap is over its budget with every exchange at its least detail, so the oldest are folded.`;
  const lineOf = { kept: high, compacted: low };
  if (RANK[to] < RANK[band]) {
    const held = item.level === to && item.score < lineOf[band] + margin;
    return held ? `${earned} It stays ${to} until its score reaches ${(lineOf[band] + margin).toFixed(2)}: the margin keeps an exchange near a line from flipping.`
      : `${earned} Shown ${to} instead: the recap is over its budget, and the coldest exchanges lose detail first.`;
  }
  const held = item.level === to && item.score >= lineOf[to] - margin;
  if (held) return `${earned} It stays ${to} until its score falls below ${(lineOf[to] - margin).toFixed(2)}: the margin keeps an exchange near a line from flipping.`;
  return `${earned} ${to === "kept" ? "Kept" : "Compacted"} instead: ${band === "compacted" && to === "kept"
    ? "it has no summary (too short to summarise, or the summary failed), so its words are quoted"
    : "the cheaper level would be no smaller than this one"}.`;
}

// The scores lead with what the next collection takes, then every exchange already collected:
// the ones it would move first, the rest folded under one line.
export function renderRelevance(k) {
  const c = state.config;
  const low = c ? c.archive_threshold : 0.3, high = c ? c.keep_threshold : 0.7;
  const plot = $("relevance");
  plot.style.setProperty("--a", `${low * 100}%`);
  plot.style.setProperty("--k", `${high * 100}%`);
  const byIndex = new Map(k.turns.map((t) => [t.turn_index, t]));
  // A message the next collection takes carries its exchange's score and level, so its first stands for it.
  const drafts = k.next_cut.spans.map((turns) => ({ ...byIndex.get(turns[0]), turns, level: null }));
  const collected = [...k.spans].reverse();
  if (!drafts.length && !collected.length) { setChildren(plot, h("p", { class: "empty" }, "Nothing to score yet: the conversation is within its newest messages.")); return; }
  const moving = collected.filter((s) => s.becomes), still = collected.filter((s) => !s.becomes);
  const stillOpen = ui.stayOpen || (!drafts.length && !moving.length);
  const row = (item) => {
    const to = item.becomes ?? item.level;
    const opening = byIndex.get(item.turns[0])?.text;
    return h("div", { class: "rate-row", id: `r-${item.turns[0]}`, "data-turns": item.turns.join(","),
      onpointerenter: () => trace(item.turns), onpointerleave: () => trace(null) },
      h("div", { class: "rate-lab" }, `#${item.turns.length > 1 ? `${item.turns[0]}–${item.turns.at(-1)}` : item.turns[0]}`),
      h("div", { class: "rate-track" }, h("span", { class: `rate-dot f-${to}`, style: `left:${item.score * 100}%`,
        ...hoverTip(`Exchange of ${messagesLabel(item.turns)}`, whyText(item), tipQuote(preview(opening, 120))) })),
      h("div", { class: "rate-why" }, whyText(item),
        item.becomes ? h("div", { class: "then" }, `→ next collection: ${BECOMES[item.becomes]}`)
          : item.level ? h("div", { class: "then muted" }, `now ${CATS[item.level].tag}`) : null));
  };
  const head = (label, n, open, toggles) => toggles
    ? h("button", { class: "rate-group", type: "button", "aria-expanded": String(open), onclick: () => { ui.stayOpen = !ui.stayOpen; renderRelevance(state.knowledge); } },
      chevron(), label, h("span", { class: "n" }, `· ${n}`))
    : h("div", { class: "rate-group" }, label, h("span", { class: "n" }, `· ${n}`));
  const unscored = k.turns.filter((t) => t.fidelity === "verbatim" && t.score === null).length;
  setChildren(plot,
    h("div", { class: "rate-list" },
      h("div", { class: "rate-row rate-axis", "aria-hidden": "true" }, h("div"),
        h("div", { class: "rate-track" },
          h("span", { class: "zone", style: `left:${(low / 2) * 100}%` }, `archived · below ${low}`),
          h("span", { class: "zone", style: `left:${((low + high) / 2) * 100}%` }, "compacted"),
          h("span", { class: "zone", style: `left:${((high + 1) / 2) * 100}%` }, `kept · ${high}+`)), h("div")),
      drafts.length ? [head("In the next collection", drafts.length, true, false), drafts.map(row)] : null,
      moving.length ? [head("Collected, moving at the next collection", moving.length, true, false), moving.map(row)] : null,
      still.length ? [head("Collected, staying where they are", still.length, stillOpen, !!(drafts.length || moving.length)), stillOpen ? still.map(row) : null] : null,
      unscored ? h("div", { class: "rate-gone" }, `The newest ${plural(unscored, "message")} stay recent, sent as written, and are scored once a collection takes them.`) : null),
    h("div", { class: "rate-legend" },
      h("span", {}, h("span", { class: "rate-dot f-archived" }), "archived"),
      h("span", {}, h("span", { class: "rate-dot f-compacted" }), "compacted"),
      h("span", {}, h("span", { class: "rate-dot f-kept" }), "kept"),
      k.spans.some((s) => s.level === "out_of_view") ? h("span", {}, h("span", { class: "rate-dot f-out_of_view" }), "out of view") : null));
}

// A "next:" tag in the conversation lands here, on its exchange's score.
export function showWhy(turnIndex) {
  showTab("engine");
  openCard("rate");
  const find = () => [...document.querySelectorAll("#relevance [data-turns]")].find((row) => row.dataset.turns.split(",").map(Number).includes(turnIndex));
  if (!find()) { ui.stayOpen = true; renderRelevance(state.knowledge); }
  const row = find();
  if (!row) return;
  row.scrollIntoView({ behavior: smooth(), block: "center" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1600);
}

// ----------------------------------------------------------------- what has happened

// Every collection in the recent checks, newest first, from its own record of what it moved. A
// replaced fact is listed with the collection that took the message replacing it.
export function renderHistory() {
  const facts = state.knowledge ? state.knowledge.facts : [];
  const collections = (state.activity ? state.activity.recent_passes : [])
    .filter((p) => p.gc_status === "completed" && p.outcome).reverse();
  const shown = ui.allMoves ? collections : collections.slice(0, MOVES_SHOWN);
  const link = (turn) => h("button", { class: "link-btn", type: "button", onclick: () => jumpTo(turn) }, `#${turn}`);
  fill("history", collections.length ? [
    h("ul", { class: "events" }, shown.map((p) => {
      const o = p.outcome;
      const taken = new Set([...o.kept, ...o.compacted, ...o.archived, ...o.folded]);
      const replaced = facts.filter((f) => f.status === "superseded" && taken.has(f.superseded_by_turn));
      return h("li", {},
        h("div", { class: "event-head" }, h("b", {}, "Collection"),
          h("span", { class: "when" }, `${agoTs(p.gc_completed_at)} · ${fmt(p.tokens_before)} → ${fmt(p.tokens_before - p.tokens_saved)} tokens`)),
        h("div", { class: "event-lines" },
          OUTCOME_LINES.filter(([key]) => o[key].length).map(([key, label, colour]) => h("div", {},
            h("span", { class: "k" }, colour ? swatch(colour) : null, label), h("span", {}, messageLinks(o[key])))),
          replaced.length ? h("div", {}, h("span", { class: "k" }, "↺ Fact replaced"),
            h("span", {}, replaced.map((f, n) => [n ? ", " : "", `“${f.topic}” `, link(f.turn), " → ", link(f.superseded_by_turn)]))) : null));
    })),
    collections.length > MOVES_SHOWN ? h("button", { class: "link-btn show-all", type: "button", onclick: () => { ui.allMoves = !ui.allMoves; renderHistory(); } },
      ui.allMoves ? "Show fewer" : `Show all ${collections.length} collections`) : null,
  ] : h("p", { class: "empty" }, "No collections in the recent checks yet."));
}

export function renderProblems() {
  const problems = state.activity ? state.activity.recent_degradations.slice().reverse() : [];
  const shown = ui.allProblems ? problems : problems.slice(0, MOVES_SHOWN);
  fill("problems", problems.length ? [
    h("ul", { class: "events" }, shown.map((r) => {
      const [label, instead] = NOT_MOVED[r.degradation.reason] || [r.degradation.reason.replaceAll("_", " "), null];
      return h("li", {},
        h("div", { class: "event-head" }, h("span", { class: "st warn" }, "⚠"), h("b", {}, label), h("span", { class: "when" }, agoTs(r.occurred_at))),
        h("div", { class: "event-lines" },
          h("div", {}, h("span", { class: "k" }, "Exchange"), h("span", {}, messageLinks(r.degradation.turn_indices))),
          instead ? h("div", {}, h("span", { class: "k" }, "Instead"), h("span", {}, instead)) : null,
          r.degradation.detail ? h("div", {}, h("span", { class: "k" }, "Detail"), h("span", {}, r.degradation.detail)) : null));
    })),
    problems.length > MOVES_SHOWN ? h("button", { class: "link-btn show-all", type: "button", onclick: () => { ui.allProblems = !ui.allProblems; renderProblems(); } },
      ui.allProblems ? "Show fewer" : `Show all ${problems.length}`) : null,
  ] : h("p", { class: "ok-line" }, h("span", { class: "st good" }, "✓ "), "None so far."));
}

// The conversation as written up to a turn: a running sum of each message's own size.
export function writtenUpTo() {
  if (!state.knowledge) return null;
  const turns = state.knowledge.turns;
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
  // A collection is a completed pass, whatever it did to the size: one that raised exchanges the
  // conversation came back to can leave the prompt larger.
  const cutsDone = passes.filter((p) => p.gc_status === "completed");
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
  const cuts = pts.filter((d) => d.p.gc_status === "completed");
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
  pts.forEach((d, n) => { if (n) sentPath += `H${x(d.i)}V${y(d.before)}`; if (d.after !== d.before) sentPath += `V${y(d.after)}`; });
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
    pts.length <= 40 ? pts.filter((d) => d.p.gc_status !== "completed" && !d.p.abandoned_cut && d.p.gc_status !== "bypassed_on_error")
      .map((d) => svg("circle", { class: "mk-check", cx: x(d.i), cy: y(d.before), r: 3.5 })) : null,
    cuts.map((d) => [svg("circle", { class: "mk-cut", cx: x(d.i), cy: y(d.after), r: 5 }),
      cuts.length <= 8 ? svg("text", { class: "cut-label", x: x(d.i) - 8, y: y(d.after) + 16, "text-anchor": "end" },
        d.after <= d.before ? `−${fmt(d.before - d.after)}` : `+${fmt(d.after - d.before)}`) : null]),
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
      figure("sent to the model", d.p.gc_status === "completed" ? `${fmt(d.before)} → ${fmt(d.after)}` : fmt(d.before)),
      d.written !== null ? figure("as written", fmt(d.written)) : null,
      d.p.abandoned_cut ? figure("a collection would have made it", fmt(d.p.abandoned_cut.context_tokens_after)) : null,
      d.p.gc_status === "completed" && d.p.outcome ? figure("messages", describeOutcome(d.p.outcome)) : null,
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
      const cut = p.gc_status === "completed";
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
    ["Collection size", "min_cut_tokens", c.min_cut_tokens, "tokens of messages to summarise or archive before a cut is prepared", "--min-cut-tokens"],
    ["Saving share", "min_saving_share", c.min_saving_share, "with the provider's cache warm, how many times what a switch makes it cache again the cut must save before it is switched in"],
    ["Stale bound", "max_stale_ratio", c.max_stale_ratio, "share of the window the model may carry that a prepared cut would condense; at it, the cut is switched in whatever the cache costs. The window is in the model's tokens, so it is converted to LLM-GC's count first, at the rate the conversation's replies measure"],
    ["Always recent", "last_n_turns_to_keep", c.last_n_turns_to_keep, "newest messages never collected: each sent as its own message, as written", "--keep-last"],
    ["Keep from", "keep_threshold", c.keep_threshold, "an exchange scoring at or above this is kept, quoted word for word"],
    ["Archive below", "archive_threshold", c.archive_threshold, "an exchange scoring below this is archived, to its opening line and facts"],
    ["Level margin", "level_margin", c.level_margin, "how far past a line a score must move before a span changes level"],
    ["Recap budget", "max_recap_tokens", c.max_recap_tokens, "most tokens the recap spends on older exchanges, at the configured window; above it, the coldest lose detail first. A conversation whose model's window the provider reports gets the same share of that window, converted to LLM-GC's count at the rate its replies measure"],
    ["Configured window", "context_window", c.context_window, "tokens assumed for a model whose window the provider has not reported", "--context-window"],
  ];
  fill("settings", rows.map(([label, name, value, desc, flag]) => h("div", { class: "setting" },
    h("div", { class: "name" }, label, h("code", {}, name)), h("div", { class: "value" }, fmt(value)),
    h("div", { class: "desc" }, desc, flag ? h("div", {}, "change with ", h("code", { class: "flag" }, `llm-gc start ${flag}`)) : null))));
}

// --------------------------------------------------------------- the three generations

// Young, old, permanent: where each message lives, from the session's own state.
export function renderGenerations(k) {
  const young = k.turns.filter((t) => t.fidelity === "verbatim");
  const at = (level) => k.spans.filter((span) => span.level === level);
  const active = k.facts.filter((f) => f.status === "active");
  const sent = active.filter((f) => f.sent).length;
  const inWords = active.filter((f) => !f.sent && f.span_level === "kept").length;
  const replaced = k.facts.length - active.length;
  const box = (key, title, gloss, count, detail, which) => h("div", { class: `gen-box gen-${key}` },
    h("div", { class: "gen-title" }, title), h("div", { class: "gen-gloss" }, gloss),
    h("div", { class: "gen-count" }, count), h("div", { class: "gen-detail" }, detail),
    h("div", { class: "gen-links" }, which));
  const arrow = (verb, how) => h("div", { class: "gen-arrow", "aria-hidden": "true" },
    h("span", { class: "gen-verb" }, verb), h("span", { class: "gen-line" }), h("span", { class: "gen-how" }, how));
  // Which messages each generation holds, each a link to the message.
  const links = (label, numbers) => numbers.length ? h("div", {}, `${label}: `, messageLinks(numbers)[1]) : null;
  const levels = ["kept", "compacted", "archived", "out_of_view"].filter((level) => at(level).length);
  fill("generations", h("div", { class: "gen-flow" },
    box("young", "Young generation", "recent: each sent as written", plural(young.length, "message"),
      `${plural(young.reduce((n, t) => n + t.tokens, 0), "token")} · the newest ${keepLast()} always stay`,
      links("messages", young.map((t) => t.turn_index))),
    arrow(`collection${k.cuts ? ` ×${fmt(k.cuts)}` : ""}`, "in exchanges, each by its score"),
    box("old", "Old generation", "the recap: every exchange at its level", plural(k.spans.length, "exchange"),
      `${levels.map((level) => `${at(level).length} ${CATS[level].tag}`).join(" · ") || "none yet"} · ${fmt(k.front_tokens)} of its ${fmt(k.recap_budget)}-token budget, by LLM-GC's count`,
      levels.map((level) => h("div", {}, `${CATS[level].tag}: `, exchangeLinks(at(level).map((span) => span.turns))))),
    arrow("facts read once", "as each exchange is made"),
    box("permanent", "Permanent generation", "facts, and what replaced what", plural(active.length, "current fact"),
      [sent ? `${fmt(sent)} on their exchange's line` : null, inWords ? `${fmt(inWords)} in quoted words` : null,
        active.length - sent - inWords ? `${fmt(active.length - sent - inWords)} out of view` : null,
        replaced ? `${fmt(replaced)} replaced` : null].filter(Boolean).join(" · ") || "none yet",
      links("from", [...new Set(active.map((f) => f.turn))]))),
    h("p", { class: "gen-note" }, "An exchange keeps its words at every level, so a later collection can show it in more detail again."));
}
