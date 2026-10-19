// The Requests tab: a Claude Code session turn by turn, oldest first as the conversation reads, each turn's
// requests on a waterfall in time, each subagent under the request whose reply started it. Measured, never
// forecast: every number is a request that happened.

import { SWITCH_REASON, TRACE_KINDS, pauseNote } from "./vocabulary.js";
import { $, compact, fill, fmt, h, plural, share } from "./dom.js";
import { state, ui } from "./state.js";
import { hoverTip, info } from "./tooltip.js";

// Requests that are not part of the conversation fold into one line per turn.
const SIDE_KINDS = new Set(["unclaimed", "replaced"]);
// The newest turns shown before "Show N earlier turns" folds the rest above them, as a chat loads older messages.
const TURNS_SHOWN = 8;
const shortId = (id) => id.slice(0, 8);
const clock = (iso) => new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const seconds = (ms) => ms < 1000 ? "<1 s" : ms < 60_000 ? `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s` : `${Math.round(ms / 60_000)} min`;
const at = (iso) => Date.parse(iso);
let shownKey = null;

// Repeated tool names read as one, counted: "Agent ×2, Bash".
function calls(names) {
  const counts = new Map();
  for (const name of names) counts.set(name, (counts.get(name) || 0) + 1);
  return [...counts].map(([name, n]) => (n > 1 ? `${name} ×${n}` : name)).join(", ");
}

const saved = (t) => t.tokens_saved > 0 ? `saved ${fmt(t.tokens_saved)} (${share(t.tokens_saved, t.tokens_from_client)}%)` : "nothing saved";
// What an answer other than 200 means, in words.
const STATUS_WORDS = { 400: "refused as invalid", 401: "credential refused", 403: "credential refused", 413: "too large", 429: "rate limited",
  500: "upstream error", 502: "upstream unreachable", 503: "upstream unavailable", 529: "overloaded" };

// `force` redraws after a toggle; otherwise only when the trace changed, so a poll keeps hover and focus.
export function renderTrace(actions, force = false) {
  const trace = state.trace;
  const key = trace ? JSON.stringify([trace, state.pause]) : null;
  if (!force && key === shownKey) return;
  shownKey = key;
  $("trace-card").hidden = !trace;
  if (!trace) return;
  const folded = !ui.turnsEarlierOpen && trace.turns.length > TURNS_SHOWN;
  const shown = folded ? trace.turns.slice(-TURNS_SHOWN) : trace.turns;
  const earlier = trace.turns.length > TURNS_SHOWN ? h("button", { class: "link-btn turns-earlier", type: "button", "aria-expanded": String(!folded),
    onclick: () => { ui.turnsEarlierOpen = folded; renderTrace(actions, true); } },
    folded ? `Show ${plural(trace.turns.length - shown.length, "earlier turn")}` : "Show fewer turns") : null;
  fill("trace-card", traceHead(trace), paused(), before(trace, actions), earlier,
    h("ol", { class: "turns" }, shown.map((turn) => turnItem(turn, trace, actions))));
}

function traceHead(trace) {
  const t = trace.totals;
  const spent = trace.summaries;
  const subagents = trace.turns.reduce((n, turn) => n + turn.subagents.length, 0) + trace.subagents_before.length;
  const stat = (label, value, sub, tip = null) => h("div", { class: "tstat", ...(tip ? hoverTip(...tip) : {}) },
    h("div", { class: "k" }, label), h("div", { class: "v" }, value), sub ? h("div", { class: "s" }, sub) : null);
  const worth = ["Tokens are not all worth the same",
    "A cache read costs a fraction of sending a token plainly, and a cache write more than sending it. The tokens LLM-GC removes would mostly have been cache reads, so set what it saved against the cache writes its collections caused."];
  return [
    h("div", { class: "card-head" }, h("div", {},
      h("h2", {}, `Claude Code session ${shortId(trace.client_session)}`),
      h("p", { class: "sub" }, `Since ${clock(trace.started_at)} · ${plural(trace.turns.length, "turn")} · ${plural(t.requests, "request")} · ${plural(subagents, "subagent")} · oldest first`))),
    h("div", { class: "trace-stats" },
      stat("Claude Code sent", fmt(t.tokens_from_client), "tokens, by LLM-GC's count"),
      stat("LLM-GC forwarded", fmt(t.tokens_forwarded), saved(t)),
      stat("Read from the cache", compact(t.cache_read_input_tokens), "by the provider's count", worth),
      stat("Written to the cache", compact(t.cache_creation_input_tokens),
        t.collection_cache_write_tokens ? `${compact(t.collection_cache_write_tokens)} of it carrying a collection` : "by the provider's count", worth),
      stat("Sent plainly · output", `${compact(t.input_tokens)} · ${compact(t.output_tokens)}`, "by the provider's count"),
      stat("LLM-GC's own model calls", spent && spent.calls ? fmt(spent.calls) : "none",
        spent && spent.calls ? `${compact(spent.input_tokens)} tokens in, ${compact(spent.output_tokens)} out` : "for this session")),
    h("p", { class: "trace-worth secondary" }, worth[0], info("billed")),
  ];
}

// While LLM-GC yields to a rate limit, the session's page says so first.
function paused() {
  if (!state.pause) return null;
  const pause = pauseNote(state.pause);
  return h("div", { class: `status-line tone-${pause.tone} trace-paused` }, h("span", { class: "badge", "aria-hidden": "true" }, pause.badge),
    h("span", { class: "msg-text", ...hoverTip(...pause.tip) }, h("b", {}, pause.title), ` ${pause.text}`));
}

function before(trace, actions) {
  const rows = trace.before;
  const agents = trace.subagents_before;
  if (!rows.length && !agents.length) return null;
  const open = ui.turnOpen.get("before") === true;
  return h("div", { class: "trace-before" },
    h("button", { class: "link-btn", type: "button", "aria-expanded": String(open),
      onclick: () => { ui.turnOpen.set("before", !open); renderTrace(actions, true); } },
      `${open ? "▼" : "▶"} Before your first message: ${plural(rows.length, "request")} not part of a conversation`, agents.length ? `, ${plural(agents.length, "subagent")}` : ""),
    open ? waterfall(rows, agents, span(rows.concat(agents.flatMap((a) => a.requests))), actions, "before") : null);
}

function turnItem(turn, trace, actions) {
  const key = `${turn.conversation_id}:${turn.position}`;
  const newest = turn === trace.turns.at(-1);
  // The newest turn starts open; once a turn is opened or closed by hand, it stays as it was left.
  const open = ui.turnOpen.has(key) ? ui.turnOpen.get(key) : newest;
  const t = turn.totals;
  const collections = [...turn.requests, ...turn.subagents.flatMap((a) => a.requests)].reduce((n, r) => n + r.carries_collections, 0);
  const duration = turn.ended_at ? seconds(at(turn.ended_at) - at(turn.started_at)) : "running…";
  return h("li", { class: `turn${open ? " open" : ""}` },
    // Oldest first, so the turn that began a new main conversation sits just below the line that marks it.
    turn.new_conversation ? h("div", { class: "turn-boundary", ...hoverTip(TRACE_KINDS.boundary[0], TRACE_KINDS.boundary[1]) },
      "A new main conversation begins here: after /compact, or an edited message") : null,
    h("button", { class: "turn-head", type: "button", "aria-expanded": String(open),
      onclick: () => { ui.turnOpen.set(key, !open); renderTrace(actions, true); } },
      h("span", { class: "chev", "aria-hidden": "true" }, open ? "▼" : "▶"),
      h("span", { class: "turn-no" }, `Turn ${turn.index}`),
      h("span", { class: "turn-words" }, turn.words ? `“${turn.words}”` : h("i", {}, "its conversation has closed")),
      h("span", { class: "turn-meta" },
        h("span", {}, clock(turn.started_at)), h("span", {}, duration), h("span", {}, plural(t.requests, "request")),
        h("span", {}, `${fmt(t.tokens_from_client)} sent overall`), t.tokens_saved > 0 ? h("span", { class: "good" }, `−${fmt(t.tokens_saved)}`) : null,
        turn.subagents.length ? h("span", { class: "tbadge" }, plural(turn.subagents.length, "subagent")) : null,
        collections ? h("span", { class: "tbadge", ...hoverTip(TRACE_KINDS.collection[0], TRACE_KINDS.collection[1]) },
          `◆ ${plural(collections, "collection")}${t.collection_cache_write_tokens ? ` · ${compact(t.collection_cache_write_tokens)} cache written` : ""}`) : null)),
    open ? h("div", { class: "turn-body" },
      h("p", { class: "turn-open secondary" }, turn.part === "main" ? "In the main conversation. " : "In an earlier main conversation, since replaced. ",
        actions.live(turn.conversation_id)
          ? h("button", { class: "link-btn", type: "button", onclick: () => actions.open(turn.conversation_id, turn.position - 1) }, "Open this message in its conversation →")
          : "Its conversation has closed, so its messages are gone; its counts stay."),
      waterfall(turn.requests, turn.subagents, span(turn.requests.concat(turn.subagents.flatMap((a) => a.requests)), turn), actions, key))
      : null);
}

// The time a list of requests covers: from the first sent to the last back, or now while one runs.
function span(rows, turn) {
  const start = turn ? at(turn.started_at) : Math.min(...rows.map((r) => at(r.forwarded_at)));
  const end = Math.max(start + 1, ...rows.map((r) => (r.ended_at ? at(r.ended_at) : Date.now())));
  return { start, end };
}

function waterfall(rows, subagents, window, actions, key) {
  const side = rows.filter((r) => SIDE_KINDS.has(r.kind));
  const showSide = ui.showSide.has(key);
  const placed = new Set();
  const lines = [];
  for (const row of rows) {
    if (SIDE_KINDS.has(row.kind) && !showSide) continue;
    lines.push(requestLine(row, window, 0));
    for (const agent of subagents.filter((a) => row.spawned.includes(a.agent))) {
      placed.add(agent.agent);
      lines.push(...subagentLines(agent, window, actions));
    }
  }
  for (const agent of subagents.filter((a) => !placed.has(a.agent))) lines.push(...subagentLines(agent, window, actions));
  return h("div", { class: "waterfall", role: "table", "aria-label": "Requests in time" },
    h("div", { class: "wf-row wf-axis", role: "row" }, h("span", { role: "columnheader" }, "Request · tokens sent → tokens the model got"),
      h("span", { class: "wf-scale", role: "columnheader" }, h("span", {}, "0"), h("span", {}, seconds(window.end - window.start)))),
    lines,
    side.length ? h("button", { class: "link-btn wf-side", type: "button", onclick: () => toggle(ui.showSide, key, actions) },
      showSide ? `Hide the ${plural(side.length, "side request")}` : `${plural(side.length, "side request")} not part of the conversation: a title, a suggestion, a check — show`) : null);
}

function requestLine(row, window, depth) {
  const [label, about] = TRACE_KINDS[row.kind];
  const failed = row.status !== null && row.status !== 200;
  // A request refused as LLM-GC rewrote it was sent again as written: LLM-GC's fault, which Claude Code never saw.
  const rewriteRefused = failed && row.status === 400 && row.tokens_forwarded < row.tokens_from_client;
  const status = rewriteRefused ? h("span", { class: "st critical", ...hoverTip("Refused as LLM-GC rewrote it",
      `The provider refused this request as LLM-GC rewrote it${row.error_type ? ` (${row.error_type})` : ""}. LLM-GC sent Claude Code's own request in its place, the next row, so Claude Code never saw the refusal. The proxy's log has the provider's reason.`) },
      ` ✕ ${row.status} refused as LLM-GC rewrote it`)
    : failed ? h("span", { class: "st critical" }, ` ✕ ${row.status} ${STATUS_WORDS[row.status] || "refused"}`)
    : row.ended_at === null ? h("span", { class: "secondary" }, " … running")
    : !row.finished && row.kind !== "interrupted" ? h("span", { class: "st warn", ...hoverTip("Cut short", "The reply did not reach its end: the client stopped reading, or the connection dropped.") }, " cut short") : null;
  const kept = row.tokens_from_client - row.tokens_forwarded;
  return h("div", { class: `wf-row kind-${row.kind}${depth ? " nested" : ""}`, role: "row" },
    h("span", { class: "wf-label", role: "cell" },
      h("span", { class: "wf-kind", ...hoverTip(label, about, requestIdTip(row)) }, label),
      h("span", { class: "wf-tokens", ...(kept > 0 ? hoverTip("Tokens", `Claude Code sent ${fmt(row.tokens_from_client)}; LLM-GC took out ${fmt(kept)}, so the model got ${fmt(row.tokens_forwarded)}. Both include LLM-GC's estimate of the model's earlier thinking and each image at the size the provider gave for it; a collection stops sending the thinking, and the images of the messages it takes.`) : {}) },
        ` · ${fmt(row.tokens_from_client)}`, kept > 0 ? [" → ", h("span", { class: "good" }, fmt(row.tokens_forwarded))] : null),
      row.tool_calls.length ? h("span", { class: "wf-calls" }, ` · then ${calls(row.tool_calls)}`) : null,
      row.resent_as_written ? h("span", { class: "st warn", ...hoverTip("Sent as written",
        "The provider refused LLM-GC's rewrite of this request, so LLM-GC sent it as Claude Code wrote it: the model got the whole conversation, as it would without LLM-GC.") }, " · sent as written") : null,
      row.carries_collections ? h("span", { class: reasonsOf(row).includes("stale_weight_at_bound") ? "wf-mark st warn" : "wf-mark",
        ...hoverTip(TRACE_KINDS.collection[0], TRACE_KINDS.collection[1] + reasonNote(row) + rewriteNote(row)) },
        row.provider && row.provider.cache_creation_input_tokens ? ` ◆ ${fmt(row.provider.cache_creation_input_tokens)} written to the cache` : " ◆") : null,
      status),
    bar(row.forwarded_at, row.ended_at, window, `kind-${row.kind}`, `${clock(row.forwarded_at)} · ${row.ended_at ? seconds(at(row.ended_at) - at(row.forwarded_at)) : "running"}`));
}

// The ID this request's lines in the proxy's log carry, so a row that looks wrong can be followed there.
// A request LLM-GC sent twice, because the provider refused its rewrite, shows the same ID on both rows.
function requestIdTip(row) {
  return h("div", { class: "tip-row tip-id" }, "In the proxy's log as request ", h("span", { class: "mono" }, row.request_id));
}

// Why each collection this request carries was applied then, as the switch policy decided it. A proxy
// started before reasons were recorded sends none, and its page is served the current scripts.
function reasonsOf(row) {
  return row.switch_reasons ?? [];
}

function reasonNote(row) {
  const reasons = reasonsOf(row).filter((reason) => reason in SWITCH_REASON);
  return reasons.length ? ` Applied because ${reasons.map((reason) => SWITCH_REASON[reason]).join("; and another because ")}.` : "";
}

// What a collection's request wrote to the cache, for the ◆'s explanation, beside what LLM-GC expected it to write.
function rewriteNote(row) {
  const usage = row.provider;
  if (!usage || !usage.cache_creation_input_tokens) return "";
  const written = ` The recap changed, so the provider cached everything after the system prompt again: ${fmt(usage.cache_creation_input_tokens)} tokens written.`;
  if (row.rewritten_tokens == null || !row.tokens_forwarded) return written;
  // This request's own rate between the two counts, so the comparison does not depend on any other request.
  const rate = (usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens) / row.tokens_forwarded;
  return `${written} LLM-GC expected ${fmt(row.rewritten_tokens)} of its own tokens to be cached again, about ${fmt(Math.round(row.rewritten_tokens * rate))} by the provider's count at this request's rate, besides the new message.`;
}

function subagentLines(agent, window, actions) {
  const key = `agent:${agent.agent}`;
  const open = ui.openAgents.has(key);
  const t = agent.totals;
  const link = agent.spawned_by;
  const head = h("div", { class: "wf-row wf-agent", role: "row" },
    h("span", { class: "wf-label", role: "cell" },
      h("button", { class: "link-btn", type: "button", "aria-expanded": String(open), onclick: () => toggle(ui.openAgents, key, actions) },
        h("span", { class: "chev", "aria-hidden": "true" }, open ? "▼" : "▶"), ` ${agent.label}`),
      actions.live(agent.conversation_id)
        ? h("button", { class: "link-btn wf-open", type: "button", onclick: () => actions.open(agent.conversation_id, null) }, "Open →") : null,
      h("span", { class: "wf-tokens" }, ` ${plural(t.requests, "request")} · ${fmt(t.tokens_from_client)} tokens sent`),
      h("span", { class: "secondary", ...hoverTip(TRACE_KINDS.subagent[0], TRACE_KINDS.subagent[1]) },
        link ? ` · started by a ${link.tool} call` : " · its starting call wasn't matched", agent.collected ? "" : " · recorded, not collected"),
      agent.words ? h("span", { class: "wf-words" }, `“${agent.words}”`) : null),
    bar(agent.started_at, agent.ended_at, window, "agent", `${clock(agent.started_at)} · ${agent.ended_at ? seconds(at(agent.ended_at) - at(agent.started_at)) : "running"}`));
  return [head, ...(open ? agent.requests.map((row) => requestLine(row, window, 1)) : [])];
}

function bar(from, to, window, kind, title) {
  const total = window.end - window.start;
  const left = Math.max(0, (at(from) - window.start) / total) * 100;
  const end = (to ? at(to) : Date.now()) - window.start;
  const width = Math.max(0.6, (end / total) * 100 - left);
  return h("span", { class: "wf-track", role: "cell" },
    h("span", { class: `wf-bar ${kind}${to ? "" : " running"}`, style: `left:${left.toFixed(2)}%;width:${Math.min(width, 100 - left).toFixed(2)}%`, title }));
}

function toggle(set, key, actions) {
  if (set.has(key)) set.delete(key); else set.add(key);
  renderTrace(actions, true);
}
