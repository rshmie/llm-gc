// The Session tab: a Claude Code session turn by turn, each turn's requests on a waterfall in time,
// each subagent under the request whose reply started it. Measured, never forecast: every number is a
// request that happened.

import { TRACE_KINDS } from "./vocabulary.js";
import { $, compact, fill, fmt, h, plural, share } from "./dom.js";
import { state, ui } from "./state.js";
import { hoverTip } from "./tooltip.js";

// Requests that are not part of the conversation fold into one line per turn.
const SIDE_KINDS = new Set(["unclaimed", "replaced"]);
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
const STATUS_WORDS = { 401: "credential refused", 403: "credential refused", 413: "too large", 429: "rate limited",
  500: "upstream error", 502: "upstream unreachable", 503: "upstream unavailable", 529: "overloaded" };

// `force` redraws after a toggle; otherwise only when the trace changed, so a poll keeps hover and focus.
export function renderTrace(actions, force = false) {
  const trace = state.trace;
  const key = trace ? JSON.stringify(trace) : null;
  if (!force && key === shownKey) return;
  shownKey = key;
  $("trace-card").hidden = !trace;
  if (!trace) return;
  fill("trace-card", traceHead(trace), before(trace, actions),
    h("ol", { class: "turns" }, [...trace.turns].reverse().map((turn) => turnItem(turn, trace, actions))));
}

function traceHead(trace) {
  const t = trace.totals;
  const spent = trace.summaries;
  const subagents = trace.turns.reduce((n, turn) => n + turn.subagents.length, 0) + trace.subagents_before.length;
  return [
    h("div", { class: "card-head" }, h("div", {},
      h("h2", {}, `Claude Code session ${shortId(trace.client_session)}`),
      h("p", { class: "sub" }, `Since ${clock(trace.started_at)} · ${plural(trace.turns.length, "turn")} · ${plural(t.requests, "request")} · ${plural(subagents, "subagent")}. Newest turn first; open one to see every request it made.`))),
    h("p", { class: "session-totals" },
      "Claude Code sent ", h("b", {}, fmt(t.tokens_from_client)), " tokens overall · LLM-GC forwarded ", h("b", {}, fmt(t.tokens_forwarded)),
      " · ", h("b", {}, saved(t))),
    h("p", { class: "session-totals" },
      `The provider's own count: ${compact(t.cache_read_input_tokens)} cache read · ${compact(t.input_tokens)} input · `,
      h("b", {}, `${compact(t.cache_creation_input_tokens)} cache write`), ` · ${compact(t.output_tokens)} output.`,
      t.collection_cache_write_tokens ? [" ", h("span", hoverTip(TRACE_KINDS.collection[0], TRACE_KINDS.collection[1]),
        h("b", {}, compact(t.collection_cache_write_tokens)), " of the cache writes were on requests that carried a collection.")] : ""),
    h("p", { class: "session-totals secondary" },
      "Tokens are not all worth the same: a cache read costs a fraction of sending a token plainly, a cache write more than sending it. ",
      "The tokens LLM-GC removes would mostly have been cache reads, so set what it saved against the cache writes its collections caused.",
      spent && spent.calls ? ` LLM-GC's own model calls: ${plural(spent.calls, "call")}, ${compact(spent.input_tokens)} tokens in, ${compact(spent.output_tokens)} out.`
        : " LLM-GC has made no model calls for this session."),
  ];
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
      : null,
    // Newest first, so the turn that began a new main conversation sits just above the line that marks it.
    turn.new_conversation ? h("div", { class: "turn-boundary", ...hoverTip(TRACE_KINDS.boundary[0], TRACE_KINDS.boundary[1]) },
      "↑ A new main conversation began with this turn: after /compact, or an edited message") : null);
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
  const status = failed ? h("span", { class: "st critical" }, ` ✕ ${row.status} ${STATUS_WORDS[row.status] || "refused"}`)
    : row.ended_at === null ? h("span", { class: "secondary" }, " … running")
    : !row.finished ? h("span", { class: "st warn", ...hoverTip("Cut short", "The reply did not reach its end: the client stopped reading, or the connection dropped.") }, " cut short") : null;
  const kept = row.tokens_from_client - row.tokens_forwarded;
  return h("div", { class: `wf-row kind-${row.kind}${depth ? " nested" : ""}`, role: "row" },
    h("span", { class: "wf-label", role: "cell" },
      h("span", { class: "wf-kind", ...hoverTip(label, about) }, label),
      h("span", { class: "wf-tokens", ...(kept > 0 ? hoverTip("Tokens", `Claude Code sent ${fmt(row.tokens_from_client)}; LLM-GC condensed ${fmt(kept)} of them, so the model got ${fmt(row.tokens_forwarded)}.`) : {}) },
        ` · ${fmt(row.tokens_from_client)}`, kept > 0 ? [" → ", h("span", { class: "good" }, fmt(row.tokens_forwarded))] : null),
      row.tool_calls.length ? h("span", { class: "wf-calls" }, ` · then ${calls(row.tool_calls)}`) : null,
      row.carries_collections ? h("span", { class: "wf-mark", ...hoverTip(TRACE_KINDS.collection[0], TRACE_KINDS.collection[1] + rewriteNote(row)) },
        row.provider && row.provider.cache_creation_input_tokens ? ` ◆ ${fmt(row.provider.cache_creation_input_tokens)} written to the cache` : " ◆") : null,
      status),
    bar(row.forwarded_at, row.ended_at, window, `kind-${row.kind}`, `${clock(row.forwarded_at)} · ${row.ended_at ? seconds(at(row.ended_at) - at(row.forwarded_at)) : "running"}`));
}

// What a collection's request wrote to the cache, for the ◆'s explanation.
function rewriteNote(row) {
  if (!row.provider || !row.provider.cache_creation_input_tokens) return "";
  return ` The provider wrote ${fmt(row.provider.cache_creation_input_tokens)} tokens of this request to its cache: the recap changed, so everything after the system prompt was cached again. A cache write costs more than sending the tokens plainly.`;
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
      h("span", { class: "wf-tokens" }, ` ${plural(t.requests, "request")} · ${fmt(t.tokens_from_client)} sent overall`),
      h("span", { class: "secondary", ...hoverTip(TRACE_KINDS.subagent[0], TRACE_KINDS.subagent[1]) },
        link ? ` · started by ${link.tool}` : " · started by a call not determined", agent.collected ? "" : " · not collected"),
      h("span", { class: "wf-words" }, agent.words ? ` · “${agent.words}”` : "")),
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
