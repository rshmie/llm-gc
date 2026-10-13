// The picked session's conversations, one row each: what each sent and saved, request by request. Below
// the trace on the Session tab, which shows the same requests turn by turn.

import { $, fill, fmt, h, percent } from "./dom.js";
import { clientSessionOf, state } from "./state.js";
import { hoverTip } from "./tooltip.js";
import { NOT_COLLECTED_TEXT } from "./vocabulary.js";

export function windowLabel(window) {
  if (!window) return "";
  const t = window.tokens;
  const size = t % 1_000_000 === 0 ? `${t / 1_000_000}M` : t % 1000 === 0 ? `${t / 1000}k` : fmt(t);
  return window.reported ? size : `${size} assumed`;
}

function partLabels(conversations) {
  let subagents = 0;
  return conversations.map((c) => c.part === "subagent" ? `Subagent ${++subagents}` : c.part === "earlier_main" ? "Earlier main" : "Main");
}

export function renderSessionCard(select) {
  const client = clientSessionOf(state.clientSession);
  $("session-card").hidden = !client;
  if (!client) return;
  const labels = partLabels(client.conversations);
  const live = new Map(state.sessions.map((s) => [s.session_id, s]));
  const row = (label, title, traffic, extra = {}) => h("tr", extra.attrs || {},
    h("td", {}, h("b", {}, label), title ? h("span", { class: "secondary" }, ` · ${title}`) : null),
    h("td", { class: "num" }, fmt(traffic.requests)),
    h("td", { class: "num" }, fmt(traffic.tokens_from_client)),
    h("td", { class: "num" }, traffic.tokens_saved ? `${fmt(traffic.tokens_saved)} (${percent(traffic.tokens_saved, traffic.tokens_from_client)}%)` : "0"),
    h("td", { class: "num" }, extra.collections ?? ""),
    h("td", {}, extra.window ?? ""));
  const rows = client.conversations.map((c, i) => {
    const session = live.get(c.conversation_id);
    const unsent = c.collections_not_yet_sent ? h("span", { class: "st warn", ...hoverTip("Not yet sent",
      "This collection was made after the conversation's newest request, so no request has carried it: it has saved nothing yet. Its saving reaches the model with the next request, and a conversation that has finished never sends one.") },
      ` ⚠ ${fmt(c.collections_not_yet_sent)} not yet sent`) : null;
    const collections = c.collected === false
      ? h("span", { class: "secondary", ...hoverTip("Not collected", NOT_COLLECTED_TEXT) }, "not collected")
      : [fmt(c.collections), unsent];
    return row(labels[i], session ? session.title : "closed", c.traffic, {
      collections,
      window: windowLabel(c.context_window),
      attrs: { class: `${c.conversation_id === state.selected ? "selected" : ""}${session ? " pickable" : ""}`,
        title: session ? "Show this conversation" : "Closed once idle: its counts stay, its messages are gone",
        onclick: session ? () => select(c.conversation_id) : null },
    });
  });
  if (client.outside_conversations.requests) {
    rows.push(row("Other requests", "a conversation's first message before it was recognised, and side requests",
      client.outside_conversations));
  }
  fill("session-card",
    h("div", { class: "card-head" }, h("div", {},
      h("h2", {}, "Conversations in this session"),
      h("p", { class: "sub" }, "One row per conversation: what each sent and saved. Each main conversation is collected on its own; a subagent's is recorded and sent as written. Click a row to open it."))),
    h("div", { class: "table-wrap" }, h("table", { class: "session-table" },
      h("thead", {}, h("tr", {}, ...["Conversation", "Requests", "Sent", "Saved", "Collections", "Window"].map((label, i) =>
        h("th", { class: i && i < 5 ? "num" : "" }, label)))),
      h("tbody", {}, rows))),
    h("p", { class: "note" }, "Sent and Saved count every request whole, system prompt and tools included, with LLM-GC's own tokenizer; the provider's own count is the line above. A window marked assumed is the configured one: the provider has not reported that model's."));
}
