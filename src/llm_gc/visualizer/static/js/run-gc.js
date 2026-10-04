// Run GC now: the dashboard's one action, and the explanation of its outcome.

import { ACTION_HEADER, STATUS } from "./vocabulary.js";
import { $, fmt, h, plural, setChildren } from "./dom.js";
import { keepLast, sessionPath, state } from "./state.js";
import { showTab } from "./shell.js";

export function explainResult(r, before) {
  const figures = (after) => [["before", `${fmt(r.tokens_before)} tokens`], ["after", `${fmt(after)} tokens`]];
  const problems = r.degradations.length
    ? `⚠ ${plural(r.degradations.length, "part")} could not be summarised and stayed word for word: see Problems on LLM-GC at work.` : null;
  if (r.status === "completed" && r.tokens_after < r.tokens_before) {
    const condensed = r.summarised + r.archived;
    return { tone: "good", title: `✓ Cleaned up: the model now gets ${fmt(r.tokens_before - r.tokens_after)} fewer tokens`,
      body: [`${plural(condensed, "older message")} were condensed into the block at the start of the prompt: ${r.summarised} summarised and ${r.archived} reduced to facts. Messages rated important were quoted whole, and the newest ones are untouched.`,
        "From now on every request carries this smaller version. Below, condensed messages show what the model reads instead of them."],
      figures: figures(r.tokens_after), extra: problems, show: true };
  }
  if (r.status === "completed") {
    return { tone: "muted", title: "– Nothing changed", body: ["Some of the messages could not be summarised, and what was left was too little to be worth a clean-up. They stay word for word and will be tried again later."], extra: problems };
  }
  if (r.status === "bypassed_no_saving" && r.abandoned_cut) {
    const a = r.abandoned_cut;
    const why = [
      a.quoted ? `${plural(a.quoted, "message")} would be quoted whole (short or important messages are never summarised), and each quote carries a small label, so it grows a little` : null,
      a.archived ? `${plural(a.archived, "message")} would be reduced to facts, but a message with nothing specific to pull out is kept as a full excerpt, no smaller than before` : null,
      a.summarised ? `${plural(a.summarised, "message")} would be summarised` : null,
    ].filter(Boolean);
    return { tone: "muted", title: "– Nothing changed: cleaning up now would have made the prompt bigger",
      body: [`Condensing the ${plural(a.turns.length, "older message")} (${fmt(a.tokens_taken)} tokens now) would have left the model with ${fmt(a.context_tokens_after)} tokens instead of ${fmt(r.tokens_before)}. A clean-up is only made when it shrinks what the model gets, so LLM-GC left them word for word.`],
      list: why,
      after: "This is normal early in a conversation and when messages are short. It pays off as the conversation grows and older messages get longer.",
      figures: [["now", `${fmt(r.tokens_before)} tokens`], ["if cleaned up", `${fmt(a.context_tokens_after)} tokens`]], extra: problems };
  }
  if (r.status === "bypassed_batch_too_small") {
    const cut = before ? before.next_cut : null;
    if (!cut || cut.turns.length === 0) {
      return { tone: "muted", title: "– Nothing to clean up yet",
        body: [`The newest ${keepLast()} messages always stay word for word, and there are no older ones to condense yet. Keep talking: the clean-up starts once older messages build up.`] };
    }
    return { tone: "muted", title: "– Nothing to clean up yet",
      body: [`The ${plural(cut.turns.length, "older message")} are all being kept whole (they are short, or rated important), so condensing them would save nothing.`] };
  }
  if (r.status === "bypassed_aging_in_flight") return { tone: "muted", title: "– A clean-up is already running", body: ["LLM-GC is already cleaning up this conversation. Try again in a moment."] };
  if (r.status === "bypassed_on_error") return { tone: "critical", title: "✕ The clean-up failed, and nothing changed", body: ["The conversation is exactly as it was. The error is in the proxy's log."] };
  return { tone: "muted", title: STATUS[r.status]?.text || r.status, body: [] };
}

export function showResult(explained) {
  const box = $("result");
  box.className = `card result tone-${explained.tone}`;
  setChildren(box,
    h("h2", {}, explained.title),
    explained.body.map((paragraph) => h("p", {}, paragraph)),
    explained.list && explained.list.length ? [h("p", {}, "Why:"), h("ul", {}, explained.list.map((item) => h("li", {}, item)))] : null,
    explained.after ? h("p", {}, explained.after) : null,
    explained.extra ? h("p", {}, explained.extra) : null,
    explained.figures ? h("div", { class: "figures" }, explained.figures.map(([label, value]) => h("div", {}, h("b", {}, value), label))) : null,
    h("div", { class: "actions" },
      explained.show ? h("button", { class: "btn", type: "button", onclick: () => { box.hidden = true; showTab("conversation"); $("map").scrollIntoView({ behavior: "smooth", block: "center" }); } }, "See what changed") : null,
      h("button", { class: "btn", type: "button", onclick: () => { box.hidden = true; } }, "Dismiss")));
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

export async function runGcNow() {
  const button = $("run-gc");
  const before = state.knowledge;
  button.dataset.running = "1";
  button.disabled = true;
  button.textContent = "Cleaning up…";
  try {
    const response = await fetch(sessionPath("/gc"), { method: "POST", headers: ACTION_HEADER });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    showResult(explainResult(await response.json(), before));
  } catch (err) {
    showResult({ tone: "critical", title: "✕ Run GC now could not reach LLM-GC", body: [`Nothing changed (${err.message}). Is the proxy still running?`] });
  } finally {
    button.dataset.running = "0";
    button.textContent = "Run GC now";
    state.version = null;
    await refresh();
  }
}
