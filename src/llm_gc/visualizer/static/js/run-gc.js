// Run GC now: the dashboard's one action, and the explanation of its outcome.

import { ACTION_HEADER, STATUS, describeOutcome } from "./vocabulary.js";
import { $, fmt, h, plural, setChildren } from "./dom.js";
import { keepLast, sessionPath, state } from "./state.js";
import { showTab } from "./shell.js";

export function explainResult(r, before) {
  const figures = (after) => [["before", `${fmt(r.tokens_before)} tokens`], ["after", `${fmt(after)} tokens`]];
  const problems = r.degradations.length
    ? `⚠ ${plural(r.degradations.length, "planned move")} didn't happen, so those messages stayed word for word: see What didn't move, on the Engine tab.` : null;
  if (r.status === "completed" && r.tokens_after < r.tokens_before) {
    const o = r.outcome;
    const moved = o.quoted + o.summarised + o.archived;
    return { tone: "good", title: `✓ Collected: the model now gets ${fmt(r.tokens_before - r.tokens_after)} fewer tokens`,
      body: [`${plural(moved, "older message")} moved into the recap: ${describeOutcome({ ...o, front_archived: 0 })}. The newest ones are untouched.`,
        o.front_archived ? `${plural(o.front_archived, "message")} collected earlier had gone cold and ${o.front_archived === 1 ? "was" : "were"} archived too.` : null,
        "From now on every request carries this smaller version. Below, collected messages show what the model reads instead of them."].filter(Boolean),
      figures: figures(r.tokens_after), extra: problems, show: true };
  }
  if (r.status === "completed") {
    return { tone: "muted", title: "– Nothing changed", body: ["Some of the messages could not be summarised, and what was left was too little to be worth a collection. They stay word for word and will be tried again later."], extra: problems };
  }
  if (r.status === "bypassed_no_saving" && r.abandoned_cut) {
    const a = { ...r.abandoned_cut, ...r.abandoned_cut.outcome };
    const why = [
      a.quoted ? `${plural(a.quoted, "message")} would be kept, quoted whole (short or important messages are never compacted), and each quote carries a small label, so it grows a little` : null,
      a.archived ? `${plural(a.archived, "message")} would be archived, reduced to facts, but a message with nothing specific to pull out is kept as a full excerpt, no smaller than before` : null,
      a.summarised ? `${plural(a.summarised, "message")} would be compacted, summarised` : null,
    ].filter(Boolean);
    return { tone: "muted", title: "– Nothing changed: collecting now would have made the prompt bigger",
      body: [`Collecting the ${plural(a.turns.length, "older message")} (${fmt(a.tokens_taken)} tokens now) would have left the model with ${fmt(a.context_tokens_after)} tokens instead of ${fmt(r.tokens_before)}. A collection is only made when it shrinks what the model gets, so LLM-GC left them word for word.`],
      list: why,
      after: "This is normal early in a conversation and when messages are short. It pays off as the conversation grows and older messages get longer.",
      figures: [["now", `${fmt(r.tokens_before)} tokens`], ["if collected", `${fmt(a.context_tokens_after)} tokens`]], extra: problems };
  }
  if (r.status === "bypassed_batch_too_small") {
    const cut = before ? before.next_cut : null;
    if (!cut || cut.turns.length === 0) return nothingWaiting(before);
    return { tone: "muted", title: "– Nothing to collect yet",
      body: [`The ${plural(cut.turns.length, "older message")} are all being kept whole (they are short, or rated important), so collecting them would save nothing.`] };
  }
  if (r.status === "bypassed_aging_in_flight") return { tone: "muted", title: "– A collection is already running", body: ["LLM-GC is already collecting this conversation. Try again in a moment."] };
  if (r.status === "bypassed_on_error") return { tone: "critical", title: "✕ The collection failed, and nothing changed", body: ["The conversation is exactly as it was. The error is in the proxy's log."] };
  return { tone: "muted", title: STATUS[r.status]?.text || r.status, body: [] };
}

// Nothing is waiting outside the newest messages: either nothing has aged out yet,
// or the check after each reply has already collected everything that did.
function nothingWaiting(k) {
  const minimum = state.config ? `${fmt(state.config.min_cut_tokens)}-token` : "token";
  if (k && k.cuts) {
    return { tone: "muted", title: "– Nothing to collect: the automatic check already did",
      body: [`Everything older than the newest ${keepLast()} messages has already been collected. GC checks after every reply, and collected each message as soon as it left the newest ${keepLast()}.`,
        `Run GC now skips only the ${minimum} minimum, so it helps when older messages are waiting below it: short messages, or early in a conversation. None are waiting now.`] };
  }
  return { tone: "muted", title: "– Nothing to collect yet",
    body: [`The conversation is still within its newest ${keepLast()} messages, which always stay word for word. Once older ones build up, Run GC now can collect them without waiting for the ${minimum} minimum.`] };
}

// The button says what it would do before it is pressed; it stays clickable to explain itself.
export function renderRunGcButton() {
  const button = $("run-gc");
  if (button.dataset.running === "1") return;
  const k = state.knowledge;
  const cut = k ? k.next_cut : null;
  const idle = !!cut && cut.turns.length === 0;
  button.setAttribute("aria-disabled", String(idle));
  button.title = !cut ? "Collect this conversation now, without waiting for the minimum."
    : idle ? (k.cuts ? `Nothing to collect: everything older than the newest ${keepLast()} messages is already collected. Click to see why.`
      : `Nothing to collect yet: the conversation is within its newest ${keepLast()} messages.`)
    : `Collect the ${plural(cut.turns.length, "waiting message")} now: ${fmt(cut.compactable_tokens)} of ${fmt(cut.min_cut_tokens)} tokens ready, without waiting for the minimum.`;
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

export async function runGcNow(refresh) {
  const button = $("run-gc");
  const before = state.knowledge;
  button.dataset.running = "1";
  button.disabled = true;
  button.textContent = "Collecting…";
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
