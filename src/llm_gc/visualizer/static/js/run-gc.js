// Run GC now: the dashboard's one action, and the explanation of its outcome.

import { ACTION_HEADER, NOT_COLLECTED_TEXT, PAUSED_TEXT, STATUS, describeOutcome } from "./vocabulary.js";
import { $, fmt, h, plural, setChildren } from "./dom.js";
import { keepLast, sessionPath, state } from "./state.js";
import { showTab } from "./shell.js";

const pauseEnds = () => state.pause ? `It ends at ${new Date(state.pause.until).toLocaleTimeString()}.` : "It has ended; try again.";

export function explainResult(r, before) {
  const figures = (after) => [["before", `${fmt(r.tokens_before)} tokens`], ["after", `${fmt(after)} tokens`]];
  const problems = r.degradations.length
    ? `⚠ ${plural(r.degradations.length, "part")} of the collection fell short; each says what the model gets instead: see What fell short, on the Engine tab.` : null;
  if (r.status === "completed") {
    const o = r.outcome;
    const moved = o.kept.length + o.compacted.length + o.archived.length + o.folded.length;
    const smaller = r.tokens_after < r.tokens_before;
    return { tone: "good", title: smaller ? `✓ Collected: the model now gets ${fmt(r.tokens_before - r.tokens_after)} fewer tokens` : "✓ Collected",
      body: [`${plural(moved, "older message")} moved into the recap: ${describeOutcome({ ...o, lowered: [], raised: [] })}. The newest ones are untouched.`,
        o.lowered.length ? `${plural(o.lowered.length, "message")} collected earlier ${o.lowered.length === 1 ? "is" : "are"} now shown with less detail: the conversation moved on from them, or the recap needed the room.` : null,
        o.raised.length ? `${plural(o.raised.length, "message")} collected earlier ${o.raised.length === 1 ? "is" : "are"} shown in more detail again, because the conversation came back to them${smaller ? "" : ", so the prompt is larger than before"}.` : null,
        "From now on every request carries this version. Below, collected messages show what the model reads instead of them."].filter(Boolean),
      figures: figures(r.tokens_after), extra: problems, show: true };
  }
  if (r.status === "bypassed_no_saving" && r.abandoned_cut) {
    const a = r.abandoned_cut, o = a.outcome;
    const why = [
      o.kept.length ? `${plural(o.kept.length, "message")} would be kept, quoted word for word (they are short, or matter now), and each quote carries a small marker, so it grows a little` : null,
      o.archived.length ? `${plural(o.archived.length, "message")} would be archived, to their exchange's opening line and facts` : null,
      o.compacted.length ? `${plural(o.compacted.length, "message")} would be compacted, summarised` : null,
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
      body: [`The ${plural(cut.turns.length, "older message")} would all be kept word for word (they are short, or matter now), so collecting them would save nothing.`] };
  }
  if (r.status === "bypassed_aging_in_flight") return { tone: "muted", title: "– A collection is already running", body: ["LLM-GC is already collecting this conversation. Try again in a moment."] };
  if (r.status === "bypassed_paused") return { tone: "muted", title: "⏸ Paused: LLM-GC is yielding to a rate limit", body: [PAUSED_TEXT, pauseEnds()] };
  if (r.status === "bypassed_not_collected") return { tone: "muted", title: "– Not collected", body: [NOT_COLLECTED_TEXT] };
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
    body: [`The conversation is still within its newest ${keepLast()} messages, which always stay recent, sent as written. Once older ones build up, Run GC now can collect them without waiting for the ${minimum} minimum.`] };
}

// The button says what it would do before it is pressed; it stays clickable to explain itself.
export function renderRunGcButton() {
  const button = $("run-gc");
  if (button.dataset.running === "1") return;
  const k = state.knowledge;
  const cut = k ? k.next_cut : null;
  const idle = !!cut && cut.turns.length === 0;
  button.setAttribute("aria-disabled", String(idle));
  button.title = k && !k.collected ? "Not collected: a subagent's conversation is recorded and sent as written."
    : !cut ? "Collect this conversation now, without waiting for the minimum."
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
