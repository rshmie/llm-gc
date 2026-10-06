// The words the dashboard uses, and what each status and category means. The categories are the
// engine's own detail levels, so the screen, the code and the recap's markers say the same thing.

import { h } from "./dom.js";

export const ACTION_HEADER = { "x-llm-gc-dashboard": "1" };
export const RUN_MIN = 4;              // consecutive archived exchanges folded into one row
export const FACTS_SHOWN = 6;          // facts listed per group before "show all"
export const EARLIER_SHOWN = 8;        // newest rows shown before "Show N older messages" folds the rest
export const CHECKS_SHOWN = 5;         // rows of "Every check" before "Show all"
export const MOVES_SHOWN = 3;          // collections listed in "What moved" and "What fell short" before "Show all"
export const MIN_BAR_PX = 10;          // narrowest a message's bar gets on the map, so a one-token message stays clickable
export const CELL_MAX_PX = 46;         // widest a message gets on the map
export const CLAMP_CHARS = 420;        // longer messages start collapsed
export const CHART_HEIGHT = 260;

// One explanation per term, used by every ⓘ and by the guide. The design docs use
// the words in brackets.
export const GLOSSARY = {
  tokens: ["Tokens", "The unit models read and providers bill: roughly three quarters of a word. A model's window holds a fixed number of them."],
  size: ["Sent with each request", "The size of one request now, not a running total: the recap, then the recent messages as written. Compared with the whole conversation as written, which is what each request would carry without LLM-GC."],
  holds: ["How each message is sent", "Every message the conversation has recorded, by how the model receives it now. Your newest message is recorded, and counted here, once its reply arrives."],
  window: ["The model's window", "The most the model can read in one request. Filling it is what forces old context out; LLM-GC decides what goes, instead of the provider truncating."],
  block: ["Recap (the “front” in the code)", "One message at the start of every request standing in for every older exchange: kept ones quoted word for word, compacted ones as their opening line, facts and summary, archived ones as their opening line and facts. It stays exactly the same until the next collection, so the provider can keep it cached."],
  exchange: ["Exchange (a “span” in the code)", "A message of yours and everything up to your next one: the reply, and any tool calls in between. A collection condenses whole exchanges, so a question is never shown apart from its answer."],
  verbatim: ["Recent (the young generation)", "The newest messages, not collected yet. Each is sent as its own message in the request, exactly as written, with anything attached. A kept message is word for word too, but it has been collected: it is quoted inside the recap, and a later collection can show it with less detail."],
  kept: ["Kept", "Collected, and quoted word for word inside the recap: its exchange matters to what is being discussed now, or is too short for a summary or its opening line to be any smaller. Decided again at every collection, so it can lose detail later."],
  compacted: ["Compacted", "The model reads its exchange's opening line, facts and summary in the recap, not the original words."],
  archived: ["Archived", "The model reads only its exchange's opening line and facts, like “database: postgres”. The words are still stored: if the conversation comes back to it, a later collection can show it in more detail."],
  out_of_view: ["Out of view", "The recap reached its budget with every exchange already at its least detail, so the oldest were folded behind one line. The model is told they exist, not what was said in them. Their words are still stored, and a later collection shows them again if there is room."],
  next: ["Changes at the next collection", "Recent messages the next collection would take, and earlier exchanges it would show at another level, each with a dot in the colour it will become. A forecast: nothing changes until the collection runs, and the conversation can change it before then."],
  relevance: ["Score", "From 0 to 1, for each exchange: how recent it is or how much what you are discussing now is about it, whichever is larger, plus whether it states a decision and how much specific detail it holds. Scored again at every collection."],
  check: ["Checks (GC passes)", "After every reply LLM-GC checks whether a collection is due. Most checks find nothing to do; that is normal, and takes a few milliseconds."],
  tracking: ["Conversation tracking", "LLM-GC recognises a conversation by its content. An edited or regenerated earlier message rewinds it; a change it cannot rewind restarts it; a conversation it has never seen, or one after a proxy restart, starts fresh. A restart or a fresh start costs one request that is not condensed."],
  forwarded: ["Total sent since the proxy started", "Every request added up, across all conversations, since the proxy last started; a restart sets it back to zero. Each request carries the whole conversation again, so this grows much faster than any conversation. The size of one request today is on the Conversation tab."],
};
export const GUIDE_TERMS = ["exchange", "relevance", "tokens", "window"];

// What the model receives of a message, from most to least: the engine's detail levels.
export const CATS = {
  verbatim:    { label: "Recent", tag: "recent", desc: "not collected yet: sent as its own message, exactly as written" },
  kept:        { label: "Kept", tag: "kept", desc: "its exchange is quoted word for word in the recap" },
  compacted:   { label: "Compacted", tag: "compacted", desc: "the model reads its exchange's summary" },
  archived:    { label: "Archived", tag: "archived", desc: "the model reads its exchange's opening line and facts" },
  out_of_view: { label: "Out of view", tag: "out of view", desc: "folded: the model knows its exchange exists, not what was said" },
};
export const ORDER = ["verbatim", "kept", "compacted", "archived", "out_of_view"];
export const BECOMES = { kept: "kept (quoted word for word)", compacted: "compacted (summarised)",
  archived: "archived (opening line and facts)", out_of_view: "out of view (folded)" };
export const STATUS = {
  completed:                { tone: "good", icon: "✓", text: "collection made" },
  bypassed_batch_too_small: { tone: "", icon: "–", text: "nothing to collect yet" },
  bypassed_no_saving:       { tone: "", icon: "–", text: "skipped: it would have made the prompt bigger" },
  bypassed_aging_in_flight: { tone: "", icon: "–", text: "skipped: a collection was already running" },
  bypassed_on_error:        { tone: "critical", icon: "✕", text: "failed: nothing changed" },
  discarded_session_closed: { tone: "", icon: "–", text: "discarded: the conversation had closed" },
};
// What a collection did. Always from the engine's outcome, which names every message it moved: a
// collection re-plans every exchange, so it can move earlier ones too, in either direction.
export const OUTCOME_LINES = [
  ["kept", "Kept", "kept"], ["compacted", "Compacted", "compacted"], ["archived", "Archived", "archived"],
  ["folded", "Out of view", "out_of_view"], ["lowered", "↓ Less detail", null], ["raised", "↑ More detail", null],
];
export function describeOutcome(o) {
  const words = { kept: "kept", compacted: "compacted", archived: "archived", folded: "out of view",
    lowered: "collected earlier, now shown with less detail", raised: "collected earlier, now shown in more detail" };
  return OUTCOME_LINES.map(([key]) => o[key].length ? `${o[key].length} ${words[key]}` : null).filter(Boolean).join(" · ");
}
// What fell short in a collection, and what the model gets instead.
export const NOT_MOVED = {
  compaction_refused: ["Summary rejected by the safeguard", "Its exchange has no summary, so it is kept word for word or archived instead. Nothing is lost."],
  compaction_failed: ["Summary request failed", "Its exchange has no summary, so it is kept word for word or archived instead. Nothing is lost."],
  archive_failed: ["Facts couldn't be read", "Its exchange has no facts: shown compacted or archived, the model reads it without them."],
  recap_over_budget: ["Recap over its budget", "Every exchange is folded out of view and the recap still doesn't fit: the budget is too small for this conversation."],
};
// How much of a message the model still has, drawn as the height of its bar (percent of full).
export const HEIGHT = { verbatim: 100, kept: 100, compacted: 52, archived: 24, out_of_view: 10 };
// The ⓘ beside each card: the explanation its one-line subtitle leaves out.
export const CARD_INFO = {
  remembers: ["Facts read from the conversation", "When a collection makes exchanges, the model that writes their summaries also reads what they settled — choices, names, numbers, dates — as topic: value notes, once, shown the facts already on record. Without a model, simple patterns read your own messages instead. Facts are a small part of what the model receives: a compacted or archived exchange shows its facts on its line in the recap, beside its opening line, and a kept one is quoted whole. When a later message changes a value, the new fact is filed under the same topic, and the exchange that stated the old one is marked “Replaced later”. Only you change a decision you made: when the assistant recommends something else, your decision stays the fact, and the recommendation stays in its exchange's summary."],
  generations: ["Generations", "Where each message lives, as in a generational garbage collector. New messages enter the young generation and are sent word for word. A collection moves older ones into the old generation as exchanges, each shown kept, compacted or archived by how much it matters now, and decided again at every collection. The facts read from every exchange are filed in the permanent generation, which also records what replaced what. The old generation is sent as the recap, at the start of the prompt, within its budget."],
  next: ["Next collection", "GC checks after every reply, and a check collects when two things hold. First, enough older messages to condense: only those to be compacted or archived count, not ones kept word for word or the newest few, since those save nothing. Second, the result must be smaller than the prompt is now. That is estimated here from each summary's target size, the same estimate the collection checks before paying for any summary, and checked again on the real summaries before anything changes. Run GC now skips the first condition, never the second."],
  rate: ["How each exchange is scored", "Each dot is an exchange's score, from 0 to 1, against your two lines: kept from the upper one, archived below the lower one, compacted in between. Every exchange is scored again at every collection, so one the conversation returns to can rise. A score must clear a line by the margin to move, so an exchange near a line doesn't flip back and forth. When the recap is over its budget, the coldest exchanges lose detail first."],
  moves: ["What moved, and when", "Every collection in the recent checks, from its own record: each message it took and the level it was given, the earlier messages it showed with less or more detail, and any fact one of its messages replaced."],
  notmoved: ["What fell short, and why", "Parts of a collection that couldn't be done: a summary the safeguard rejected or that failed, facts that couldn't be read, or a recap over its budget. Each says what the model gets instead. Nothing is lost: an exchange always keeps its words."],
  chart: ["Prompt size, check by check", "LLM-GC checks after every reply. The blue line is what the model is sent: it holds its size until a message arrives, then steps up, and drops at each collection. The grey line is the conversation as written. The gap between them is saved on every request."],
  checks: ["Every check", "Checks that found nothing to do fold into one row, so collections stand out in bold. A check that failed outright shows here too."],
  settings: ["Settings in effect", "First guesses, meant to be tuned. Settings with a flag can be changed when starting the proxy; the rest are GCConfig fields."],
};

export const swatch = (key) => h("span", { class: `swatch f-${key}` });
export const categoryOf = (turn) => turn.fidelity;
