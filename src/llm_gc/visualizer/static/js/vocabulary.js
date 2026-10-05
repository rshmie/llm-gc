// The words the dashboard uses, and what each status and category means.

import { $, compact, h } from "./dom.js";

export const ACTION_HEADER = { "x-llm-gc-dashboard": "1" };
export const RUN_MIN = 4;              // consecutive collected messages folded into one row
export const FACTS_SHOWN = 6;          // facts listed per group before "show all"
export const EARLIER_SHOWN = 8;        // newest rows shown before "Show N older messages" folds the rest
export const CHECKS_SHOWN = 5;         // rows of "Every check" before "Show all"
export const MOVES_SHOWN = 3;          // collections listed in "What moved" and "What didn't move" before "Show all"
export const MIN_BAR_PX = 10;          // narrowest a message's bar gets on the map, so a one-token message stays clickable
export const CELL_MAX_PX = 46;         // widest a message gets on the map
export const CLAMP_CHARS = 420;        // longer messages start collapsed
export const CHART_HEIGHT = 260;

// One explanation per term, used by every ⓘ and by the guide. The design docs use
// the words in brackets.
export const GLOSSARY = {
  tokens: ["Tokens", "The unit models read and providers bill: roughly three quarters of a word. A model's window holds a fixed number of them."],
  size: ["Each request carries, now", "The size of one request today, not a total: the recap, then the recent messages word for word, compared with the conversation as written. Your newest message is counted once its reply arrives."],
  window: ["The model's window", "The most the model can read in one request. Filling it is what forces old context out; LLM-GC decides what goes, instead of the provider truncating."],
  cleanup: ["Collection (a “cut” in the code)", "LLM-GC does not rewrite the conversation on every message: changing the start of the prompt makes the provider drop its cache and the model lose its earlier reasoning. GC checks after every reply, and only once older, low-relevance messages add up to enough does a check collect them, all at once: each is kept, compacted or archived, and leaves the young generation. The newest messages always stay word for word."],
  ready: ["Ready to collect", "Older messages rated low enough to be compacted or archived. Short messages and the newest ones are kept whole, so they do not count. When this reaches the target, the check after the next reply makes a collection."],
  block: ["Recap (the “front” in the code)", "One message at the start of every request standing in for everything older: kept messages quoted whole, compacted ones as summaries, archived ones as facts. It stays exactly the same until the next collection, so the provider can keep it cached."],
  verbatim: ["Word for word", "The model gets this message exactly as it was written."],
  quoted: ["Kept", "Quoted whole into the recap, still word for word: rated too important to compact, or too short to be worth it."],
  summarised: ["Compacted", "Summarised together with its neighbours. The model reads the summary in the recap, not the original."],
  facts: ["Archived", "Reduced to the facts pulled out of it, like “database: postgres”; at least one of them is in the recap."],
  unseen: ["Out of view", "Facts from it were stored, but the recap carries only a few and none of these made it. The model cannot see anything of this message."],
  superseded: ["Contradicted", "A later message changed what this one said (a different database, say), so its old fact is no longer sent."],
  next: ["Next collection", "The older messages the next collection would take, and what each would become. Nothing changes until it runs."],
  relevance: ["Relevance", "A score from 0 to 1 combining how recent a message is, whether later messages refer back to it, how much specific detail it holds, and whether it records a decision."],
  check: ["Checks (GC passes)", "After every reply LLM-GC checks whether a collection is due. Most checks find nothing to do; that is normal, and takes a few milliseconds."],
  tracking: ["Conversation tracking", "LLM-GC recognises a conversation by its content. An edited or regenerated earlier message rewinds it; a change it cannot rewind restarts it; a conversation it has never seen, or one after a proxy restart, starts fresh. A restart or a fresh start costs one request that is not condensed."],
  forwarded: ["Total sent since the proxy started", "Every request added up, across all conversations, since the proxy last started; a restart sets it back to zero. Each request carries the whole conversation again, so this grows much faster than any conversation. The size of one request today is on the Conversation tab."],
};
export const GUIDE_TERMS = ["relevance", "tokens", "window"];

// What the model still has of a message, from most to least.
export const CATS = {
  verbatim:   { label: "Word for word", tag: "word for word", desc: "sent exactly as written" },
  quoted:     { label: "Kept", tag: "kept", desc: "quoted whole in the recap" },
  summarised: { label: "Compacted", tag: "compacted", desc: "the model reads a summary instead" },
  facts:      { label: "Archived", tag: "archived", desc: "only facts pulled out of it remain" },
  unseen:     { label: "Out of view", tag: "out of view", desc: "facts stored, none sent: the model sees nothing of it" },
  superseded: { label: "Contradicted", tag: "contradicted", desc: "a later message changed what it said" },
};
export const ORDER = ["verbatim", "quoted", "summarised", "facts", "unseen", "superseded"];
export const GONE = new Set(["facts", "unseen", "superseded"]);
export const BECOMES = { quoted: "kept (quoted whole)", summarised: "compacted (summarised)", facts_only: "archived (reduced to facts)" };
export const BECOMES_CAT = { quoted: "quoted", summarised: "summarised", facts_only: "facts" };
export const CLASS_TO_CAT = { keep: "verbatim", compact: "summarised", archive: "facts" };
export const CLASS_LABEL = { keep: "keep", compact: "compact", archive: "archive" };
export const STATUS = {
  completed:                { tone: "good", icon: "✓", text: "collection made" },
  bypassed_batch_too_small: { tone: "", icon: "–", text: "nothing to collect yet" },
  bypassed_no_saving:       { tone: "", icon: "–", text: "skipped: it would have made the prompt bigger" },
  bypassed_aging_in_flight: { tone: "", icon: "–", text: "skipped: a collection was already running" },
  bypassed_below_threshold: { tone: "", icon: "–", text: "skipped: below threshold" },
  bypassed_on_error:        { tone: "critical", icon: "✕", text: "failed: nothing changed" },
  discarded_session_closed: { tone: "", icon: "–", text: "discarded: the conversation had closed" },
};
export const TRANSITION = { promoted: "Compacted", archived: "Archived", superseded: "Fact contradicted",
  promoted_then_archived: "Compacted, then the summary archived" };
// What a clean-up did to the messages it took. Always from the engine's outcome, never the sweeper's plan:
// the same clean-up can overrule the plan, so a message planned to be quoted can end it as facts.
export function describeOutcome(o) {
  return [o.quoted ? `${o.quoted} kept` : null, o.summarised ? `${o.summarised} compacted` : null,
    o.archived ? `${o.archived} archived` : null,
    o.front_archived ? `${o.front_archived} collected earlier, now archived` : null].filter(Boolean).join(" · ");
}
// The kind of move behind each transition and each degradation, in the engine's own terms.
export const MOVE_KIND = { promoted: "summarised", archived: "facts", superseded: null, promoted_then_archived: "same collection" };
export const MOVE_CAT = { promoted: "summarised", archived: "facts", superseded: "superseded", promoted_then_archived: "facts" };
export const NOT_MOVED = {
  compaction_refused: ["Summary rejected by the safeguard", "compaction"],
  compaction_failed: ["Summary request failed", "compaction"],
  archive_failed: ["Facts couldn't be stored", "archive"],
};
// How much of a message the model still has, drawn as the height of its bar (percent of full).
export const HEIGHT = { verbatim: 100, quoted: 100, summarised: 52, facts: 24, unseen: 24, superseded: 24 };
// The ⓘ beside each LLM-GC at work card: the explanation its one-line subtitle leaves out.
export const CARD_INFO = {
  generations: ["Generations", "Where each message lives, as in a generational garbage collector. New messages enter the young generation and are sent word for word. A collection (a cut, in the code) moves older ones out: keep quotes a message whole into the old generation, compact summarises it there, archive reduces it to facts in the permanent generation. The old generation and the facts are sent together in the recap, at the start of the prompt; an old-generation entry that goes cold is archived in turn."],
  next: ["Next collection", "GC checks after every reply, and a check collects when two things hold. First, enough older messages to collect: only those to be compacted or archived count, not ones kept whole or the newest few, since those save nothing. Second, the result must be smaller than the prompt is now; that can only be checked as it runs, because summaries aren't written until then. Run GC now skips the first condition, never the second."],
  rate: ["How each message is rated", "Each dot is a message's relevance, from 0 to 1, against your two thresholds: below the lower one it is archived, from the upper one kept, and compacted in between. A ring means a rule keeps it whole whatever its score. Only messages still word for word are rated; collected ones keep no score."],
  moves: ["What moved, and when", "Every message a collection moved, grouped by collection: compacted (summarised) or archived (reduced to facts). A fact contradicted by a later message is listed too."],
  notmoved: ["What didn't move, and why", "Moves a collection planned but couldn't make, with the reason: a summary the safeguard rejected, a request to the provider that failed (compaction), or facts that couldn't be stored (archive). Those messages stay word for word: nothing is lost, the prompt is just bigger."],
  chart: ["Prompt size, check by check", "LLM-GC checks after every reply. The blue line is what the model is sent: it holds its size until a message arrives, then steps up, and drops at each collection. The grey line is the conversation as written. The gap between them is saved on every request."],
  checks: ["Every check", "Checks that found nothing to do fold into one row, so collections stand out in bold. A check that failed outright shows here too."],
  settings: ["Settings in effect", "First guesses, meant to be tuned. Settings with a flag can be changed when starting the proxy; the rest are GCConfig fields."],
};

export const swatch = (key) => h("span", { class: `swatch f-${key}` });
export const categoryOf = (turn) => turn.fidelity === "facts_only" ? (turn.facts_sent > 0 ? "facts" : "unseen") : turn.fidelity;
export const speaker = (role) => role === "user" ? "you" : role || "—";
