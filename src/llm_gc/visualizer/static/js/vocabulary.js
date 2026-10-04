// The words the dashboard uses, and what each status and category means.

import { $, compact, h } from "./dom.js";

export const ACTION_HEADER = { "x-llm-gc-dashboard": "1" };
export const RUN_MIN = 4;              // consecutive condensed-away messages folded into one row
export const FACTS_SHOWN = 6;          // facts listed per group before "show all"
export const EARLIER_SHOWN = 8;        // newest rows shown before "Show N earlier messages" folds the rest
export const CELL_MAX_PX = 46;         // widest a message gets on the map
export const CLAMP_CHARS = 420;        // longer messages start collapsed
export const CHART_HEIGHT = 260;

// One explanation per term, used by every ⓘ and by the guide. The design docs use
// the words in brackets.
export const GLOSSARY = {
  tokens: ["Tokens", "The unit models read and providers bill: roughly three quarters of a word. A model's window holds a fixed number of them."],
  size: ["Tokens per request", "The conversation as written, against what LLM-GC forwards for it: the condensed block, then the recent messages word for word. Your newest message is counted once its reply arrives."],
  window: ["The model's window", "The most the model can read in one request. Filling it is what forces old context out; LLM-GC decides what goes, instead of the provider truncating."],
  cleanup: ["Clean-up (a “cut”)", "LLM-GC does not rewrite the conversation on every message: changing the start of the prompt makes the provider drop its cache and the model lose its earlier reasoning. So it waits until older, low-relevance messages add up to enough to be worth it, then condenses them all at once. The newest messages always stay word for word."],
  ready: ["Ready to condense", "Older messages rated low enough to be summarised or reduced to facts. Short messages and the newest ones are kept whole, so they do not count. When this reaches the target, the next reply triggers a clean-up."],
  block: ["Condensed block (the “front”)", "One message at the start of the prompt standing in for everything condensed so far: important messages quoted whole, routine ones summarised, cold ones reduced to facts. It stays exactly the same until the next clean-up, so the provider can keep it cached."],
  verbatim: ["Word for word", "The model gets this message exactly as it was written."],
  quoted: ["Quoted whole", "Moved into the condensed block, still word for word: it was rated too important to summarise, or is too short to be worth it."],
  summarised: ["Summarised", "Condensed together with its neighbours into a short summary. The model reads the summary, not the original."],
  facts: ["Kept as facts", "Only facts pulled out of it remain, like “database: postgres”, and at least one of them is in the condensed block."],
  unseen: ["Out of view", "Facts from it were stored, but the condensed block carries only a few and none of these made it. The model cannot see anything of this message."],
  superseded: ["Contradicted", "A later message changed what this one said (a different database, say), so its old fact is no longer sent."],
  next: ["Next clean-up", "The older messages the next clean-up would take, and what each would become. Nothing changes until it runs."],
  relevance: ["Relevance", "A score from 0 to 1 combining how recent a message is, whether later messages refer back to it, how much specific detail it holds, and whether it records a decision."],
  check: ["Checks (“passes”)", "After every reply LLM-GC checks whether a clean-up is due. Most checks find nothing to do; that is normal, and takes a few milliseconds."],
  tracking: ["Conversation tracking", "LLM-GC recognises a conversation by its content. An edited or regenerated earlier message rewinds it; a change it cannot rewind restarts it; a conversation it has never seen, or one after a proxy restart, starts fresh. A restart or a fresh start costs one request that is not condensed."],
  forwarded: ["Tokens sent to the provider", "Totals across every request through the proxy. Each request carries the whole conversation again, so these grow much faster than the conversation itself."],
};
export const GUIDE_TERMS = ["cleanup", "block", "relevance", "check", "tokens", "window"];

// What the model still has of a message, from most to least.
export const CATS = {
  verbatim:   { label: "Word for word", tag: "word for word", desc: "sent exactly as written" },
  quoted:     { label: "Quoted whole", tag: "quoted whole", desc: "in the condensed block, still word for word" },
  summarised: { label: "Summarised", tag: "summarised", desc: "the model reads a summary instead" },
  facts:      { label: "Kept as facts", tag: "kept as facts", desc: "only facts pulled out of it remain" },
  unseen:     { label: "Out of view", tag: "out of view", desc: "facts stored, none sent: the model sees nothing of it" },
  superseded: { label: "Contradicted", tag: "contradicted", desc: "a later message changed what it said" },
};
export const ORDER = ["verbatim", "quoted", "summarised", "facts", "unseen", "superseded"];
export const GONE = new Set(["facts", "unseen", "superseded"]);
export const BECOMES = { quoted: "quoted whole", summarised: "summarised", facts_only: "reduced to facts" };
export const BECOMES_CAT = { quoted: "quoted", summarised: "summarised", facts_only: "facts" };
export const CLASS_TO_CAT = { keep: "verbatim", compact: "summarised", archive: "facts" };
export const CLASS_LABEL = { keep: "keep whole", compact: "summarise", archive: "reduce to facts" };
export const STATUS = {
  completed:                { tone: "good", icon: "✓", text: "clean-up made" },
  bypassed_batch_too_small: { tone: "", icon: "–", text: "nothing to clean up yet" },
  bypassed_no_saving:       { tone: "", icon: "–", text: "skipped: it would have made the prompt bigger" },
  bypassed_aging_in_flight: { tone: "", icon: "–", text: "skipped: a clean-up was already running" },
  bypassed_below_threshold: { tone: "", icon: "–", text: "skipped: below threshold" },
  bypassed_on_error:        { tone: "critical", icon: "✕", text: "failed: nothing changed" },
  discarded_session_closed: { tone: "", icon: "–", text: "discarded: the conversation had closed" },
};
export const TRANSITION = { promoted: "Summarised", archived: "Reduced to facts", superseded: "Fact contradicted" };

export const swatch = (key) => h("span", { class: `swatch f-${key}` });
export const categoryOf = (turn) => turn.fidelity === "facts_only" ? (turn.facts_sent > 0 ? "facts" : "unseen") : turn.fidelity;
export const speaker = (role) => role === "user" ? "you" : role || "—";
