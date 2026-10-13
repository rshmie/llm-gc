// The words the dashboard uses, and what each status and category means. The categories are the
// engine's own detail levels, so the screen, the code and the recap's markers say the same thing.

import { h } from "./dom.js";

export const ACTION_HEADER = { "x-llm-gc-dashboard": "1" };
export const RUN_MIN = 4;              // consecutive archived exchanges folded into one row
export const EARLIER_SHOWN = 8;        // newest rows shown before "Show N older messages" folds the rest
export const CHECKS_SHOWN = 5;         // rows of "Every check" before "Show all"
export const MAX_EXCHANGE_LINKS = 6;   // exchanges a generation card links to, per level, before "and N more"
export const FAR_JUMP_SCREENS = 1.5;   // a jump to a message further away than this many screens is instant: a long smooth scroll can stop part way
export const MOVES_SHOWN = 3;          // collections listed in "What moved" and "What fell short" before "Show all"
export const MAP_PX_PER_MESSAGE = 14; // room the map keeps per message, so a long conversation scrolls sideways instead of shrinking to slivers
export const MIN_BAR_PX = 10;          // narrowest a message's bar gets on the map, so a one-token message stays clickable
export const CELL_MAX_PX = 46;         // widest a message gets on the map
export const CLAMP_CHARS = 420;        // longer messages start collapsed
export const CHART_HEIGHT = 260;

// One explanation per term, used by every ⓘ and by the guide. The design docs use
// the words in brackets.
export const GLOSSARY = {
  tokens: ["Tokens", "The unit models read and providers bill: roughly three quarters of a word. A model's window holds a fixed number of them."],
  size: ["The next request", "A forecast, not a running total: what LLM-GC will send of this conversation with your next message - the recap, then the recent messages as written - compared with the whole conversation as written, which it would carry without LLM-GC. The system prompt, tools and your client's own instructions are not in either figure: they go with every request, word for word, either way. \"So far\" is measured: every request this conversation sent, counted whole."],
  holds: ["How each message is sent", "Every message the conversation has recorded, by how the model receives it now. A message is recorded when a later request carries it with its reply."],
  tools: ["Tool calls and results", "An agent such as Claude Code asks its client to run a tool – write a file, run a command – and the client sends back what came out, under the user's role. LLM-GC reads both as text and collects them with their exchange like any other message, never separating a call from its result."],
  instructions: ["Your client's instructions", "Text your client put inside this message for the model, not words of the conversation: Claude Code's <system-reminder> blocks carry your CLAUDE.md, memory and context. LLM-GC never condenses them. They go with every request word for word: inside this message while it is recent, and at the start of the recap once it is collected."],
  newest: ["Newest request", "What your client sent last, on top of the recorded conversation. It joins the conversation, and is counted, when a later request carries it with its reply. Clients also send side requests that never join: after each reply, Claude Code asks the model to suggest your next prompt, and that request shows here until the next one arrives."],
  reply: ["Newest reply", "The answer to the newest request, as it reached your client. It joins the conversation, and is counted, when your next request carries it; until then nothing is sent with it. A reply that is regenerated or never finishes does not join, so it is shown here but never recorded from here."],
  window: ["The model's window", "The most the model can read in one request, for the model this conversation's requests name, as the provider reports it; until it does, or if it cannot, the configured window. Filling it is what forces old context out; LLM-GC decides what goes, instead of the provider truncating. The recap's budget is a share of it."],
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
  tracking: ["Conversation tracking", "LLM-GC recognises a conversation by its content, so a history that continues none in progress starts fresh: a new conversation, a subagent's, one after /compact or an edited message, and any after a proxy restart. A client that names its conversation with the x-llm-gc-session header has an edited message rewound instead, or restarted when it cannot be. A restart or a fresh start costs one request that is not condensed."],
  billed: ["What the provider reported", "The usage the provider reported on each reply, added up: what it bills, counted by its own tokenizer. Cache reads cost a fraction of input; cache writes cost more than input; output costs most. LLM-GC's own model calls are not in it: they are beside it."],
  forwarded: ["Total sent since the proxy started", "Every request added up, whole - system prompt, tools and messages - across all conversations, since the proxy last started; a restart sets it back to zero. Each request carries the whole conversation again, so this grows much faster than any conversation. Counted with LLM-GC's own tokenizer, the same way on both sides; what the provider billed, by its own count, is beside it."],
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
  bypassed_not_collected:   { tone: "", icon: "–", text: "not collected: recorded and sent as written" },
};
// Why a conversation is recorded but never collected: today, only a subagent's.
export const NOT_COLLECTED_TEXT = "LLM-GC records this conversation and shows it, but sends it as written: it is a subagent's, which usually ends before a collection could pay for itself, and whose answer is only as good as what it read. The collect_subagents setting turns collection on.";
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
// What each status on a fact means: shown when it is hovered, and listed in "How fact collection works".
// Each label is the outcome for the model, so it reads without the guide.
export const FACT_STATUS = {
  line: ["✓", "good", "sent as a fact", "Its exchange is summarised or cut to its opening line, and the recap lists this fact beside it, so the model is told the fact itself, not only a summary that might leave it out."],
  words: ["✓", "good", "sent word for word", "Its exchange is kept word for word in the recap, so the model reads the fact in the words of the message that stated it."],
  unsent: ["⚠", "warn", "not sent", "Its exchange is out of view: the recap says only that the exchange exists, so the model can't see this fact."],
  noted: ["✓", "good", "sent marked replaced", "The recap marks the old value as replaced, beside the exchange that said it, so the model reads it as no longer current."],
  oldUnsent: ["", "", "old value not sent", "The exchange that said the old value is out of view, so neither the old value nor the note about it is sent."],
};
// What fell short in a collection, and what the model gets instead.
export const NOT_MOVED = {
  compaction_refused: ["Summary rejected by the safeguard", "Its exchange has no summary, so it is kept word for word or archived instead. Nothing is lost."],
  compaction_failed: ["Summary request failed", "Its exchange has no summary, so it is kept word for word or archived instead. Nothing is lost."],
  archive_failed: ["Facts couldn't be read", "Its exchange has no facts: shown compacted or archived, the model reads it without them."],
  recap_over_budget: ["Recap over its budget", "Every exchange is folded out of view and the recap still doesn't fit: the budget is too small for this conversation."],
};
// Said where a refused credential shows: on both tabs, so it is not found only by opening a card.
export const CREDENTIAL_REFUSED_TITLE = "The upstream is refusing LLM-GC's credential.";
export const CREDENTIAL_REFUSED_TEXT = "LLM-GC can't make summaries, so collections are being skipped and the prompt has stopped getting smaller. What was already condensed stays. Check the credential LLM-GC was started with, then restart it.";
// How much of a message the model still has, drawn as the height of its bar (percent of full).
export const HEIGHT = { verbatim: 100, kept: 100, compacted: 52, archived: 24, out_of_view: 10 };
// The ⓘ beside each card: the explanation its one-line subtitle leaves out.
export const CARD_INFO = {
  facts: ["What the model is told, fact by fact", "What the conversation settled, as topic: value notes, and how each one reaches the model. “How fact collection works”, below the facts, says how they are read and what each status means. Hover a status for its meaning."],
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
