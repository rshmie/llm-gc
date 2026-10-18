// The conversation view: what the model is told, fact by fact, exchange by exchange and message by message.
//
// Every value comes from an endpoint; this view formats and explains, and derives
// nothing beyond grouping, counting, and sums and differences of numbers it was given.

import { BECOMES, CATS, CLAMP_CHARS, CREDENTIAL_REFUSED_SHORT, CREDENTIAL_REFUSED_TITLE, FAR_JUMP_SCREENS, MAP_PX_PER_MESSAGE, EARLIER_SHOWN, GLOSSARY, HEIGHT, MIN_BAR_PX, NOT_COLLECTED_TEXT, ORDER, RUN_MIN, FACT_STATUS, SWITCH_REASON, TOOL_ROUND_TEXT, categoryOf, describeOutcome, swatch, pauseNote } from "./vocabulary.js";
import { $, agoTs, compact, countEl, fill, fmt, h, meterBar, plural, ranges, reducedMotion, setChildren, share, storeGet, storeSet } from "./dom.js";
import { LIST_ORDER_KEY, collecting, countCats, credentialRefusal, factsByTurn, has, instructionTokens, keepLast, marked, newestReply, pendingTurns, preview, query, recapLimit, state, trafficOf, turnMatches, ui, writtenTokens } from "./state.js";
import { hideTooltip, hint, info, showTooltip, tipAction, tipQuote, hoverTip } from "./tooltip.js";
import { offerReturn, showTab } from "./shell.js";
import { openReference } from "./guide.js";
import { openCard, showWhy, whyText } from "./engine.js";

const smooth = () => (reducedMotion() ? "auto" : "smooth");
export const speakerName = (role) => role === "user" ? "You" : role ? role[0].toUpperCase() + role.slice(1) : "—";
const avatarText = (role) => role === "user" ? "You" : role === "assistant" ? "AI" : (role || "?").slice(0, 2).toUpperCase();
const avatar = (role) => h("div", { class: `av ${role === "user" ? "user" : "assistant"}`, "aria-hidden": "true" }, avatarText(role));
// A turn of tool results is the client reporting what its tools returned, a relayed turn is another
// agent's report, and a turn Claude Code wrote entirely is its own notice. All come under the user
// role, and none is the user's words. A turn the user wrote only part of is shared: the header names
// both writers, and each part Claude Code or a tool wrote is set apart, so the rest is the user's. A
// tool round the user interrupted is one: what they typed arrives beside the round's results.
const byClient = (turn) => Boolean(turn.parts) && turn.parts.every((part) => part.writer !== "speaker");
const shared = (turn) => Boolean(turn.parts) && !byClient(turn);
const speakerOf = (turn) => turn.relayed ? "Subagent report" : shared(turn) ? "You and Claude Code"
  : turn.tool_results ? "Tool results" : byClient(turn) ? "Claude Code" : speakerName(turn.role);
const avatarOf = (turn) => turn.relayed ? h("div", { class: "av assistant", "aria-hidden": "true" }, "Agent")
  : shared(turn) ? avatar(turn.role)
  : turn.tool_results ? h("div", { class: "av user", "aria-hidden": "true" }, "Tool")
  : byClient(turn) ? h("div", { class: "av user", "aria-hidden": "true" }, "CC") : avatar(turn.role);
const toolTip = () => hoverTip(GLOSSARY.tools[0], GLOSSARY.tools[1]);
const relayedTip = () => hoverTip(GLOSSARY.relayed[0], GLOSSARY.relayed[1]);
const clientTip = () => hoverTip(GLOSSARY.client[0], GLOSSARY.client[1]);
const sharedTip = () => hoverTip(GLOSSARY.shared[0], GLOSSARY.shared[1]);
// The speaker, explained whenever it is not simply you or the assistant.
const speakerTip = (turn) => turn.relayed ? relayedTip() : shared(turn) ? sharedTip()
  : turn.tool_results ? toolTip() : byClient(turn) ? clientTip() : {};
const who = (turn) => h("span", { class: "who", ...speakerTip(turn) }, speakerOf(turn));
// Beside the role the provider receives: who wrote a turn, when the role alone would credit it to the wrong side.
const writerTag = (turn) => speakerOf(turn) === speakerName(turn.role) ? null
  : h("span", { class: "tag", ...speakerTip(turn) }, speakerOf(turn));
const toolTag = (label) => h("span", { class: "tag", ...toolTip() }, label);
const callsTag = (turn) => turn.tool_calls ? toolTag("calls a tool") : null;
// Instructions the client puts inside the message: never condensed, so their size stands apart.
function instructionsTag(turn) {
  const tokens = instructionTokens(turn.turn_index);
  return tokens ? h("span", { class: "tag", ...hoverTip(GLOSSARY.instructions[0], GLOSSARY.instructions[1]) },
    `+ client instructions · ${plural(tokens, "token")}`) : null;
}
// Images and documents: sized by the provider, so their size stands apart from the words; sent only while recent.
function imagesTag(turn) {
  const tokens = turn.attachment_tokens ?? 0;
  const unsized = turn.unsized_attachments ?? 0;
  if (!tokens && !unsized) return null;
  const sent = turn.fidelity === "verbatim";
  const words = [tokens ? `+ images · ${plural(tokens, "token")}${sent ? "" : ", no longer sent"}` : null,
    unsized ? `${plural(unsized, "image")} not sized` : null].filter(Boolean).join(" · ");
  return h("span", { class: "tag", ...hoverTip(GLOSSARY.images[0], GLOSSARY.images[1]) }, words);
}
const changing = (k) => k.turns.filter((t) => t.becomes).length;
const capitalised = (text) => text[0].toUpperCase() + text.slice(1);
export const messagesLabel = (turns) => turns.length === 1 ? `message ${turns[0]}` : `messages ${ranges(turns)}`;
// A level as a tooltip names it: a badge in the level's colour, then what the model reads for it.
const levelBadge = (cat) => h("span", { class: `tip-badge f-${cat}` }, CATS[cat].label);
const levelLine = (cat) => h("div", { class: "tip-level" }, levelBadge(cat), ` ${capitalised(CATS[cat].desc)}.`);
// The colour of a level; out of view is drawn as a warning outline rather than a fill.
const levelColour = (cat) => cat === "out_of_view" ? "var(--warn)" : `var(--fid-${cat})`;

export function renderHero(k) {
  renderSummary(k);
  renderMap(k);
}

// ----------------------------------------------------------------- the three questions

// A warning can be dismissed, and comes back only when there is more of what it warns about.
const dismissedKey = () => `llmgc.dismissed.${state.selected}`;
function dismissed() {
  try { return JSON.parse(storeGet(dismissedKey()) || "{}"); } catch { return {}; }
}
function dismiss(kind, upTo) {
  storeSet(dismissedKey(), JSON.stringify({ ...dismissed(), [kind]: upTo }));
  renderSummary(state.knowledge);
}

// A current fact reaches the model on its exchange's line in the recap, or in its words when that exchange is quoted.
const reachesModel = (fact) => fact.sent || fact.span_level === "kept";
const MARKS = { good: "✓", warn: "!", critical: "✕" };
// One line of a column: a mark, the words, then a link to where it shows, and a way to dismiss it.
function sumLine(tone, words, action = null, dismissal = null) {
  return h("li", { class: `sum-line${tone ? ` tone-${tone}` : ""}` },
    h("span", { class: "sum-mark", "aria-hidden": "true" }, MARKS[tone] || "·"),
    h("span", { class: "sum-words" }, words,
      action ? [" ", h("button", { class: "link-btn", type: "button", onclick: action[1] }, `${action[0]} →`)] : null),
    dismissal ? h("button", { class: "x", type: "button", "aria-label": "Dismiss", title: "Hide until there is more of this",
      onclick: () => dismiss(...dismissal) }, "×") : null);
}
const toMessages = (filter) => () => { setFilter(filter); $("messages").scrollIntoView({ behavior: smooth(), block: "start" }); };
const toFacts = () => $("facts-title").scrollIntoView({ behavior: smooth(), block: "start" });
const toEngine = (card) => () => { offerReturn("conversation", window.scrollY); showTab("engine"); openCard(card, true); };

// What the model receives of the conversation, from most to least.
function remembers(k, counts, facts) {
  const exchanges = (level) => k.spans.filter((s) => s.level === level).length;
  const n = (count) => h("b", {}, fmt(count));
  const sent = facts.filter(reachesModel).length;
  return [
    sumLine("", [n(counts.verbatim), ` recent ${counts.verbatim === 1 ? "message" : "messages"}, exactly as written`]),
    counts.kept ? sumLine("", [n(counts.kept), ` ${counts.kept === 1 ? "message" : "messages"} quoted word for word in the recap`]) : null,
    counts.compacted ? sumLine("", [n(counts.compacted), ` ${counts.compacted === 1 ? "message" : "messages"} as ${plural(exchanges("compacted"), "summary", "summaries")}`]) : null,
    counts.archived ? sumLine("", [n(counts.archived), ` ${counts.archived === 1 ? "message" : "messages"} by ${exchanges("archived") === 1 ? "its exchange's" : "their exchanges'"} opening line and facts`]) : null,
    facts.length ? sumLine("", [n(sent), ` of ${plural(facts.length, "fact")} told to it, as facts or in their words`]) : null,
  ].filter(Boolean);
}

// What it no longer has word for word, and whether anything is gone from view.
function forgotten(k, counts, facts) {
  const condensed = counts.compacted + counts.archived + counts.out_of_view;
  if (!condensed) return [sumLine("good", counts.kept ? "Nothing: every message is sent or quoted word for word." : "Nothing yet: every message is sent as written.")];
  const unseen = facts.filter((f) => !reachesModel(f)).length;
  const replaced = k.facts.filter((f) => f.status === "superseded").length;
  const unread = k.spans.filter((s) => s.facts_unread).reduce((sum, s) => sum + s.turns.length, 0);
  return [
    sumLine("", [h("b", {}, fmt(condensed)), ` ${condensed === 1 ? "message" : "messages"} no longer word for word`]),
    counts.out_of_view
      ? sumLine("warn", [h("b", {}, fmt(counts.out_of_view)), " out of view: it knows they exist, not what was said"], ["Show", toMessages("out_of_view")])
      : sumLine("good", "Nothing out of view"),
    unseen ? sumLine("warn", [h("b", {}, fmt(unseen)), ` ${unseen === 1 ? "fact" : "facts"} it can't see`], ["Show", toFacts])
      : facts.length ? sumLine("good", "Every fact still reaches it") : null,
    unread ? sumLine("warn", [h("b", {}, fmt(unread)), ` ${unread === 1 ? "message's" : "messages'"} facts couldn't be read yet: it gets them quoted or summarised until the next collection reads them`], ["Show", toFacts]) : null,
    replaced ? sumLine("", [h("b", {}, fmt(replaced)), ` earlier ${replaced === 1 ? "value" : "values"} replaced, and marked so`]) : null,
    h("li", { class: "sum-note" }, "Every word is stored: a later collection shows it in more detail again if the conversation comes back to it."),
  ].filter(Boolean);
}

// Whether anything needs a look: what the model can't see, a collection that fell short, a filling window.
// The all-clear is said only when it is true, not when its warnings were dismissed.
function worries(k, counts, facts) {
  const hidden = dismissed();
  const unseenFacts = facts.filter((f) => !reachesModel(f)).length;
  const shortfalls = state.activity ? state.activity.recent_degradations : [];
  const latestShortfall = shortfalls.length ? shortfalls.at(-1).occurred_at : 0;
  const fill_ = windowUse(k);
  const staged = k.staged;
  const items = [];
  // First, and with no way to hide it: unlike the others, it does not pass until someone acts.
  // A refused key is the more specific of the two, and says to restart; it stands for the pause it causes.
  if (credentialRefusal()) items.push(sumLine("warn", [h("b", {}, CREDENTIAL_REFUSED_TITLE), ` ${CREDENTIAL_REFUSED_SHORT}`], ["See why", toEngine("notmoved")]));
  else if (state.pause?.lasting) {
    const pause = pauseNote(state.pause);
    items.push(sumLine(pause.tone, [h("b", {}, pause.title), ` ${pause.short}`], ["Why", toEngine("next")]));
  }
  if (counts.out_of_view > (hidden.out_of_view ?? 0)) items.push(sumLine("warn",
    [h("b", {}, `It can no longer see ${plural(counts.out_of_view, "message")}.`), ` The recap is over its budget of ${recapLimit()}, so the oldest are folded into one line.`],
    ["Show them", toMessages("out_of_view")], ["out_of_view", counts.out_of_view]));
  if (unseenFacts) items.push(sumLine("warn", [h("b", {}, `${plural(unseenFacts, "fact")} ${unseenFacts === 1 ? "doesn't" : "don't"} reach it.`), " Their exchanges are out of view."], ["Show", toFacts]));
  if (latestShortfall > (hidden.shortfalls ?? 0)) items.push(sumLine("warn",
    [h("b", {}, `${plural(shortfalls.length, "part")} of a collection fell short.`), " Each says what the model gets instead; nothing is lost."],
    ["See why", toEngine("notmoved")], ["shortfalls", latestShortfall]));
  const unsized = k.turns.filter((t) => t.fidelity === "verbatim").reduce((n, t) => n + (t.unsized_attachments ?? 0), 0);
  if (unsized) items.push(sumLine("",
    [`${plural(unsized, "image")} in recent messages ${unsized === 1 ? "counts" : "count"} as a placeholder: the provider gave no size, so LLM-GC's counts here are low by ${unsized === 1 ? "its" : "their"} size.`],
    ["Why", () => openReference("term-images")]));
  if (fill_ && fill_.share >= WINDOW_WARN) items.push(sumLine("warn", [h("b", {}, `The prompt fills ${Math.round(fill_.share * 100)}% of the model's window.`), ` ${fill_.how}`]));
  if (staged && staged.stale_tokens >= STALE_NOTE * staged.stale_bound_tokens && !staged.switches) items.push(sumLine("",
    [`The model still gets ${plural(staged.stale_tokens, "stale token")}, ${Math.round((100 * staged.stale_tokens) / staged.stale_bound_tokens)}% of their limit. At the limit, LLM-GC applies the prepared collection.`],
    ["Why", toEngine("next")]));
  if (state.pause && !state.pause.lasting) {
    const pause = pauseNote(state.pause);
    items.push(sumLine(pause.tone, [h("b", {}, pause.title), ` ${pause.short}`], ["Why", toEngine("next")]));
  }
  if (collecting()) items.push(sumLine("", "Preparing a collection now: this updates when it finishes."));
  const clear = !counts.out_of_view && !unseenFacts && !shortfalls.length && !credentialRefusal() && !state.pause?.lasting
    && !(fill_ && fill_.share >= WINDOW_WARN);
  if (clear) items.unshift(sumLine("good", [h("b", {}, "No."), facts.length ? " Every fact reaches the model," : "", " nothing is out of view, and no collection fell short."]));
  else if (!items.some((item) => /tone-(warn|critical)/.test(item.className))) items.unshift(sumLine("", "Nothing new since you dismissed what was here."));
  return items;
}
// A window this full is worth a look; stale weight this close to its bound is worth a mention.
const WINDOW_WARN = 0.8;
const STALE_NOTE = 0.8;

// How full the model's window is: by the provider's own count of the last request when there is one, since the
// window is in its tokens and that count includes the system prompt and tools; otherwise LLM-GC's estimate.
function windowUse(k) {
  const w = k.context_window;
  if (!w) return null;
  const c = trafficOf(state.selected);
  const last = c && c.traffic.last_reply;
  return last ? { share: last.provider_prompt_tokens / w, how: "By the provider's count of the last request, system prompt and tools included." }
    : { share: k.context_tokens / w, how: "By LLM-GC's estimate of the conversation alone." };
}

export function renderSummary(k) {
  if (!k) return;
  const counts = countCats(k);
  const facts = k.facts.filter((f) => f.status === "active");
  fill("sum-remembers", h("ul", { class: "sum-list" }, remembers(k, counts, facts)));
  fill("sum-forgotten", h("ul", { class: "sum-list" }, forgotten(k, counts, facts)));
  fill("sum-worry", h("ul", { class: "sum-list" }, worries(k, counts, facts)));
  fill("sum-foot", tokenLines(k));
  renderBar(k);
}

// The token figures, each named for whose count it is: LLM-GC's own tokenizer is an estimate, the provider's is what it bills.
function tokenLines(k) {
  // Both sides include the model's earlier thinking and every image: without LLM-GC, all of it would be sent.
  const written = writtenTokens(k) + (k.written_thinking_tokens ?? 0) + (k.written_attachment_tokens ?? 0);
  const less = written - k.context_tokens - (k.thinking_tokens ?? 0) - (k.attachment_tokens ?? 0);
  const c = trafficOf(state.selected);
  const t = c && c.traffic;
  const last = t && t.last_reply;
  const use = windowUse(k);
  const passes = state.activity ? state.activity.recent_passes : [];
  const lastCollection = passes.filter((p) => p.gc_status === "completed").at(-1);
  const saving = !k.collected ? "not collected" : !k.cuts ? "nothing collected yet" : less >= 0 ? `${share(less, written)}% less than as written` : "more than as written: the recap outgrew what it replaced";
  const ratio = last && last.tokens_forwarded ? last.provider_prompt_tokens / last.tokens_forwarded : null;
  const percent = (part) => part < 0.001 && part > 0 ? "<0.1" : (part * 100).toFixed(part < 0.1 ? 1 : 0);
  const windowMeter = use ? h("span", { class: "sum-window" }, ` · ${percent(use.share)}% of the model's window`,
    h("span", { class: "meter mini", role: "meter", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(use.share * 100)), "aria-label": "Share of the model's window" },
      meterBar("window-share", use.share * 100))) : null;
  return [
    h("p", { class: "sum-tokens" }, "Next request ", h("b", {}, `≈ ${fmt(k.context_tokens)}`), " tokens of conversation", info("size"),
      k.thinking_tokens ? [" + ", h("b", {}, `≈ ${fmt(k.thinking_tokens)}`), " of the model's earlier thinking", info("thinking")] : null,
      k.attachment_tokens ? [" + ", h("b", {}, fmt(k.attachment_tokens)), " of images", info("images")] : null,
      ` · ${saving}`,
      last ? null : windowMeter),
    last ? h("p", { class: "sum-tokens" }, "Last request ",
      hint(h("b", {}, fmt(last.provider_prompt_tokens)), "The last request, by the provider's count",
        `${compact(last.usage.cache_read_input_tokens)} read from its cache, ${compact(last.usage.cache_creation_input_tokens)} written to it and ${compact(last.usage.input_tokens)} sent plainly, the system prompt and tools included.`),
      " tokens by the provider's count", windowMeter,
      ratio ? [" · ", hint(`${ratio.toFixed(2)}× LLM-GC's count`, "Two counts of one request",
        `LLM-GC counted it at ${fmt(last.tokens_forwarded)} with its own tokenizer, its estimate of the model's earlier thinking and its images included.${k.token_rate ? ` LLM-GC sizes the recap's budget and the stale limit at ${k.token_rate.toFixed(2)}×, measured on its latest request that carried no thinking, image or document.` : ""}`)] : null) : null,
    h("p", { class: "sum-tokens secondary" },
      lastCollection ? [`Last collection ${agoTs(lastCollection.gc_completed_at)}${lastCollection.outcome ? `: ${describeOutcome(lastCollection.outcome)}` : ""} · `,
        h("button", { class: "link-btn", type: "button", onclick: toEngine("moves") }, "What moved →")] : "No collection yet",
      t ? ` · So far ${plural(t.requests, "request")}, ${t.tokens_saved ? `${share(t.tokens_saved, t.tokens_from_client)}% saved` : "nothing saved yet"}` : null),
  ];
}

// ----------------------------------------------------------------- the bar above every tab: the cache, and the next collection

const duration = (seconds) => seconds < 90 ? `${Math.round(seconds)} s` : seconds < 5400 ? `${Math.round(seconds / 60)} min` : `${+(seconds / 3600).toFixed(1)} h`;

// Each chip says one thing in a few words; hovering says the rest.
export function renderBar(k) {
  // A chip's tooltip says what it means right now, in a sentence; a clickable chip says where it leads.
  const chip = (tone, label, title, text, onclick = null) => h(onclick ? "button" : "span",
    { class: `bar-chip${tone ? ` tone-${tone}` : ""}`, ...(onclick ? { type: "button", onclick } : {}),
      ...hoverTip(title, text, onclick ? tipAction("Click for the Next collection card") : null) }, label);
  const cache = state.cache;
  const keeps = cache && cache.lifetime_s ? duration(cache.lifetime_s) : "";
  const cacheChip = !cache ? null
    : cache.uses_cache === null ? chip("", "Cache: no reply yet", "The provider's prompt cache",
      "No request has been answered yet, so nothing says whether the provider caches this conversation.")
    : cache.uses_cache === false ? chip("", "Cache: not used", "No prompt cache",
      "This conversation does not use the prompt cache. The provider reads every request in full, so a smaller prompt is always cheaper.")
    : cache.state === "warm" ? chip("good", `Cache warm · ${duration(Math.max(0, cache.lifetime_s - cache.idle_s))} left`, "Cache warm",
      `The provider still has this prompt cached, so the next request costs less if its start is unchanged. The cache expires ${keeps} after the last request.`)
    : chip("cold", `Cache cold · idle ${duration(cache.idle_s)}`, "Cache cold",
      `No request for longer than the ${keeps} the provider keeps a prompt cached. The provider reads the next request in full anyway, so applying a collection now costs nothing extra.`);
  const s = k && k.staged;
  const cut = k && k.next_cut;
  const next = !k ? null
    : !k.collected ? chip("", "Not collected", "Not collected", NOT_COLLECTED_TEXT)
    : collecting() ? chip("", "Preparing a collection…", "Preparing a collection", "LLM-GC is writing summaries and reading facts. What the model is sent does not change yet.", toEngine("next"))
    : s && s.switches ? chip("good", "Collection: applied with next request", "Applied with the next request", `${capitalised(SWITCH_REASON[s.reason] || s.reason)}.`, toEngine("next"))
    : s && s.reason === "tool_round_open" ? chip("", "Collection: prepared, waiting for tool calls", "Waiting for the model to finish using tools", TOOL_ROUND_TEXT, toEngine("next"))
    : s ? chip("", "Collection: prepared, waiting", "Prepared, waiting for a cheaper moment",
      `The cache is warm. Applying it now would make the provider cache ${fmt(s.rewritten_tokens)} tokens again, to save ${fmt(s.saving_tokens)} tokens on each later request.`, toEngine("next"))
    : cut.batch_ready ? chip("", "Collection: prepared after the next reply", "Ready to prepare", "Enough older messages are waiting. LLM-GC prepares a collection after the next reply.", toEngine("next"))
    : cut.compactable_tokens ? chip("", `Collection: not enough yet · ${fmt(cut.compactable_tokens)} of ${fmt(cut.min_cut_tokens)} tokens`, "Not enough yet",
      `LLM-GC prepares a collection once the older messages to condense reach ${fmt(cut.min_cut_tokens)} tokens.`, toEngine("next"))
    : chip("", "Collection: nothing to collect yet", "Nothing to collect yet", `No message is older than the newest ${keepLast()} yet. The newest messages always stay as written.`, toEngine("next"));
  fill("bar-chips", cacheChip, next);
}

// ----------------------------------------------------------------- the map: what the model has of each message

// One bar per exchange the recap holds - the model reads one entry for each - and one per recent message,
// reused across renders and keyed by what it stands for: a re-planned exchange changes its colour and height
// in place, which is what lets the change transition instead of jumping. `barOfTurn` finds a message's bar.
const barCache = new Map();
let barOfTurn = new Map();
let lastCats = new Map();
let ledgerSegs = null;
export function clearMapCache() { barCache.clear(); barOfTurn = new Map(); lastCats = new Map(); ledgerSegs = null; }

// Appends what is new instead of re-inserting what is already there: an element taken out of the
// page and put back loses its running transition.
function reconcile(host, nodes) {
  const current = [...host.children];
  if (current.length <= nodes.length && current.every((node, i) => node === nodes[i])) {
    for (const node of nodes.slice(current.length)) host.appendChild(node);
  } else setChildren(host, nodes);
}

function ledger() {
  if (!ledgerSegs) {
    ledgerSegs = Object.fromEntries(["block", "saved", "recent", "pending"].map((key) => [key, h("div", { class: `an-s ${key}` }, h("i"))]));
    setChildren($("an-ledger"), Object.values(ledgerSegs));
  }
  return ledgerSegs;
}

// The smallest size, in tokens, a bar is drawn at so that none is narrower than MIN_BAR_PX on a track this
// wide. Raising the floor widens the small bars and so narrows the rest a little, which raises the floor
// needed: the loop settles within a few passes, because the track always holds every message at
// MAP_PX_PER_MESSAGE, which is more than MIN_BAR_PX. On a track too narrow for that (a hidden tab) it
// does not settle and the bars come out equal, which is the honest limit.
function floorTokens(sizes, total, trackPx) {
  let floor = (total * MIN_BAR_PX) / trackPx;
  for (let pass = 0; pass < 24; pass++) {
    const drawn = sizes.reduce((n, size) => n + Math.max(size ?? floor, floor), 0);
    floor = (drawn * MIN_BAR_PX) / trackPx;
  }
  return floor;
}

// Fades the edge the map scrolls on, so bars out of view don't read as the start or end of the conversation.
function markMoreSideways(el) {
  const left = el.scrollLeft > 1;
  const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
  if (left || right) el.dataset.more = left && right ? "both" : left ? "left" : "right";
  else delete el.dataset.more;
}

export function renderMap(k) {
  const facts = factsByTurn(k);
  const byIndex = new Map(k.turns.map((t) => [t.turn_index, t]));
  const items = [
    ...k.spans.map((span) => ({ span, key: `s${span.turns[0]}`, i: span.turns[0], turns: span.turns, cat: span.level,
      tok: span.turns.reduce((n, i) => n + (byIndex.get(i)?.tokens ?? 0), 0), text: span.text })),
    ...k.turns.filter((t) => t.fidelity === "verbatim").map((t) => ({ turn: t, key: `t${t.turn_index}`, i: t.turn_index, turns: [t.turn_index],
      cat: categoryOf(t), tok: t.tokens, role: t.role, text: t.text })),
    ...pendingTurns(k).map((p) => ({ pending: true, key: `p${p.turn_index}`, i: p.turn_index, turns: [p.turn_index], cat: "pending", tok: p.tokens, role: p.role, text: p.text })),
  ];
  const messages = items.reduce((n, x) => n + x.turns.length, 0);
  const barsHost = $("an-bars");
  const scroller = $("map");
  // Read before anything changes: a map the reader had scrolled to its newest end follows new messages.
  const atNewest = scroller.scrollLeft + scroller.clientWidth >= scroller.scrollWidth - 24;
  // Room for every message, however many: past the card's width the map scrolls sideways as one.
  scroller.style.setProperty("--track", `${items.length * MAP_PX_PER_MESSAGE}px`);
  if (!scroller.dataset.wired) {
    scroller.dataset.wired = "1";
    scroller.addEventListener("scroll", () => markMoreSideways(scroller), { passive: true });
  }
  // Widths are tokens, with a floor so a one-token message stays at least MIN_BAR_PX wide and clickable.
  const total = items.reduce((n, x) => n + (x.tok ?? 0), 0) || items.length || 1;
  const floor = floorTokens(items.map((x) => x.tok), total, Math.max(240, barsHost.clientWidth || 900));
  const width = (x) => Math.max(x.tok ?? floor, floor);
  barsHost.classList.toggle("dense", items.length > 80);
  barsHost.classList.toggle("searching", !!query());
  $("map").setAttribute("aria-label", `Every message in order, oldest first: ${messages} in all, the collected ones as their exchanges`);

  const groupOf = (x) => x.pending ? "newest" : x.cat !== "verbatim" ? "condensed" : x.turn.becomes ? "next" : "tail";
  const runs = [];
  for (const x of items) {
    const g = groupOf(x), r = runs.at(-1);
    if (r && r.g === g) { r.n += x.turns.length; r.w += width(x); } else runs.push({ g, n: x.turns.length, w: width(x) });
  }
  const GROUP = { condensed: "Collected", next: "Next", tail: "Recent", newest: "" };
  setChildren($("an-labels"), runs.map((r) => h("div", { class: "an-l", style: `flex-grow:${r.w}` },
    h("span", { title: r.g === "newest" ? "Newest request" : `${GROUP[r.g]} · ${plural(r.n, "message")}` },
      r.g === "newest" ? " " : [h("b", {}, GROUP[r.g]), ` · ${r.n}`]))));

  let changed = 0;
  barOfTurn = new Map();
  const cells = items.map((x) => {
    let el = barCache.get(x.key);
    if (!el) {
      el = h("div", { class: "bar", tabindex: "0", role: "button",
        onclick: () => jumpTo(el._i),
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); jumpTo(el._i); } },
        onpointerenter: (e) => { showTooltip(e, ...el._tip); trace(el._turns); },
        onpointerleave: () => { hideTooltip(); trace(null); },
        onfocus: (e) => { showTooltip(e, ...el._tip); trace(el._turns); },
        onblur: () => { hideTooltip(); trace(null); } });
      el._cell = h("div", { class: "an-c" }, el);
      barCache.set(x.key, el);
    }
    el._i = x.i;
    el._turns = x.turns;
    for (const i of x.turns) barOfTurn.set(i, el);
    el._cell.style.flexGrow = String(width(x));
    const t = x.turn;
    const moving = x.span || t;
    const hit = query() ? (x.pending ? has(x.text) : x.span ? has(x.text) || x.turns.some((i) => turnMatches(byIndex.get(i), facts)) : turnMatches(t, facts)) : null;
    el.className = `bar${x.pending ? " pending" : ` f-${x.cat}`}${hit === null ? "" : hit ? " hit" : " miss"}`;
    const before = lastCats.get(x.key);
    el.style.setProperty("--d", before !== undefined && before !== x.cat ? `${(changed++ * 0.03).toFixed(2)}s` : "0s");
    lastCats.set(x.key, x.cat);
    el.style.setProperty("--h", `${x.pending ? 100 : HEIGHT[x.cat]}%`);
    setChildren(el, moving && moving.becomes ? h("span", { class: "q", style: `--qb:${levelColour(moving.becomes)}` }) : null);
    const size = plural(x.tok, "token");
    const name = x.span ? `${capitalised(messagesLabel(x.turns))}, one exchange` : `Message ${x.i}`;
    el.setAttribute("aria-label", x.pending ? `Message ${x.i}, newest request, ${size}: not in the conversation yet`
      : `${name}, ${size} as written: ${CATS[x.cat].label}`);
    el._tip = x.pending
      ? [`Message ${x.i} · newest request · ${size}`, GLOSSARY.newest[1], tipQuote(preview(x.text))]
      : x.span
        ? [[swatch(x.cat), `${capitalised(messagesLabel(x.turns))} · ${size} as written`], levelLine(x.cat),
          x.span.becomes ? h("div", { class: "tip-level" }, "Next collection: ", levelBadge(x.span.becomes), ` ${whyText(x.span)}`) : null,
          h("div", {}, "The model reads:"), tipQuote(preview(x.text)), tipAction("Click to open")]
        : [[swatch(x.cat), `Message ${x.i} · ${speakerOf(t)} · ${size}`], levelLine(x.cat),
          t.becomes ? h("div", { class: "tip-level" }, "Next collection: ", levelBadge(t.becomes), ` ${whyText(t)}`) : null,
          tipQuote(preview(x.text)), tipAction("Click to open")];
    return el._cell;
  });
  reconcile(barsHost, cells);

  // The Sent row: the recap and the saving split the width of the condensed messages
  // above them, and recent messages sit under themselves.
  const segs = ledger();
  const cond = items.filter((x) => !x.pending && x.cat !== "verbatim");
  const verb = items.filter((x) => !x.pending && x.cat === "verbatim");
  const pend = items.filter((x) => x.pending);
  const sum = (list) => list.reduce((n, x) => n + width(x), 0);
  const saved = writtenTokens(k) - k.context_tokens;
  const thinkingSaved = Math.max(0, (k.written_thinking_tokens ?? 0) - (k.thinking_tokens ?? 0));
  const imagesSaved = Math.max(0, (k.written_attachment_tokens ?? 0) - (k.attachment_tokens ?? 0));
  const besides = [thinkingSaved ? `≈${fmt(thinkingSaved)} tokens of the model's earlier thinking (an estimate)` : null,
    imagesSaved ? `${fmt(imagesSaved)} tokens of images` : null].filter(Boolean);
  const condW = sum(cond);
  const blockW = condW && saved > 0 ? (condW * k.front_tokens) / (k.front_tokens + saved) : condW;
  const recent = k.context_tokens - k.front_tokens;
  const pendTok = pend.reduce((n, x) => n + (x.tok ?? 0), 0);
  const parts = {
    block: [blockW, `The recap · ${plural(k.front_tokens, "token")}`, "Stands in for the collected messages above it.", cond],
    saved: [condW - blockW, `Saved · ${plural(Math.max(0, saved), "token")}`, besides.length
      ? `What collections took out of every request, besides ${besides.join(" and ")}.`
      : "What collections took out of every request.", cond],
    recent: [sum(verb), `Recent messages · ${plural(recent, "token")}`, "Not collected yet: each sent as its own message, as written.", verb],
    pending: [sum(pend), `Newest request · ${plural(pendTok, "token")}`, "Its own message, sent on top of the recorded ones; not in the conversation yet.", pend],
  };
  for (const [key, [grow, title, text, list]] of Object.entries(parts)) {
    const seg = segs[key];
    seg.style.flexGrow = String(grow);
    seg.classList.toggle("zero", !grow);
    seg.onpointerenter = (e) => { showTooltip(e, title, text); trace(list.map((x) => x.i), seg); };
    seg.onpointerleave = () => { hideTooltip(); trace(null); };
  }
  renderKey(k, items, { saved, recent, pendTok });

  // A new conversation opens at its newest end, where the live action is; after that the reader's place is
  // kept, unless they were already at the newest end, so new messages stay in view.
  if (ui.mapSession !== state.selected || atNewest) scroller.scrollLeft = scroller.scrollWidth;
  ui.mapSession = state.selected;
  markMoreSideways(scroller);
  $("an-hint").textContent = scroller.scrollWidth > scroller.clientWidth + 1 ? " · scrolls sideways" : "";
}

// The key mirrors the map's two rows: what a bar means, then what the Sent row is made of.
function renderKey(k, items, { saved, recent, pendTok }) {
  const counts = countCats(k);
  const queued = k.turns.some((t) => t.becomes);
  const pending = items.some((x) => x.pending);
  const item = (mark, ...text) => h("span", {}, mark, h("span", {}, ...text));
  const explained = (key, mark, ...text) => h("span", hoverTip(GLOSSARY[key][0], GLOSSARY[key][1]), mark, h("span", {}, ...text));
  fill("an-key",
    h("div", { class: "an-rowlab" }, "Bars"),
    h("div", { class: "items" },
      ORDER.filter((key) => counts[key]).map((key) => explained(key, h("i", { class: `lg f-${key}`, style: `height:${Math.max(3, Math.round(HEIGHT[key] * 0.14))}px` }), CATS[key].label)),
      queued ? explained("next", h("i", { class: "qdot" }), "Dot: changes at the next collection, in the colour it will become") : null,
      pending ? item(h("i", { class: "key pending" }), "Outline: the newest request, not in the conversation yet") : null),
    h("div", { class: "an-rowlab" }, "Sent"),
    h("div", { class: "items" },
      k.front_tokens ? item(h("i", { class: "key block" }), "Recap ", h("b", {}, fmt(k.front_tokens))) : null,
      saved > 0 && k.front_tokens ? item(h("i", { class: "key saved" }), "Saved ", h("b", {}, fmt(saved))) : null,
      saved < 0 ? item(null, "The recap is ", h("b", {}, fmt(-saved)), " tokens larger than what it replaced") : null,
      item(h("i", { class: "key recent" }), "Recent messages ", h("b", {}, fmt(recent))),
      pending ? item(h("i", { class: "key pending" }), "Newest request ", h("b", {}, fmt(pendTok))) : null));
}

// Hovering a message anywhere traces it: its bar, its Sent segment, its row, and its score.
export function trace(ids, seg = null) {
  for (const el of document.querySelectorAll(".linked")) el.classList.remove("linked");
  $("an-bars").classList.toggle("tracing", !!ids);
  $("an-ledger").classList.toggle("tracing", !!ids);
  if (!ids) return;
  const k = state.knowledge;
  const segs = ledger();
  // Exchange rows and score rows stand for several messages; they carry them in data-turns.
  for (const row of document.querySelectorAll("#timeline [data-turns], #relevance [data-turns]")) {
    if (row.dataset.turns.split(",").map(Number).some((n) => ids.includes(n))) row.classList.add("linked");
  }
  for (const i of ids) {
    barOfTurn.get(i)?.classList.add("linked");
    const t = k && k.turns.find((x) => x.turn_index === i);
    const key = !t ? "pending" : t.fidelity !== "verbatim" ? "block" : "recent";
    segs[key].classList.add("linked");
    if (key === "block") segs.saved.classList.add("linked");
  }
  if (seg) seg.classList.add("linked");
}

// After a collection the collected bars shrink first, then the block slides in beneath them; a brief
// glow marks where they went.
export function celebrateCleanup() {
  if (reducedMotion() || !ledgerSegs) return;
  const block = ledgerSegs.block;
  setTimeout(() => { block.classList.add("landed"); setTimeout(() => block.classList.remove("landed"), 1100); }, 900);
}

// ----------------------------------------------------------------- exchange by exchange

export function setFilter(key) {
  ui.filter = key;
  openCard("messages");
  if (ui.promptView) setPromptView(false);
  renderMessages();
}

export function renderMessages() {
  const k = state.knowledge;
  if (!k) return;
  renderChips(k);
  if (ui.promptView) renderPrompt(k); else renderTimeline(k);
  renderFacts(k);
  renderMap(k);
}

export function renderChips(k) {
  const counts = countCats(k);
  const pending = pendingTurns(k).length + (newestReply() ? 1 : 0);
  const nextCount = changing(k);
  const chip = (key, label, count, mark) => h("button", { class: "chip-btn", type: "button", "aria-pressed": String(ui.filter === key),
    ...(GLOSSARY[key] ? hoverTip(GLOSSARY[key][0], GLOSSARY[key][1]) : {}),
    onclick: () => setFilter(ui.filter === key && key !== "all" ? "all" : key) }, mark, label, h("span", { class: "count" }, countEl(`chip-${key}`, count)));
  fill("chips", chip("all", "All", k.turns.length + pending),
    ORDER.filter((key) => counts[key]).map((key) => chip(key, CATS[key].label, counts[key], swatch(key))),
    nextCount ? chip("next", "Changes next", nextCount, h("span", { class: "qdot" })) : null);
}

// One row per exchange the old generation holds, one per run of folded exchanges (they share a
// line in the recap), then one per recent message, then the newest request.
export function buildRows(k) {
  const byIndex = new Map(k.turns.map((t) => [t.turn_index, t]));
  const rows = [];
  for (const span of k.spans) {
    const turns = span.turns.map((n) => byIndex.get(n));
    const last = rows.at(-1);
    if (span.level === "out_of_view" && last && last.kind === "folded") { last.spans.push(span); last.turns.push(...turns); }
    else if (span.level === "out_of_view") rows.push({ kind: "folded", cat: "out_of_view", spans: [span], turns });
    else rows.push({ kind: "span", cat: span.level, span, turns });
  }
  for (const turn of k.turns) if (turn.fidelity === "verbatim") rows.push({ kind: "turn", cat: "verbatim", turn, turns: [turn] });
  for (const pending of pendingTurns(k)) rows.push({ kind: "pending", cat: "verbatim", pending, turns: [] });
  const reply = newestReply();
  if (reply) rows.push({ kind: "reply", cat: "verbatim", pending: reply, turns: [] });
  return rows;
}

export function rowVisible(row, facts) {
  const f = ui.filter;
  // The newest message and its reply are not recorded yet, so they belong to no category but "all".
  const unrecorded = row.kind === "pending" || row.kind === "reply";
  const passes = f === "all" || (f === "next" ? row.turns.some((t) => t.becomes) : !unrecorded && row.cat === f);
  if (!passes) return false;
  if (!query()) return true;
  if (unrecorded) return has(row.pending.text);
  const line = row.kind === "span" ? row.span.text : row.kind === "folded" ? row.spans[0].text : null;
  return has(line) || row.turns.some((t) => turnMatches(t, facts));
}

// Long stretches of archived exchanges fold into one row, so the page leads with what the model
// reads in detail. Filtering or searching shows every match.
export function foldRuns(rows) {
  if (ui.filter !== "all" || query()) return rows;
  const out = [];
  let run = [];
  const flush = () => { if (run.length >= RUN_MIN) out.push({ kind: "run", rows: run, turns: run.flatMap((r) => r.turns) }); else out.push(...run); run = []; };
  for (const row of rows) {
    if (row.kind === "span" && row.cat === "archived") run.push(row);
    else { flush(); out.push(row); }
  }
  flush();
  return out;
}

// How many original messages a row stands for.
export const rowMessageCount = (row) => row.kind === "pending" || row.kind === "reply" ? 1 : row.turns.length;

const isQueued = (row) => row.kind === "turn" && !!row.turn.becomes;
const RECAP_ROWS_SHOWN = 3;   // recap rows shown before "Show N older messages"
const isCondensed = (row) => row.kind === "run" || row.kind === "span" || row.kind === "folded";

// What each view says about its order. Both views honour the one switch; each says how its order relates to the model's.
const ORDER_NOTE = {
  timeline: {
    newest: "Newest first. The model reads it the other way round, the recap before the recent messages. For each exchange: what the model reads, and what was said.",
    oldest: "Oldest first, as the model reads it: for each exchange, what the model reads and what was said.",
  },
  prompt: {
    newest: "Newest first. The model reads it the other way round, the recap before the recent messages. Copy takes it in the model's order.",
    oldest: "Oldest first, as the model reads it: the recap, then the recent messages.",
  },
};

// The control says the order now showing, and the subtitle says how that order relates to the model's.
export function renderOrderControl() {
  const now = ui.newestFirst ? "Newest first" : "Oldest first";
  const other = ui.newestFirst ? "oldest first" : "newest first";
  setChildren($("order-label"), now);
  $("order").setAttribute("aria-label", `${now}. Switch to ${other}`);
  $("order").title = `Switch to ${other}`;
  const notes = ui.promptView ? ORDER_NOTE.prompt : ORDER_NOTE.timeline;
  setChildren($("messages-sub"), ui.newestFirst ? notes.newest : notes.oldest);
}

export function setOrder(newestFirst) {
  ui.newestFirst = newestFirst;
  storeSet(LIST_ORDER_KEY, newestFirst ? "newest" : "oldest");
  renderOrderControl();
  if (state.knowledge) { if (ui.promptView) renderPrompt(state.knowledge); else renderTimeline(state.knowledge); }
  // Turning the list over leaves a reader who had scrolled into it with no place to be: start them at its head.
  if ($("messages").getBoundingClientRect().top < 0) $("messages").scrollIntoView({ behavior: "auto", block: "start" });
}

// The list is rebuilt on every change, and newest first grows at its head, so a redraw would push down
// whatever a reader had scrolled to. Put the first row in view back where it was. A row a collection has
// folded away is gone, and the reader is left where the page puts them. `rows` selects the rows of
// `list` that carry an id to find again after the redraw.
function keepPlace(list, rows, redraw) {
  if (!ui.newestFirst || list.getBoundingClientRect().top >= 0) return redraw();
  const anchor = [...list.querySelectorAll(rows)].find((row) => row.getBoundingClientRect().bottom > 0);
  const before = anchor?.getBoundingClientRect().top;
  redraw();
  const same = anchor && $(anchor.id);
  if (same) window.scrollBy({ top: same.getBoundingClientRect().top - before, behavior: "instant" });
}

export function renderTimeline(k) {
  const facts = factsByTurn(k);
  const visible = buildRows(k).filter((row) => rowVisible(row, facts));
  // Folding works on the rows oldest first, as the model reads the prompt: the recap, then the recent messages.
  // Only the order they are laid out in follows the viewer's choice.
  const folded = foldRuns(visible);
  const filtered = ui.filter !== "all" || query();
  // Unfiltered, a long conversation shows its newest rows and the last few of the recap; the earlier ones
  // sit behind a button at the far end from the newest, as a chat loads older messages. Filtering, searching
  // or jumping to a message shows everything that matches.
  const lastCondensed = folded.findLastIndex(isCondensed);
  const keep = Math.max(EARLIER_SHOWN, lastCondensed === -1 ? 0 : folded.length - lastCondensed - 1 + RECAP_ROWS_SHOWN);
  const canFold = !filtered && folded.length - keep >= RUN_MIN;
  const foldEarlier = canFold && !ui.earlierOpen;
  const shown = foldEarlier ? folded.slice(folded.length - keep) : folded;
  // Each part of the prompt is labelled where it begins, in the order the model reads it.
  const queued = k.turns.filter((t) => t.fidelity === "verbatim" && t.becomes).length;
  const items = [];
  let part = null;
  for (const row of ui.newestFirst ? shown.toReversed() : shown) {
    const now = isCondensed(row) ? "recap" : row.kind === "pending" || row.kind === "reply" ? "newest" : "recent";
    if (!filtered && now !== part) {
      items.push(h("div", { class: "boundary" }, h("span", {},
        now === "recap" ? ["In the recap · ", h("b", {}, fmt(k.front_tokens)), " tokens, standing in for every older exchange"]
          : now === "recent" ? ["Recent, sent as written", queued ? ` · the next collection takes the oldest ${queued === 1 ? "message" : plural(queued, "message")}` : ""]
            : "Not in the conversation yet")));
      part = now;
    }
    items.push(rowElement(row, facts));
  }
  const timeline = $("timeline");
  renderSearchVerdict(k);
  const note = $("filter-note");
  note.hidden = !filtered;
  if (filtered) {
    const shownCount = visible.reduce((n, row) => n + rowMessageCount(row), 0);
    setChildren(note, `Showing ${plural(shownCount, "message")}${query() ? ` matching “${ui.query.trim()}”` : ""}${ui.filter !== "all" ? ` · ${ui.filter === "next" ? "changing at the next collection" : CATS[ui.filter].tag}` : ""}`,
      h("button", { class: "link-btn", type: "button", onclick: () => { ui.query = ""; $("search").value = ""; setFilter("all"); } }, "Show all"));
  }
  const earlierCount = folded.slice(0, folded.length - keep).reduce((n, row) => n + rowMessageCount(row), 0);
  const setEarlier = (open) => {
    ui.earlierOpen = open;
    renderTimeline(state.knowledge);
    $("earlier-toggle")?.scrollIntoView({ block: "nearest" });
  };
  const earlierBtn = canFold ? h("div", { class: `earlier-row${ui.newestFirst ? " last" : ""}` },
    h("button", { class: "link-btn", id: "earlier-toggle", type: "button", "aria-expanded": String(!foldEarlier), onclick: () => setEarlier(foldEarlier) },
      foldEarlier ? `Show ${plural(earlierCount, "earlier message")}` : "Show fewer messages")) : null;
  const rows = items.length ? items : h("p", { class: "empty", style: "padding:14px 0" },
    filtered ? "No message matches." : "No messages recorded yet.");
  keepPlace(timeline, "article.msg[id]", () => setChildren(timeline, ui.newestFirst ? [rows, earlierBtn] : [earlierBtn, rows]));
  for (const row of timeline.querySelectorAll("[data-turns]")) {
    const ids = row.dataset.turns.split(",").map(Number);
    row.addEventListener("pointerenter", () => trace(ids));
    row.addEventListener("pointerleave", () => trace(null));
  }
}

function rowElement(row, facts) {
  return row.kind === "run" ? runRow(row) : row.kind === "span" ? spanRow(row) : row.kind === "folded" ? foldedRow(row, facts)
    : row.kind === "pending" ? pendingRow(row.pending) : row.kind === "reply" ? replyRow(row.pending) : turnRow(row.turn);
}

export function textBlock(text, { faded = false, key = null } = {}) {
  if (!text) return h("div", { class: "lost" }, "no text: thinking, or blocks with no text form");
  const long = text.length > CLAMP_CHARS || text.split("\n").length > 6 || (faded && text.length > 160);
  const open = !long || (key && ui.open.has(key)) || (query() && has(text));
  if (!long) return h("div", { class: `text${faded ? " faded" : ""}` }, marked(text));
  const toggle = () => {
    const nowOpen = !ui.open.has(key);
    if (nowOpen) ui.open.add(key); else ui.open.delete(key);
    body.classList.toggle("clamped", !nowOpen);
    more.textContent = nowOpen ? "Show less" : "Show more";
  };
  const body = h("div", { class: `text clampable${faded ? " faded" : ""}${open ? "" : " clamped"}`, onclick: toggle }, marked(text));
  const more = h("button", { class: "more", type: "button", onclick: toggle }, open ? "Show less" : "Show more");
  return [body, more];
}

// A message's words. Where Claude Code or a tool wrote part of the message, that part is set apart and named.
const partNames = { client: () => [GLOSSARY.client[0], clientTip()], tool: () => ["Tool results", toolTip()] };
function turnText(turn, { faded = false } = {}) {
  const key = `t-${turn.turn_index}`;
  if (!turn.parts || byClient(turn)) return textBlock(turn.text, { faded, key });
  return turn.parts.map((part, i) => {
    const words = textBlock(part.text, { faded, key: `${key}-${i}` });
    if (part.writer === "speaker") return words;
    const [name, tip] = partNames[part.writer]();
    return h("div", { class: "client-part" }, h("div", { class: "client-label" }, h("span", { class: "term", ...tip }, name)), words);
  });
}

export function fidTag(cat) {
  return h("span", { class: "tag", ...hoverTip(CATS[cat].label, GLOSSARY[cat][1]) }, swatch(cat), CATS[cat].tag);
}

// A status on a fact, with its meaning on hover; the dotted underline says there is one.
function statusTag(key) {
  const [glyph, tone, label, meaning] = FACT_STATUS[key];
  return h("span", { class: "fact-status", ...hoverTip(glyph ? `${glyph} ${label}` : label, meaning) },
    glyph ? h("span", { class: `st ${tone}` }, `${glyph} `) : null, h("span", { class: "term" }, label));
}
// How a current fact reaches the model: listed beside its exchange, in its exchange's words, or not at all.
const factStatus = (fact) => statusTag(fact.sent ? "line" : fact.span_level === "kept" ? "words" : "unsent");
// Who stated a fact: the user's decisions are theirs to change, an assistant's value is a recommendation.
const WHO = { user: "you", assistant: "AI" };
const said = (turn, speaker) => `message ${turn}${WHO[speaker] ? ` · ${WHO[speaker]}` : ""}`;
// A fact read from a statement that names no topic, like "we decided to ship on Friday".
const topicOf = (fact) => fact.topic === "general" ? "no topic" : fact.topic;

function factChip(fact) {
  return h("span", { class: `fact-chip s-${fact.status}` }, h("b", {}, topicOf(fact)),
    h("span", { class: "fact-text" }, marked(fact.content)), factStatus(fact));
}

// "next: archived" explains itself: hover for the reason, click for its score on the Engine tab.
// `item` is a recent message, or an exchange already collected.
export function nextTag(item, turnIndex) {
  return h("button", { class: "tag next why", type: "button", "aria-label": `Next collection: ${BECOMES[item.becomes]}. Show why on the Engine tab`,
    onpointerenter: (e) => showTooltip(e, `Why ${BECOMES[item.becomes]}?`, whyText(item), tipAction("Click to see its score on the Engine tab")),
    onpointerleave: hideTooltip, onclick: () => { hideTooltip(); showWhy(turnIndex); } },
    h("span", { class: `swatch f-${item.becomes}` }), `next: ${BECOMES[item.becomes]}`);
}

export function turnRow(turn) {
  return h("article", { class: "msg", id: `turn-${turn.turn_index}`, "data-turns": String(turn.turn_index) },
    h("div", { class: "msg-no" }, turn.turn_index),
    avatarOf(turn),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" },
        who(turn), callsTag(turn), instructionsTag(turn), imagesTag(turn),
        turn.becomes ? nextTag(turn, turn.turn_index) : null,
        h("span", { class: "tok" }, plural(turn.tokens, "token"))),
      turnText(turn)));
}

// The messages of an exchange as they were said, each still reachable from the map.
function originals(turns, faded) {
  return turns.map((t) => h("div", { class: "orig", id: `turn-${t.turn_index}` },
    h("div", { class: "msg-head" }, who(t), callsTag(t), instructionsTag(t), imagesTag(t), h("span", { class: "num" }, `message ${t.turn_index}`),
      h("span", { class: "tok" }, plural(t.tokens, "token"))),
    turnText(t, { faded })));
}

function originalsToggle(key, turns, open) {
  return h("button", { class: "more orig-toggle", type: "button", "aria-expanded": String(open),
    onclick: () => { if (ui.open.has(key)) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
    `${open ? "Hide" : "Show"} what was said in ${messagesLabel(turns.map((t) => t.turn_index))}`);
}

const rangeNo = (turns) => turns.length === 1 ? turns[0].turn_index : `${turns[0].turn_index}–${turns.at(-1).turn_index}`;

// An exchange in the old generation: what the model reads for it, then what was said.
export function spanRow(row) {
  const { span, turns } = row;
  const first = turns[0].turn_index;
  const written = turns.reduce((n, t) => n + t.tokens, 0);
  const head = h("div", { class: "msg-head" }, h("span", { class: "who" }, plural(turns.length, "message")), fidTag(span.level),
    span.becomes ? nextTag(span, first) : null);
  const article = (...body) => h("article", { class: "msg", id: `span-${first}`, "data-turns": span.turns.join(",") },
    h("div", { class: "msg-no" }, rangeNo(turns)), h("div", { "aria-hidden": "true" }), h("div", { class: "msg-body" }, head, ...body));
  if (span.level === "kept") {
    return article(h("div", { class: "keeps-intro" }, `Quoted word for word in the recap · ${plural(span.tokens, "token")} with its markers, for ${plural(written, "token")} as written`),
      originals(turns, false));
  }
  const key = `orig-${first}`;
  const open = ui.open.has(key) || (query() && turns.some((t) => has(t.text)));
  return article(
    h("div", { class: "summary-box" }, h("div", { class: "label" }, `The model reads this instead · ${plural(span.tokens, "token")}, in place of ${plural(written, "token")} as written`),
      textBlock(span.text, { key: `line-${first}` })),
    originalsToggle(key, turns, open),
    open ? originals(turns, true) : null);
}

// A run of exchanges folded out of view: the one line the model reads for all of them.
export function foldedRow(row, facts) {
  const { spans, turns } = row;
  const first = turns[0].turn_index;
  const key = `orig-${first}`;
  const open = ui.open.has(key) || (query() && turns.some((t) => has(t.text)));
  const stored = turns.flatMap((t) => facts.get(t.turn_index) || []).filter((f) => f.status === "active");
  return h("article", { class: "msg gone", id: `span-${first}`, "data-turns": turns.map((t) => t.turn_index).join(",") },
    h("div", { class: "msg-no" }, rangeNo(turns)), h("div", { "aria-hidden": "true" }),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, `${plural(spans.length, "exchange")} · ${plural(turns.length, "message")}`), fidTag("out_of_view")),
      h("div", { class: "summary-box warn" }, h("div", { class: "label" }, h("span", { class: "st warn" }, "⚠ "), "The model reads only this line for all of them"),
        h("div", { class: "text" }, marked(spans[0].text))),
      stored.length ? h("div", { class: "keeps" }, h("div", { class: "keeps-intro" }, "Facts read from them, stored but not sent:"), stored.map(factChip)) : null,
      originalsToggle(key, turns, open),
      open ? originals(turns, true) : null));
}

export function pendingRow(p) {
  // Named for what the proxy knows, never for who wrote it: a side request a client
  // makes, such as Claude Code asking for a suggested next prompt, looks the same.
  return h("article", { class: "msg pending", id: `turn-${p.turn_index}`, "data-turns": String(p.turn_index) },
    h("div", { class: "msg-no" }, p.turn_index), h("div", { class: "av user", "aria-hidden": "true" }, "New"),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, "Newest request"),
        h("span", { class: "muted" }, "not in the conversation yet"), info("newest"), instructionsTag(p)),
      turnText(p)));
}

// The reply as it reached the client. Shown, never recorded from here: the next request says whether the client kept it.
function replyRow(r) {
  return h("article", { class: "msg pending", id: `turn-${r.turn_index}`, "data-turns": String(r.turn_index) },
    h("div", { class: "msg-no" }, r.turn_index), h("div", { class: "av assistant", "aria-hidden": "true" }, "AI"),
    h("div", { class: "msg-body" },
      h("div", { class: "msg-head" }, h("span", { class: "who" }, "Newest reply"),
        h("span", { class: "muted" }, "not in the conversation yet"), info("reply"),
        h("span", { class: "tok" }, plural(r.tokens, "token"))),
      textBlock(r.text, { key: `t-${r.turn_index}` })));
}

export function runRow(run) {
  const first = run.turns[0].turn_index, last = run.turns.at(-1).turn_index;
  const key = `run-${first}`;
  const open = ui.open.has(key);
  // Opened, its exchanges follow the page's order; what was said inside each still reads as it was spoken.
  const rows = ui.newestFirst ? run.rows.toReversed() : run.rows;
  return h("div", { class: `run${open ? " open" : ""}` },
    h("button", { class: "run-head", type: "button", "aria-expanded": String(open),
      onclick: () => { if (open) ui.open.delete(key); else ui.open.add(key); renderTimeline(state.knowledge); } },
      h("span", { class: "run-range" }, `${first}–${last}`),
      h("span", { class: "run-rail" }),
      h("span", {},
        h("span", { class: "run-title" }, `${plural(run.rows.length, "exchange")} archived: the model reads one line for each`),
        h("span", { class: "run-sub" }, h("span", { class: "minis" }, run.rows.map(() => h("span", { class: "mini f-archived" }))),
          plural(run.turns.length, "message"))),
      h("span", { class: "run-toggle" }, open ? "Hide" : "Show")),
    open ? rows.map(spanRow) : null);
}

export function jumpTo(turnIndex) {
  if (ui.tab !== "conversation") offerReturn(ui.tab, window.scrollY);
  showTab("conversation");
  setPromptView(false);
  const k = state.knowledge;
  if (!k) return;
  const facts = factsByTurn(k);
  // Make sure the message is on the page: clear a filter that hides it, open what folds it.
  const row = buildRows(k).find((r) => r.turns.some((t) => t.turn_index === turnIndex) || (r.pending && r.pending.turn_index === turnIndex));
  if (row && !rowVisible(row, facts)) { ui.filter = "all"; ui.query = ""; $("search").value = ""; }
  ui.earlierOpen = true;
  openCard("messages");
  if (row && (row.kind === "span" || row.kind === "folded")) ui.open.add(`orig-${row.turns[0].turn_index}`);
  if (row && row.cat === "archived") {
    const run = foldRuns(buildRows(k).filter((r) => rowVisible(r, facts)))
      .find((r) => r.kind === "run" && r.turns.some((t) => t.turn_index === turnIndex));
    if (run) ui.open.add(`run-${run.turns[0].turn_index}`);
  }
  renderMessages();
  const target = $(`turn-${turnIndex}`);
  if (!target) return;
  // A long smooth scroll runs for a second or more, and a page that redraws while it runs can leave
  // it part way down the list. Far targets are jumped to; the flash shows where they landed.
  const { top, height } = target.getBoundingClientRect();
  const far = Math.abs(top + height / 2 - innerHeight / 2) > innerHeight * FAR_JUMP_SCREENS;
  target.scrollIntoView({ behavior: far ? "auto" : smooth(), block: "center" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1600);
}

export function renderPrompt(k) {
  const tail = k.turns.filter((t) => t.fidelity === "verbatim");
  const whole = [k.front_text ? `[user]\n${k.front_text}` : null, ...tail.map((t) => `[${t.role}]\n${t.text || ""}`)].filter(Boolean).join("\n\n");
  // Copy always takes the prompt in the order the model reads it, whichever way the list is laid out.
  const copy = h("button", { class: "btn", type: "button", title: "Copies the prompt in the order the model reads it", onclick: async () => {
    try { await navigator.clipboard.writeText(whole); copy.textContent = "Copied"; } catch { copy.textContent = "Could not copy"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1500);
  } }, "Copy");
  const head = h("div", { class: "prompt-head" }, h("div", { class: "note" }, "Exactly what LLM-GC forwards before your next message. ",
    hint("Your client's own instructions aren't shown", "Not shown here", "Your client's system prompt, its tools and the instructions it puts inside messages. They go word for word with every request; tool calls and results appear in the text form LLM-GC reads them in."),
    k.front_text ? ". The recap stays the same until the next collection." : ". Nothing is collected yet, so your client's own request goes untouched."), copy);
  // Built as the model reads it, the recap then the recent messages; only the layout follows the viewer's choice.
  const messages = [
    k.front_text ? h("div", { class: "prompt-msg front", id: "prompt-recap" }, h("div", { class: "msg-head" }, h("span", { class: "who" }, "user"),
      h("span", { class: "tag" }, h("span", { class: "key block" }), `the recap · ${plural(k.front_tokens, "token")}`), info("block")), promptText(k.front_text, "prompt-front")) : null,
    ...tail.map((t) => h("div", { class: "prompt-msg", id: `prompt-turn-${t.turn_index}` }, h("div", { class: "msg-head" }, h("span", { class: "who" }, t.role),
      writerTag(t) ?? (t.tool_calls ? toolTag("tool calls") : null), instructionsTag(t),
      h("span", { class: "num" }, `message ${t.turn_index} · ${plural(t.tokens, "token")}`)), promptText(t.text || "(no text)", `prompt-${t.turn_index}`))),
  ];
  keepPlace($("prompt"), ".prompt-msg[id]", () => setChildren($("prompt"), head, ui.newestFirst ? messages.toReversed() : messages));
}

// A long block starts folded, as messages do elsewhere; Copy still takes the whole prompt.
function promptText(text, key) {
  if (text.length <= CLAMP_CHARS && text.split("\n").length <= 8) return h("pre", {}, marked(text));
  const open = ui.open.has(key) || (query() && has(text));
  const toggle = () => {
    const nowOpen = !ui.open.has(key);
    if (nowOpen) ui.open.add(key); else ui.open.delete(key);
    body.classList.toggle("clamped", !nowOpen);
    more.textContent = nowOpen ? "Show less" : "Show more";
  };
  const body = h("pre", { class: open ? "" : "clamped" }, marked(text));
  const more = h("button", { class: "more", type: "button", onclick: toggle }, open ? "Show less" : "Show more");
  return [body, more];
}

// What the model reads of a message said in an exchange shown at each level below kept.
const BELOW_KEPT = {
  compacted: "now summarised: the model reads its exchange's summary, which may word it differently",
  archived: "now archived: the model reads only its exchange's opening line and facts",
  out_of_view: "now out of view: the model knows its exchange exists, not what was said",
};

// A search answers the question it is usually asked: does the model still receive these words?
// Matching is literal, as the search is, so the answer is about these exact words.
function renderSearchVerdict(k) {
  const box = $("search-verdict");
  const words = ui.query.trim();
  const verbatim = k.turns.filter((t) => t.fidelity === "verbatim" && has(t.text)).map((t) => t.turn_index);
  const newest = pendingTurns(k).filter((p) => has(p.text));
  const inRecap = has(k.front_text);
  const condensed = k.turns.filter((t) => t.fidelity !== "verbatim" && has(t.text));
  box.hidden = !words || (!verbatim.length && !newest.length && !inRecap && !condensed.length);
  if (box.hidden) return;
  if (verbatim.length || inRecap || newest.length) {
    const how = [verbatim.length ? `as written, in recent ${messagesLabel(verbatim)}` : null, inRecap ? "in the recap" : null,
      newest.length ? "in the newest request, which is not in the conversation yet" : null].filter(Boolean).join(", and ");
    box.className = "search-verdict good";
    setChildren(box, h("b", {}, `“${words}”: the model receives it`), ` ${how}.`);
    return;
  }
  const said = Object.entries(BELOW_KEPT).map(([level, how]) => {
    const turns = condensed.filter((t) => t.fidelity === level).map((t) => t.turn_index);
    return turns.length ? `Said in ${messagesLabel(turns)}, ${how}.` : null;
  }).filter(Boolean);
  box.className = "search-verdict warn";
  setChildren(box, h("b", {}, `“${words}”: the model does not receive these exact words.`), ` ${said.join(" ")}`,
    " The words are stored, so a later collection shows them again if the conversation comes back to them.");
}

export function setPromptView(on) {
  ui.promptView = on;
  $("view-prompt").setAttribute("aria-pressed", String(on));
  $("view-timeline").setAttribute("aria-pressed", String(!on));
  $("prompt").hidden = !on;
  $("timeline").hidden = on;
  $("chips").hidden = on;
  // Both views follow the one order switch; its subtitle differs, so it is redrawn with the view.
  renderOrderControl();
  $("filter-note").hidden = on || (ui.filter === "all" && !query());
  if (state.knowledge) { if (on) renderPrompt(state.knowledge); else renderTimeline(state.knowledge); }
}

// ----------------------------------------------------------------- what the model is told, fact by fact

// A current fact reaches the model on its exchange's line in the recap, or in its words when that
// exchange is kept; `sent` is set from the facts the recap actually carries.
const reaches = (fact) => fact.sent || fact.span_level === "kept";
// The facts the model can't see first, then the newest: what most needs a look leads.
const lookFirst = (a, b) => reaches(a) - reaches(b) || b.turn - a.turn;
const changedAt = (fact) => fact.superseded_by_turn ?? fact.turn;
// Whether the engine reads facts from the assistant's turns; without a model it reads only the user's.
const readsAssistant = () => !state.config || !state.config.facts_read_from || state.config.facts_read_from.includes("assistant");

// Every fact read so far by who stated it, with how it reaches the model, then each value replaced.
// The head - the counts, how far the reading reaches, and any warning - stays in view when the card is
// folded. It leads with "as of message N", never "current": facts are read only when messages are
// collected, so a value a recent message has already changed would read as still in force.
export function renderFacts(k) {
  const matches = (f) => !query() || has(f.content) || has(f.topic) || has(f.superseded_by);
  const active = k.facts.filter((f) => f.status === "active");
  const replaced = k.facts.filter((f) => f.status === "superseded");
  const unseen = active.filter((f) => !reaches(f));
  const collected = k.turns.filter((t) => t.fidelity !== "verbatim").map((t) => t.turn_index);
  const recent = k.turns.filter((t) => t.fidelity === "verbatim").map((t) => t.turn_index);
  // An exchange made while its facts couldn't be read holds none, whatever its messages settled.
  const unreadTurns = k.spans.filter((s) => s.facts_unread).flatMap((s) => s.turns);
  const unread = unreadTurns.length ? h("p", { class: "facts-unread" },
    h("span", { class: "st warn" }, "⚠ "), "Facts couldn't be read from ", hint(messagesLabel(unreadTurns), "Facts not read",
      "None are listed for them yet: the next collection reads them again. Until then they are never archived: the model reads them word for word, or as their summary."), ".") : null;
  if (!k.facts.length) {
    fill("facts");
    fill("facts-head", unread || h("p", { class: "facts-note" }, collecting()
      ? "A collection is reading this conversation now; the facts it reads appear here when it finishes."
      : !k.cuts ? "Nothing collected yet: facts are read from messages when a collection takes them."
      : readsAssistant() ? "No settled facts in the collected messages yet. A fact is a topic with a value, such as “database: Postgres”."
      : "No facts found. Without a model, only your own messages are read, for statements such as “The database is Postgres.”"));
    return;
  }
  fill("facts-head",
    h("div", { class: "facts-headline" },
      collected.length ? h("span", {}, "As of ", h("b", {}, `message ${Math.max(...collected)}`)) : null,
      h("span", {}, h("b", {}, fmt(active.length)), ` ${active.length === 1 ? "fact" : "facts"}`),
      reachLine(active.length, unseen.length),
      replaced.length ? h("span", {}, h("b", {}, fmt(replaced.length)), ` earlier ${replaced.length === 1 ? "value" : "values"} replaced`)
        : h("span", {}, "nothing replaced")),
    h("p", { class: "facts-note" },
      collected.length ? `Read from ${messagesLabel(collected)}. ` : null,
      recent.length ? hint("Newer messages are read once collected.", "Facts can lag the newest messages",
        `${capitalised(messagesLabel(recent))} ${recent.length === 1 ? "is" : "are"} sent word for word, so a value changed there reaches the model now and shows here once collected.`) : null),
    unread);
  const assistantFacts = readsAssistant() ? active.filter((f) => f.speaker === "assistant").sort(lookFirst) : null;
  const scrolled = new Map([...$("facts").querySelectorAll(".facts-col .list")].map((list) => [list.dataset.col, list.scrollTop]));
  fill("facts", h("div", { class: "facts-cols" },
    factColumn(matches, { id: "user", title: "You said", mark: avatar("user"), row: currentFact,
      facts: active.filter((f) => f.speaker !== "assistant").sort(lookFirst),
      empty: ["No facts from you yet", "Choices, names, numbers and dates you state appear here once their messages are collected."] }),
    factColumn(matches, { id: "assistant", title: "The assistant said", mark: avatar("assistant"), row: currentFact, facts: assistantFacts,
      empty: assistantFacts
        ? ["No assistant facts yet", "Values the assistant states, such as a recommended setting, appear here once their messages are collected."]
        : ["The assistant's messages aren't read", "Facts are read from your messages only here, as they are when LLM-GC runs without a model. "
          + "The model still reads the assistant's replies, in the recap and as written."] }),
    factColumn(matches, { id: "replaced", title: "Replaced", mark: h("div", { class: "av replaced", "aria-hidden": "true" }, "↺"),
      row: replacedFact, facts: replaced.sort((a, b) => changedAt(b) - changedAt(a)),
      empty: ["No replacements on record", "When a later message changes a value, such as Elasticsearch → OpenSearch, the old and new values appear here."] })));
  // Redrawn at every update and keystroke: each column stays where its reader scrolled it.
  for (const list of $("facts").querySelectorAll(".facts-col .list")) {
    list.scrollTop = scrolled.get(list.dataset.col) ?? 0;
    markMore(list);
  }
}

// Fades the edge a list scrolls on, so facts out of its box don't read as the end of the column.
function markMore(list) {
  const above = list.scrollTop > 1;
  const below = list.scrollTop + list.clientHeight < list.scrollHeight - 1;
  if (above || below) list.dataset.more = above && below ? "both" : above ? "above" : "below";
  else delete list.dataset.more;
}

// How many of the facts listed reach the model, worded so the parts visibly add up to the whole.
function reachLine(total, unseen) {
  const seen = total - unseen;
  const reach = (n) => n === 1 ? " reaches the model" : " reach the model";
  const missing = (words) => h("span", { class: "alert" }, h("span", { class: "st warn" }, "⚠ "), ...words);
  if (!unseen) return h("span", {}, total === 1 ? "it reaches the model" : ["all ", h("b", {}, fmt(total)), reach(total)]);
  if (!seen) return missing([total === 1 ? "it doesn't" : "none", reach(2)]);
  return [h("span", {}, h("b", {}, fmt(seen)), reach(seen)), missing([h("b", {}, fmt(unseen)), unseen === 1 ? " doesn't" : " don't"])];
}

// One column: every fact under a heading, in a box that scrolls once they outgrow it, so the card
// keeps its height however many pile up; or a card saying why it has none, so an empty column never
// reads as missing content. `facts` is null for a side whose turns are not read.
function factColumn(matches, { id, title, mark, facts, row, empty }) {
  const shown = (facts ?? []).filter(matches);
  const key = `facts-${id}`;
  if (!shown.length) {
    const [what, why] = facts?.length ? ["No match", "None of these facts match your search."] : empty;
    return h("section", { class: "facts-col empty", "aria-label": title }, mark, h("div", {}, h("b", {}, what), h("p", {}, why)));
  }
  return h("section", { class: "facts-col", "aria-labelledby": `${key}-title` },
    h("h3", { id: `${key}-title` }, mark, title, h("span", { class: "count" }, fmt(shown.length))),
    h("div", { class: "list", "data-col": id, onscroll: (e) => markMore(e.currentTarget) }, shown.map(row)));
}

// A current fact, as a link to the message that stated it; the column says who.
function currentFact(fact) {
  return h("button", { class: "fact", type: "button", onclick: () => jumpTo(fact.turn) },
    h("span", { class: "fact-topic" }, topicOf(fact)),
    h("span", { class: "fact-what" }, marked(fact.content)),
    h("span", { class: "fact-meta" }, h("span", { class: "fact-link num" }, `message ${fact.turn}`), factStatus(fact),
      h("span", { class: "fact-go", "aria-hidden": "true" }, "→")));
}

// A replaced value, what replaced it, and a link to each message. The recap notes the replacement
// on the old value's exchange; when that exchange is out of view, the old value isn't sent at all.
function replacedFact(fact) {
  const link = (turn, speaker) => h("button", { class: "link-btn num", type: "button", title: `Show message ${turn}`, onclick: () => jumpTo(turn) },
    said(turn, speaker));
  const noted = fact.span_level !== null && fact.span_level !== "out_of_view";
  return h("div", { class: "fact s-superseded" },
    h("span", { class: "fact-topic" }, topicOf(fact)),
    h("span", { class: "fact-what" }, h("span", { class: "was" }, marked(fact.content)),
      fact.superseded_by !== null ? h("span", { class: "now" }, "→ ", marked(fact.superseded_by)) : null),
    h("span", { class: "fact-meta" }, link(fact.turn, fact.speaker),
      fact.superseded_by_turn !== null ? ["→", link(fact.superseded_by_turn, fact.superseded_by_speaker)] : null,
      statusTag(noted ? "noted" : "oldUnsent")));
}

// "How fact collection works", below the facts: drawn once, and from the same meanings as each status's tooltip.
export function renderFactsGuide() {
  fill("facts-how",
    h("p", {}, "When a collection takes messages, the model that writes their summaries also reads what they settled — choices, names, ",
      "numbers, dates — as short topic: value notes, once. Without a model, simple patterns read your own messages instead, ",
      "catching statements such as “The database is Postgres.”"),
    h("dl", { class: "facts-statuses" }, Object.values(FACT_STATUS).map(([glyph, tone, label, meaning]) => [
      h("dt", {}, glyph ? h("span", { class: `st ${tone}` }, `${glyph} `) : null, label), h("dd", {}, meaning)])),
    h("p", {}, "When a later message changes a value, the new fact is filed under the same topic and the old one moves to Replaced. ",
      "Only you change a decision you made: when the assistant recommends something else, your decision stays, and the ",
      "recommendation stays in its exchange's summary."),
    h("p", {}, "Facts are a small part of what the model is told: summaries and recent messages carry the rest."));
}

