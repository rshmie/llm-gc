// Entry point: polling, the conversation picker, and wiring the controls.

import { CARD_INFO, CATS, GLOSSARY, ORDER, swatch } from "./vocabulary.js";
import { $, clearMotionCaches, fill, h, plural, setChildren, storeGet, storeSet } from "./dom.js";
import { OTHER, collecting, getJSON, sessionPath, state, ui } from "./state.js";
import { hideTooltip } from "./tooltip.js";
import { THEMES, applyTheme, fillConfigValues, offerReturn, setLive, showCleanupNotice, showTab } from "./shell.js";
import { openReference, setGuide } from "./guide.js";
import { celebrateCleanup, clearMapCache, jumpTo, renderBar, renderChips, renderFacts, renderFactsGuide, renderHero, renderMap, renderMessages, renderOrderControl, renderPrompt, renderSummary, renderTimeline, setOrder, setPromptView } from "./conversation.js";
import { initCards, openCard, renderChart, renderChecks, renderEngineStatus, renderGenerations, renderHistory, renderNext, renderProblems, renderProxy, renderRelevance, renderSettings } from "./engine.js";
import { renderRunGcButton, runGcNow } from "./run-gc.js";
import { renderSessionCard } from "./session.js";
import { renderTrace } from "./trace.js";

let shownCollecting = false;
// A picker value naming a Claude Code session whose conversations have all closed: its requests remain.
const REQUESTS_ONLY = "requests:";

// A Claude Code session runs several conversations through the proxy: its main one, one per subagent,
// and a new main one after /compact or an edited message. Each is collected on its own, so each is its
// own entry; the picker groups them under the session they came from and names each one's part.
function groupSessions(sessions) {
  const groups = new Map();
  for (const s of sessions) {
    const key = s.client_session || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  }
  const started = (s) => Date.parse(s.started_at);
  return [...groups].map(([clientSession, members]) => {
    if (!clientSession) return { clientSession: null, entries: members.map((s) => ({ s, part: null })) };
    const byStart = [...members].sort((a, b) => started(a) - started(b));
    const mains = byStart.filter((s) => !s.agent);
    const agents = byStart.filter((s) => s.agent);
    // The newest main conversation is the one Claude Code is in; an older one was replaced.
    const entries = [
      ...mains.slice(-1).map((s) => ({ s, part: { kind: "main", label: "Main" } })),
      ...agents.map((s, i) => ({ s, part: { kind: "subagent", label: `Subagent ${i + 1}` } })),
      ...mains.slice(0, -1).reverse().map((s) => ({ s, part: { kind: "earlier", label: "Earlier main" } })),
    ];
    return { clientSession, since: started(byStart[0]), entries, agents: agents.length, earlier: Math.max(mains.length - 1, 0) };
  });
}

const shortId = (id) => id.slice(0, 8);
const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// Everything the picker offers, the most recently active first: each Claude Code session with its live
// conversations, each session the proxy counted whose conversations have closed, then the conversations
// no client session named.
function pickerGroups() {
  const groups = groupSessions(state.sessions);
  const named = groups.filter((g) => g.clientSession);
  const closed = (state.proxy ? state.proxy.client_sessions : [])
    .filter((c) => !named.some((g) => g.clientSession === c.client_session))
    .map((c) => ({ clientSession: c.client_session, since: Date.parse(c.first_request_at), entries: [], agents: 0, earlier: 0 }));
  return [...named, ...closed, ...groups.filter((g) => !g.clientSession)];
}

// The conversation shown: the one picked, or the main conversation of the most recent session. A session's
// main conversation is followed when a newer one replaces it - after /compact - unless another was picked.
function chooseConversation(groups) {
  const entries = groups.flatMap((g) => g.entries);
  const current = entries.find((e) => e.s.session_id === state.selected);
  if (current) {
    const group = groups.find((g) => g.entries.includes(current));
    const main = group.entries.find((e) => e.part && e.part.kind === "main");
    if (ui.followMain && main && main !== current) showConversation(main.s.session_id);
    else state.clientSession = current.s.client_session || OTHER;
    return;
  }
  // A session picked for its requests alone stays picked while the proxy still counts it.
  if (!state.selected && state.clientSession && groups.some((g) => g.clientSession === state.clientSession && !g.entries.length)) return;
  const first = groups.find((g) => g.entries.length);
  const pick = first ? (first.entries.find((e) => e.part && e.part.kind === "main") || first.entries[0]) : null;
  if (pick) { showConversation(pick.s.session_id); ui.followMain = true; return; }
  const requestsOnly = groups[0];
  state.selected = null;
  state.clientSession = requestsOnly ? requestsOnly.clientSession : null;
}

// Show one conversation, and the session it ran in on the Requests tab.
function showConversation(sessionId) {
  const entry = state.sessions.find((s) => s.session_id === sessionId);
  const clientSession = entry ? entry.client_session || OTHER : null;
  if (clientSession !== state.clientSession) { state.trace = null; ui.turnOpen.clear(); ui.openAgents.clear(); ui.showSide.clear(); }
  state.clientSession = clientSession;
  resetConversation(sessionId);
}

// A conversation the person picked: followed as the main one only if it is the main one.
function pickConversation(sessionId) {
  const entry = pickerGroups().flatMap((g) => g.entries).find((e) => e.s.session_id === sessionId);
  ui.followMain = !!(entry && entry.part && entry.part.kind === "main");
}

function resetConversation(sessionId) {
  state.selected = sessionId; state.version = null; state.knowledge = null;
  ui.open.clear(); ui.filter = "all"; ui.query = ""; $("search").value = ""; ui.earlierOpen = false;
  clearMapCache();
  clearMotionCaches();
}

// The browser rebuilds its open menu, scrolled back to the top, whenever the options change. So an option names
// its conversation and nothing that grows while it runs - the page shows the live counts - and the options are
// replaced only when a name does change.
let shownPicker = null;

function renderPicker(groups) {
  const select = $("session-select");
  const conversation = ({ s, part }) => ({ value: s.session_id, title: s.session_id, text: `${part ? `${part.label} · ` : ""}${s.title || s.session_id}` });
  const listed = groups.map((g) => !g.clientSession
    ? { label: "No Claude Code session named", options: g.entries.map(conversation) }
    : { label: `Claude Code session ${shortId(g.clientSession)} · since ${clock(g.since)}`,
      options: g.entries.length ? g.entries.map(conversation)
        : [{ value: `${REQUESTS_ONLY}${g.clientSession}`, text: "Its conversations have closed: requests only" }] });
  const key = JSON.stringify(listed);
  if (key !== shownPicker) {
    shownPicker = key;
    setChildren(select, listed.length
      ? listed.map((g) => h("optgroup", { label: g.label }, g.options.map((o) => h("option", { value: o.value, title: o.title }, o.text))))
      : h("option", {}, "none yet"));
  }
  select.disabled = !groups.length;
  const value = state.selected || (state.clientSession ? `${REQUESTS_ONLY}${state.clientSession}` : "");
  if (select.value !== value) select.value = value;
  $("conv-bar").hidden = !groups.length;
  const button = $("run-gc");
  button.disabled = !state.selected || button.dataset.running === "1";
  renderOrigin(groupSessions(state.sessions));
}

// The conversations no client session named have no trace: the Requests tab lists them instead.
function renderOther(groups) {
  $("other-card").hidden = state.clientSession !== OTHER;
  if (state.clientSession !== OTHER) return;
  const entries = groups.filter((g) => !g.clientSession).flatMap((g) => g.entries);
  fill("other-card",
    h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Conversations no Claude Code session named"),
      h("p", { class: "sub" }, "Their client names no session, so there is no request-by-request trace of them. Open one to see what the model receives."))),
    h("ul", { class: "other-list" }, entries.map(({ s }) => h("li", {},
      h("button", { class: "link-btn", type: "button", onclick: () => openConversation(s.session_id, null) }, s.title || s.session_id),
      h("span", { class: "secondary" }, ` · ${plural(s.recorded_turns, "message")} · ${plural(s.cuts, "collection")}`)))));
}

// From the Requests tab to one conversation, at a message when one is named, with a way back.
function openConversation(sessionId, turnIndex) {
  offerReturn("requests", window.scrollY);
  pickConversation(sessionId);
  if (sessionId === state.selected && state.knowledge) {
    // Already shown: nothing will be fetched, so jump now.
    showTab("conversation");
    if (turnIndex !== null) jumpTo(turnIndex);
    return;
  }
  showConversation(sessionId);
  ui.pendingJump = turnIndex;
  showTab("conversation");
  refresh();
}

const traceActions = {
  live: (sessionId) => !!sessionId && state.sessions.some((s) => s.session_id === sessionId),
  open: openConversation,
};

// Where the selected conversation sits in its Claude Code session, so its numbers are read as its own.
function renderOrigin(groups) {
  const group = groups.find((g) => g.clientSession && g.entries.some((e) => e.s.session_id === state.selected));
  $("origin-line").hidden = !group;
  if (!group) return;
  const { part } = group.entries.find((e) => e.s.session_id === state.selected);
  const what = {
    main: [h("b", {}, "Main conversation"), " of this Claude Code session."],
    subagent: [h("b", {}, part.label), " of this Claude Code session: a subagent works in a conversation of its own, recorded but not collected."],
    earlier: [h("b", {}, "Earlier main conversation"), ": a newer one replaced it after /compact or an edited message, so it is not expected to grow; it closes once idle."],
  }[part.kind];
  fill("origin-line", ...what);
}

function renderConversation() {
  const k = state.knowledge;
  hideTooltip();
  renderHero(k);
  renderChips(k);
  if (ui.promptView) renderPrompt(k); else renderTimeline(k);
  renderFacts(k);
  renderRelevance(k);
}

async function refresh() {
  try {
    const overview = await getJSON("/api/overview");
    state.config = overview.config;
    fillConfigValues(state.config);
    state.proxy = overview.proxy;
    state.pause = overview.pause;
    state.sessions = overview.sessions;
    const groups = pickerGroups();
    chooseConversation(groups);
    renderPicker(groups);
    if (state.clientSession && state.clientSession !== OTHER) {
      state.trace = await getJSON(`/api/trace/${encodeURIComponent(state.clientSession)}`);
    } else {
      state.trace = null;
    }
    renderTrace(traceActions);
    renderOther(groups);
    renderSessionCard((sessionId) => openConversation(sessionId, null));
    renderSettings();
    $("no-session").hidden = !!state.clientSession;
    showTab(ui.tab);

    if (state.selected) {
      const current = state.sessions.find((s) => s.session_id === state.selected);
      // Content changes only with the conversation's version, and a reply arrives without one;
      // activity is cheap and polled every time.
      const replyShown = state.transcript && state.transcript.reply ? state.transcript.reply.turn_index : null;
      // The verdict on a prepared collection is the engine's, for the cache as it was read: re-read when the cache turns.
      const cacheTurned = !!state.knowledge && (current.cache?.state ?? null) !== (state.knowledge.cache?.state ?? null);
      state.cache = current.cache;
      if (current.version !== state.version || (current.newest_reply ?? null) !== replyShown || cacheTurned) {
        const previousCuts = state.knowledge ? state.knowledge.cuts : null;
        const [k, transcript] = await Promise.all([getJSON(sessionPath()), getJSON(sessionPath("/transcript"))]);
        if (k) {
          state.knowledge = k;
          state.transcript = transcript;
          state.version = k.version;
          renderConversation();
          if (ui.pendingJump !== null && ui.tab === "conversation") {
            const turnIndex = ui.pendingJump;
            ui.pendingJump = null;
            jumpTo(turnIndex);
          }
          if (previousCuts !== null && k.cuts > previousCuts) {
            showCleanupNotice(k.cuts - previousCuts > 1
              ? `${plural(k.cuts - previousCuts, "collection")} just applied: older messages moved into the recap.`
              : "A collection was just applied: older messages moved into the recap.");
            celebrateCleanup();
          }
        }
      }
      state.activity = await getJSON(sessionPath("/activity"));
      // The measured counts, the window's source and the cache change with every request, not only with the conversation.
      if (state.knowledge) { renderSummary(state.knowledge); renderNext(state.knowledge); renderGenerations(state.knowledge); }
      // The facts panel is otherwise drawn only when the conversation changes; it says when a collection is running.
      if (state.knowledge && !state.knowledge.facts.length && collecting() !== shownCollecting) renderFacts(state.knowledge);
      shownCollecting = collecting();
      renderProxy();
      renderEngineStatus();
      renderChart();
      renderChecks();
      renderProblems();
      renderHistory();
    }
    if (!state.knowledge) renderBar(null);
    renderRunGcButton();
    setLive("live", `live · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
  } catch {
    setLive("down", "proxy unreachable");
  }
}

// ----------------------------------------------------------------- wiring

// The tab in the address wins, then the one last used; the Conversation tab otherwise. The Requests tab was
// once called Session, and old addresses and remembered choices still say so.
const TABS = ["conversation", "requests", "engine"];
const named = (tab) => (tab === "session" ? "requests" : tab);
const hashed = named(location.hash.slice(1));
const remembered = named(storeGet("llmgc.tab"));
ui.tab = TABS.includes(hashed) ? hashed : TABS.includes(remembered) ? remembered : "conversation";
$("tab-requests").addEventListener("click", () => { showTab("requests"); renderTrace(traceActions, true); });
$("tab-conversation").addEventListener("click", () => showTab("conversation"));
$("tab-engine").addEventListener("click", () => showTab("engine"));
$("view-timeline").addEventListener("click", () => setPromptView(false));
$("order").addEventListener("click", () => setOrder(!ui.newestFirst));
$("view-prompt").addEventListener("click", () => setPromptView(true));
$("run-gc").addEventListener("click", () => runGcNow(refresh));
$("open-guide").addEventListener("click", () => setGuide(true));
$("intro-guide").addEventListener("click", () => setGuide(true));
$("close-guide").addEventListener("click", () => setGuide(false));
$("backdrop").addEventListener("click", () => setGuide(false));
$("intro-close").addEventListener("click", () => { $("intro").hidden = true; storeSet("llmgc.intro", "dismissed"); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && $("guide").classList.contains("open")) setGuide(false);
  if (e.key === "/" && document.activeElement.tagName !== "INPUT" && ui.tab === "conversation" && state.selected) { e.preventDefault(); $("search").focus(); }
});
$("session-select").addEventListener("change", (e) => {
  const value = e.target.value;
  if (value.startsWith(REQUESTS_ONLY)) {
    // A session whose conversations have closed: only its requests can be shown.
    state.clientSession = value.slice(REQUESTS_ONLY.length); state.trace = null;
    ui.turnOpen.clear(); ui.openAgents.clear(); ui.showSide.clear();
    resetConversation(null);
    showTab("requests");
  } else {
    pickConversation(value);
    showConversation(value);
  }
  refresh();
});
let searchTimer = null;
$("search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { ui.query = e.target.value; if (ui.query.trim()) openCard("messages"); renderMessages(); }, 120);
});
$("theme").addEventListener("click", () => {
  const order = Object.keys(THEMES);
  const now = storeGet("llmgc.theme") || "system";
  applyTheme(order[(order.indexOf(now) + 1) % order.length]);
  renderChart();
});
new ResizeObserver(() => renderChart()).observe($("chart"));
// Bar widths keep a floor in pixels, so the map is redrawn when its width changes.
let mapWidth = 0;
new ResizeObserver(() => {
  const width = $("an-bars").clientWidth;
  if (state.knowledge && width && Math.abs(width - mapWidth) > 2) { mapWidth = width; renderMap(state.knowledge); }
}).observe($("an-bars"));
initCards();
renderFactsGuide();

applyTheme(storeGet("llmgc.theme") || "system");
renderOrderControl();
$("intro").hidden = storeGet("llmgc.intro") === "dismissed";
// The guide's reference: every term and every card in full, each an ⓘ's target.
const reference = (kind) => ([key, [title, short, more]]) => [h("dt", { id: `ref-${kind}-${key}` }, title), h("dd", {}, more || short)];
fill("glossary", Object.entries(GLOSSARY).map(reference("term")));
fill("card-glossary", Object.entries(CARD_INFO).map(reference("card")));
// Each part of the request diagram opens its own entry, as an ⓘ does.
for (const part of document.querySelectorAll(".anatomy [data-ref]")) part.addEventListener("click", () => openReference(part.dataset.ref));
fill("guide-colours", ORDER.map((key) => h("div", {}, swatch(key), h("div", {}, h("b", {}, CATS[key].label), h("span", {}, ` — ${GLOSSARY[key][1]}`)))),
  h("div", {}, h("span", { class: "qdot", style: "margin:3px 2px 0" }), h("div", {}, h("b", {}, "Changes at the next collection"), h("span", {}, " — a dot on the message's bar, in the colour it will become. On the map, a bar's height shows how much of the message the model still has, and its width how long the message is."))));

refresh();
setInterval(refresh, 1000);
