// Entry point: polling, the session picker, and wiring the controls.

import { CATS, GLOSSARY, GUIDE_TERMS, ORDER, swatch } from "./vocabulary.js";
import { $, clearMotionCaches, fill, h, plural, setChildren, storeGet, storeSet } from "./dom.js";
import { OTHER, collecting, getJSON, sessionPath, state, ui } from "./state.js";
import { hideTooltip } from "./tooltip.js";
import { THEMES, applyTheme, fillConfigValues, offerReturn, setGuide, setLive, showCleanupNotice, showTab } from "./shell.js";
import { celebrateCleanup, clearMapCache, jumpTo, renderChips, renderFacts, renderFactsGuide, renderHero, renderLastCollection, renderMap, renderMessages, renderPrompt, renderSize, renderStatus, renderTimeline, renderWindow, setPromptView } from "./conversation.js";
import { initCards, renderChart, renderChecks, renderEngineStatus, renderGenerations, renderHistory, renderNext, renderProblems, renderProxy, renderRelevance, renderSettings } from "./engine.js";
import { renderRunGcButton, runGcNow } from "./run-gc.js";
import { renderSessionCard } from "./session.js";
import { renderTrace } from "./trace.js";

let shownCollecting = false;

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

// The sessions the picker offers, the most recently active first: each Claude Code session the proxy has
// counted or that a live conversation names, then the conversations no client session named, as one entry.
function clientSessions() {
  const counted = state.proxy ? state.proxy.client_sessions : [];
  const named = new Map(counted.map((c) => [c.client_session, { id: c.client_session, since: Date.parse(c.first_request_at), requests: c.traffic.requests }]));
  for (const s of state.sessions) {
    if (s.client_session && !named.has(s.client_session)) named.set(s.client_session, { id: s.client_session, since: Date.parse(s.started_at), requests: null });
  }
  const entries = [...named.values()];
  if (state.sessions.some((s) => !s.client_session)) entries.push({ id: OTHER, since: null, requests: null });
  return entries;
}

function chooseClientSession(entries) {
  if (entries.some((e) => e.id === state.clientSession)) return;
  state.clientSession = entries.length ? entries[0].id : null;
  state.trace = null;
}

function renderClientPicker(entries) {
  const select = $("client-select");
  setChildren(select, entries.length ? entries.map((e) => h("option", { value: e.id, title: e.id === OTHER ? "" : e.id },
    e.id === OTHER ? "Other conversations (no client session named)"
      : `Claude Code session ${shortId(e.id)} · since ${clock(e.since)}${e.requests !== null ? ` · ${plural(e.requests, "request")}` : ""}`))
    : h("option", {}, "none yet"));
  select.disabled = !entries.length;
  if (state.clientSession) select.value = state.clientSession;
}

// The conversations of the picked session, as its own parts; the conversation shown stays one of them.
function sessionEntries() {
  const groups = groupSessions(state.sessions);
  const group = groups.find((g) => (state.clientSession === OTHER ? !g.clientSession : g.clientSession === state.clientSession));
  return group ? group.entries : [];
}

// The session's main conversation first: it is the one the person is in. Followed when a newer one replaces
// it - after /compact - unless the person picked a conversation themselves.
function chooseConversation(entries) {
  const main = entries.find((e) => e.part && e.part.kind === "main");
  const shown = entries.some((e) => e.s.session_id === state.selected);
  if (shown && !(ui.followMain && main && main.s.session_id !== state.selected)) return;
  resetConversation((main || entries[0] || { s: { session_id: null } }).s.session_id);
  ui.followMain = true;
}

// A conversation the person picked: followed as the main one only if it is the main one.
function pickConversation(sessionId) {
  const entry = sessionEntries().find((e) => e.s.session_id === sessionId);
  ui.followMain = !!(entry && entry.part && entry.part.kind === "main");
}

function resetConversation(sessionId) {
  state.selected = sessionId; state.version = null; state.knowledge = null;
  ui.open.clear(); ui.filter = "all"; ui.query = ""; $("search").value = ""; ui.earlierOpen = false;
  clearMapCache();
  clearMotionCaches();
}

function renderSessions(entries) {
  const select = $("session-select");
  const option = ({ s, part }) => h("option", { value: s.session_id, title: s.session_id },
    `${part ? `${part.label} · ` : ""}${s.title ? s.title : s.session_id} · ${plural(s.recorded_turns, "message")} · ${plural(s.cuts, "collection")}${s.aging_in_flight ? " · collecting…" : ""}`);
  setChildren(select, entries.length ? entries.map(option) : h("option", {}, "none live: closed once idle"));
  select.disabled = !entries.length;
  if (state.selected) select.value = state.selected;
  const button = $("run-gc");
  button.disabled = !state.selected || button.dataset.running === "1";
  renderOrigin(groupSessions(state.sessions));
}

// Conversations no client session named have no trace: the Session tab lists them instead.
function renderOther(entries) {
  $("other-card").hidden = state.clientSession !== OTHER;
  if (state.clientSession !== OTHER) return;
  fill("other-card",
    h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Other conversations"),
      h("p", { class: "sub" }, "Their client names no session, so they are listed on their own, without a trace. Open one to see what the model receives."))),
    h("ul", { class: "other-list" }, entries.map(({ s }) => h("li", {},
      h("button", { class: "link-btn", type: "button", onclick: () => openConversation(s.session_id, null) }, s.title || s.session_id),
      h("span", { class: "secondary" }, ` · ${plural(s.recorded_turns, "message")} · ${plural(s.cuts, "collection")}`)))));
}

// From the Session tab to one conversation, at a message when one is named, with a way back.
function openConversation(sessionId, turnIndex) {
  offerReturn("session", window.scrollY);
  pickConversation(sessionId);
  if (sessionId === state.selected && state.knowledge) {
    // Already shown: nothing will be fetched, so jump now.
    showTab("conversation");
    if (turnIndex !== null) jumpTo(turnIndex);
    return;
  }
  resetConversation(sessionId);
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
  const session = `Claude Code session ${shortId(group.clientSession)}`;
  const what = {
    main: [h("b", {}, "Main conversation"), ` of ${session}.`],
    subagent: [h("b", {}, part.label), ` of ${session}: a subagent works in a conversation of its own.`],
    earlier: [h("b", {}, "Earlier main conversation"), ` of ${session}. A newer one replaced it after /compact or an edited message, so this one is not expected to grow; it closes once idle.`],
  }[part.kind];
  const parts = [group.entries.some((e) => e.part.kind === "main") ? "the main one" : null,
    group.agents ? plural(group.agents, "subagent") : null, group.earlier ? plural(group.earlier, "earlier main conversation") : null].filter(Boolean);
  const count = group.entries.length === 1 ? " It is the only conversation the session has here."
    : ` The session has ${plural(group.entries.length, "conversation")} here: ${parts.join(", ")}.`;
  fill("origin-line", ...what, count, " LLM-GC collects each main conversation on its own, and records each subagent's without collecting it. The Session tab shows the whole session; the Engine tab's totals, every conversation since the proxy started.");
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
    state.sessions = overview.sessions;
    const picks = clientSessions();
    chooseClientSession(picks);
    renderClientPicker(picks);
    const entries = sessionEntries();
    chooseConversation(entries);
    renderSessions(entries);
    if (state.clientSession && state.clientSession !== OTHER) {
      state.trace = await getJSON(`/api/trace/${encodeURIComponent(state.clientSession)}`);
    } else {
      state.trace = null;
    }
    renderTrace(traceActions);
    renderOther(entries);
    renderSessionCard((sessionId) => openConversation(sessionId, null));
    renderSettings();
    $("no-session").hidden = !!state.clientSession;
    showTab(ui.tab);

    if (state.selected) {
      const current = state.sessions.find((s) => s.session_id === state.selected);
      // Content changes only with the conversation's version, and a reply arrives without one;
      // activity is cheap and polled every time.
      const replyShown = state.transcript && state.transcript.reply ? state.transcript.reply.turn_index : null;
      if (current.version !== state.version || (current.newest_reply ?? null) !== replyShown) {
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
              ? `${plural(k.cuts - previousCuts, "collection")} just ran: older messages left the young generation.`
              : "A collection just ran: older messages left the young generation.");
            celebrateCleanup();
          }
        }
      }
      state.activity = await getJSON(sessionPath("/activity"));
      // The measured counts and the window's source change with every request, not only with the conversation.
      if (state.knowledge) { renderStatus(); renderLastCollection(); renderSize(state.knowledge); renderWindow(state.knowledge); renderNext(state.knowledge); renderGenerations(state.knowledge); }
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
    renderRunGcButton();
    setLive("live", `live · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
  } catch {
    setLive("down", "proxy unreachable");
  }
}

// ----------------------------------------------------------------- wiring

// The tab in the address wins, then the one last used; the Session tab otherwise.
const TABS = ["session", "conversation", "engine"];
const hashed = location.hash.slice(1);
ui.tab = TABS.includes(hashed) ? hashed : TABS.includes(storeGet("llmgc.tab")) ? storeGet("llmgc.tab") : "session";
$("tab-session").addEventListener("click", () => { showTab("session"); renderTrace(traceActions, true); });
$("tab-conversation").addEventListener("click", () => showTab("conversation"));
$("tab-engine").addEventListener("click", () => showTab("engine"));
$("view-timeline").addEventListener("click", () => setPromptView(false));
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
$("session-select").addEventListener("change", (e) => { pickConversation(e.target.value); resetConversation(e.target.value); refresh(); });
$("client-select").addEventListener("change", (e) => {
  state.clientSession = e.target.value; state.trace = null;
  ui.turnOpen.clear(); ui.openAgents.clear(); ui.showSide.clear();
  ui.followMain = true;
  resetConversation(null);
  refresh();
});
let searchTimer = null;
$("search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { ui.query = e.target.value; if (ui.query.trim()) ui.listOpen = true; renderMessages(); }, 120);
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
$("intro").hidden = storeGet("llmgc.intro") === "dismissed";
ui.listOpen = storeGet("llmgc.messages") !== "closed";
fill("glossary", GUIDE_TERMS.map((term) => [h("dt", {}, GLOSSARY[term][0]), h("dd", {}, GLOSSARY[term][1])]));
fill("guide-colours", ORDER.map((key) => h("div", {}, swatch(key), h("div", {}, h("b", {}, CATS[key].label), h("span", {}, ` — ${GLOSSARY[key][1]}`)))),
  h("div", {}, h("span", { class: "qdot", style: "margin:3px 2px 0" }), h("div", {}, h("b", {}, "Changes at the next collection"), h("span", {}, " — a dot on the message's bar, in the colour it will become. On the map, a bar's height shows how much of the message the model still has, and its width how long the message is."))));

refresh();
setInterval(refresh, 1000);
