// Entry point: polling, the session picker, and wiring the controls.

import { CATS, GLOSSARY, GUIDE_TERMS, ORDER, swatch } from "./vocabulary.js";
import { $, clearMotionCaches, fill, h, plural, setChildren, storeGet, storeSet } from "./dom.js";
import { getJSON, sessionPath, state, ui } from "./state.js";
import { hideTooltip } from "./tooltip.js";
import { THEMES, applyTheme, setGuide, setLive, showCleanupNotice, showTab } from "./shell.js";
import { clearMapCache, renderAttention, renderChips, renderHero, renderHeroMeta, renderMemory, renderMessages, renderPrompt, renderTimeline, setPromptView } from "./conversation.js";
import { renderChart, renderChecks, renderHistory, renderLastCheck, renderProblems, renderRelevance, renderSettings, renderTiles } from "./engine.js";
import { runGcNow } from "./run-gc.js";

function renderSessions() {
  const select = $("session-select");
  const options = state.sessions.map((s) => h("option", { value: s.session_id, title: s.session_id },
    `${s.title ? s.title : s.session_id} · ${plural(s.recorded_turns, "message")} · ${plural(s.cuts, "clean-up")}${s.aging_in_flight ? " · cleaning up…" : ""}`));
  setChildren(select, options.length ? options : h("option", {}, "none yet"));
  select.disabled = !options.length;
  if (state.selected) select.value = state.selected;
  const button = $("run-gc");
  button.disabled = !state.selected || button.dataset.running === "1";
}

function renderConversation() {
  const k = state.knowledge;
  hideTooltip();
  renderHero(k);
  renderChips(k);
  if (ui.promptView) renderPrompt(k); else renderTimeline(k);
  renderMemory(k);
  renderRelevance(k);
}

async function refresh() {
  try {
    const overview = await getJSON("/api/overview");
    state.config = overview.config;
    state.proxy = overview.proxy;
    state.sessions = overview.sessions;
    if (!state.sessions.some((s) => s.session_id === state.selected)) {
      state.selected = state.sessions.length ? state.sessions[0].session_id : null;
      state.version = null;
      state.knowledge = null;
      ui.open.clear();
      ui.earlierOpen = false;
      clearMapCache();
      clearMotionCaches();
    }
    renderSessions();
    renderSettings();
    $("no-session").hidden = !!state.selected;
    showTab(ui.tab);

    if (state.selected) {
      const current = state.sessions.find((s) => s.session_id === state.selected);
      // Content changes only with the conversation's version; activity is cheap and polled every time.
      if (current.version !== state.version) {
        const previousCuts = state.knowledge ? state.knowledge.cuts : null;
        const [k, transcript] = await Promise.all([getJSON(sessionPath()), getJSON(sessionPath("/transcript"))]);
        if (k) {
          state.knowledge = k;
          state.transcript = transcript;
          state.version = k.version;
          renderConversation();
          if (previousCuts !== null && k.cuts > previousCuts) {
            showCleanupNotice(k.cuts - previousCuts > 1
              ? `${plural(k.cuts - previousCuts, "clean-up")} just ran — older messages condensed.`
              : "A clean-up just ran — older messages condensed.");
          }
        }
      }
      state.activity = await getJSON(sessionPath("/activity"));
      if (state.knowledge) { renderHeroMeta(state.knowledge); renderAttention(); }
      renderTiles();
      renderLastCheck();
      renderChart();
      renderChecks();
      renderProblems();
      renderHistory();
    }
    setLive("live", `live · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
  } catch {
    setLive("down", "proxy unreachable");
  }
}

// ----------------------------------------------------------------- wiring

ui.tab = (location.hash === "#engine" || (location.hash !== "#conversation" && storeGet("llmgc.tab") === "engine")) ? "engine" : "conversation";
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
$("session-select").addEventListener("change", (e) => {
  state.selected = e.target.value; state.version = null; state.knowledge = null;
  ui.open.clear(); ui.filter = "all"; ui.query = ""; $("search").value = ""; ui.earlierOpen = false;
  clearMapCache();
  clearMotionCaches();
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

applyTheme(storeGet("llmgc.theme") || "system");
$("intro").hidden = storeGet("llmgc.intro") === "dismissed";
ui.listOpen = storeGet("llmgc.messages") !== "closed";
fill("glossary", GUIDE_TERMS.map((term) => [h("dt", {}, GLOSSARY[term][0]), h("dd", {}, GLOSSARY[term][1])]));
fill("guide-colours", ORDER.map((key) => h("div", {}, swatch(key), h("div", {}, h("b", {}, CATS[key].label), h("span", {}, ` — ${GLOSSARY[key][1]}`)))),
  h("div", {}, h("span", { class: "swatch will-swatch" }), h("div", {}, h("b", {}, "Taken by the next clean-up"), h("span", {}, " — the band at the foot is what the message will become."))));

refresh();
setInterval(refresh, 1000);
