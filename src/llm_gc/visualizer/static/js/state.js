// What the endpoints last returned, what the viewer has chosen, and the
// grouping and search over them.

import { ORDER, categoryOf } from "./vocabulary.js";
import { $, h, storeGet } from "./dom.js";

// `selected` is the conversation picked at the top, shown on the Conversation and Engine tabs, and `clientSession`
// the Claude Code session it ran in, whose requests the Requests tab shows - or OTHER for conversations no client
// session named. A session whose conversations have all closed can be picked for its requests alone.
export const OTHER = "__other__";
export const state = { config: null, proxy: null, pause: null, sessions: [], clientSession: null, trace: null, selected: null, version: null,
  knowledge: null, transcript: null, activity: null,
  // The shown conversation's cache as the proxy reads it now, from the overview: it ages every second.
  cache: null };
// Where the viewer's choice of list order is kept. Newest first unless they chose otherwise: what they open the
// dashboard to check is what just happened, and the recap that leads the other order is the part read least.
export const LIST_ORDER_KEY = "llmgc.order";
export const ui = { tab: "conversation", filter: "all", query: "", promptView: false, open: new Set(), chartKey: null, earlierOpen: false,
  newestFirst: storeGet(LIST_ORDER_KEY) !== "oldest",
  stayOpen: false, allChecks: false, allMoves: false, allExchanges: false, allProblems: false, mapSession: null,
  turnOpen: new Map(), openAgents: new Set(), showSide: new Set(), pendingJump: null, turnsEarlierOpen: false,
  // Whether the shown conversation follows the session's main one: true until the person picks another.
  followMain: true };

export const keepLast = () => (state.config ? state.config.last_n_turns_to_keep : 5);
// What the recap may spend on older exchanges: its token budget, a share of the window of the model being called.
export const recapLimit = () => state.knowledge ? `${Number(state.knowledge.recap_budget).toLocaleString()} tokens`
  : !state.config ? "a share of the model's window" : `${Number(state.config.max_recap_tokens).toLocaleString()} tokens`;
// The proxy's measured traffic for one conversation, and for the client session it ran in; null without a proxy.
export function trafficOf(sessionId) {
  if (!state.proxy) return null;
  const all = [...state.proxy.client_sessions.flatMap((c) => c.conversations), ...state.proxy.other_conversations];
  return all.find((c) => c.conversation_id === sessionId) || null;
}
export function clientSessionOf(clientSession) {
  if (!state.proxy) return null;
  return state.proxy.client_sessions.find((c) => c.client_session === clientSession) || null;
}
export const captured = () => !!(state.transcript && state.transcript.captured);
// Read from the session list, polled every time: the conversation itself is refetched only when it changes,
// and a collection starts and runs without changing it.
export const collecting = () => !!state.sessions.find((s) => s.session_id === state.selected && s.aging_in_flight);
// The upstream's refusal of LLM-GC's own credential, while it still stands: the newest refusal that no
// later collection has got past. `error_code` is the engine's, set when a summary or a facts read is
// refused with a 401 or 403; the credential is read once at startup, so it persists until a restart.
// A later collection that finished with nothing falling short means a call went through, and clears it.
export const CREDENTIAL_REFUSED = "LLM_AUTH_FAILED";
export function credentialRefusal() {
  if (!state.activity) return null;
  const refusals = state.activity.recent_degradations.filter((r) => r.degradation.error_code === CREDENTIAL_REFUSED);
  if (!refusals.length) return null;
  const latest = refusals.at(-1);
  const recovered = state.activity.recent_passes.some((p) => p.gc_status === "completed" && !p.degradations.length
    && p.gc_completed_at > latest.occurred_at);
  return recovered ? null : latest;
}
export const preview = (text, n = 150) => !text ? null : text.length > n ? `“${text.slice(0, n).trim()}…”` : `“${text}”`;

export async function getJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}
export const sessionPath = (suffix = "") => `/api/sessions/${encodeURIComponent(state.selected)}${suffix}`;

export const query = () => ui.query.trim().toLowerCase();
export const has = (text) => !!text && text.toLowerCase().includes(query());
// The text with every match wrapped in <mark>, as nodes.
export function marked(text) {
  const needle = query();
  if (!needle || !text) return [text];
  const out = [];
  const lower = text.toLowerCase();
  let at = 0;
  for (let i = lower.indexOf(needle); i !== -1; i = lower.indexOf(needle, at)) {
    out.push(text.slice(at, i), h("mark", {}, text.slice(i, i + needle.length)));
    at = i + needle.length;
  }
  out.push(text.slice(at));
  return out;
}

// ----------------------------------------------------------------- grouping what the endpoints gave

export function factsByTurn(k) {
  const byTurn = new Map();
  for (const fact of k.facts) {
    if (!byTurn.has(fact.turn)) byTurn.set(fact.turn, []);
    byTurn.get(fact.turn).push(fact);
  }
  return byTurn;
}
export function countCats(k) {
  const counts = Object.fromEntries(ORDER.map((key) => [key, 0]));
  for (const turn of k.turns) counts[categoryOf(turn)] += 1;
  return counts;
}
// Messages the client has sent that the conversation records only with the next request: the one
// thing the dashboard's own copy of what was said is still needed for.
// The reply to the newest request, read as it reached the client: not in the conversation until a later request carries it.
export const newestReply = () => (captured() && state.transcript.reply) || null;
export const pendingTurns = (k) => captured() ? state.transcript.turns.filter((t) => t.turn_index >= k.recorded_turns) : [];
// The instructions the client puts inside a message as it sends them now: it can rewrite them after
// the message was recorded, so the dashboard's copy of the latest request is the one to read.
export function instructionTokens(turnIndex) {
  const turn = captured() ? state.transcript.turns[turnIndex] : null;
  return turn && turn.turn_index === turnIndex ? turn.instruction_tokens : 0;
}
export function turnMatches(turn, facts) {
  if (!query()) return true;
  return has(turn.text) || (facts.get(turn.turn_index) || []).some((f) => has(f.content) || has(f.topic));
}
// The conversation as written: every recorded message at its own size, counted the way the engine counts.
export const writtenTokens = (k) => k.turns.reduce((sum, t) => sum + t.tokens, 0);
