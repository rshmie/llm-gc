// What the endpoints last returned, what the viewer has chosen, and the
// grouping and search over them.

import { ORDER, categoryOf } from "./vocabulary.js";
import { $, h } from "./dom.js";

export const state = { config: null, proxy: null, sessions: [], selected: null, version: null, knowledge: null, transcript: null, activity: null };
export const ui = { tab: "conversation", filter: "all", query: "", promptView: false, open: new Set(), chartKey: null, earlierOpen: false, listOpen: true,
  stayOpen: false, allChecks: false, allMoves: false, allProblems: false };

export const keepLast = () => (state.config ? state.config.last_n_turns_to_keep : 5);
// What bounds the facts the recap carries: its token budget, a share of the model's window.
export const recapLimit = () => !state.config ? "a limited share of the window in facts"
  : `${Number(state.config.max_recap_tokens).toLocaleString()} tokens`;
export const captured = () => !!(state.transcript && state.transcript.captured);
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

export function saidByTurn() { return new Map((captured() ? state.transcript.turns : []).map((t) => [t.turn_index, t])); }
export function factsByTurn(k) {
  const byTurn = new Map();
  for (const fact of k.facts) for (const turn of fact.source_turns) {
    if (!byTurn.has(turn)) byTurn.set(turn, []);
    byTurn.get(turn).push(fact);
  }
  return byTurn;
}
export function countCats(k) {
  const counts = Object.fromEntries(ORDER.map((key) => [key, 0]));
  for (const turn of k.turns) counts[categoryOf(turn)] += 1;
  return counts;
}
// Messages the client has sent that the conversation records only with the next request.
export const pendingTurns = (k) => captured() ? state.transcript.turns.filter((t) => t.turn_index >= k.recorded_turns) : [];
export function turnMatches(turn, said, facts) {
  if (!query()) return true;
  return has(said.get(turn.turn_index)?.text) || has(turn.text) || (facts.get(turn.turn_index) || []).some((f) => has(f.content) || has(f.topic));
}
