// The LLM-GC at work view: the proxy's counters, every check, ratings and settings.

import { BECOMES, CHART_HEIGHT, CLASS_LABEL, CLASS_TO_CAT, STATUS, TRANSITION, speaker } from "./vocabulary.js";
import { $, agoTs, compact, fill, fmt, h, plural, ranges, setChildren, svg } from "./dom.js";
import { captured, keepLast, preview, state, ui } from "./state.js";
import { hideTooltip, info, placeTooltip, tipQuote, tooltipContent } from "./tooltip.js";

export function renderTiles() {
  const p = state.proxy;
  const k = state.knowledge;
  const passes = state.activity ? state.activity.recent_passes : [];
  const lastCut = [...passes].reverse().find((x) => x.gc_status === "completed" && x.tokens_saved > 0);
  const tile = (label, term, value, caption, extra) => h("div", { class: "card tile" },
    h("div", { class: "stat-label" }, label, term ? info(term) : null), h("div", { class: "value" }, value),
    extra || null, caption ? h("div", { class: "caption" }, caption) : null);
  fill("tiles",
    tile("Requests through the proxy", null, p ? fmt(p.requests) : "—",
      p ? [`${fmt(p.gc_applied)} handled by LLM-GC`, p.gc_bypassed ? [" · ", h("span", { class: "st warn" }, "⚠ "), `${fmt(p.gc_bypassed)} passed on untouched after an error`] : " · none passed on after an error"]
        : "no proxy beside this dashboard"),
    tile("Tokens sent to the provider", "forwarded", p ? fmt(p.tokens_forwarded) : "—",
      p ? `${p.ratio !== null ? `${Math.round(p.ratio * 100)}% of` : "of"} ${fmt(p.tokens_in_client_history)} your apps sent, all conversations` : null,
      p && p.ratio !== null ? h("div", { class: "meter" }, h("div", { class: "some", style: `width:${Math.min(100, p.ratio * 100)}%` })) : null),
    tile("Clean-ups in this conversation", "cleanup", k ? fmt(k.cuts) : "—",
      lastCut ? `last ${agoTs(lastCut.gc_completed_at)}: ${fmt(lastCut.tokens_before)} → ${fmt(lastCut.tokens_before - lastCut.tokens_saved)} tokens` : "none in the recent checks"),
    tile("Conversation tracking", "tracking", null, null,
      h("div", { class: "triple" }, [[p ? p.sessions_seeded : "—", "started fresh"], [p ? p.history_rewinds : "—", "rewound after an edit"], [p ? p.history_restarts : "—", "restarted"]]
        .map(([n, label]) => h("div", {}, h("b", {}, typeof n === "number" ? fmt(n) : n), h("span", {}, label))))));
}

export function statusChip(status) {
  const s = STATUS[status] || { tone: "", icon: "?", text: status };
  return h("span", { class: "status-chip" }, h("span", { class: `dot ${s.tone}` }), `${s.icon} ${s.text}`);
}

// The conversation as written up to a turn: a running sum of the transcript's own counts.
export function writtenUpTo() {
  if (!captured()) return null;
  const turns = [...state.transcript.turns].sort((a, b) => a.turn_index - b.turn_index);
  const sums = new Map();
  let total = 0;
  for (const t of turns) { total += t.tokens; sums.set(t.turn_index, total); }
  return (turnIndex) => sums.has(turnIndex) ? sums.get(turnIndex) : null;
}

export function niceStep(max, count) {
  const raw = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  return Math.max(1, (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * magnitude);
}

export function renderChart() {
  const host = $("chart");
  const passes = state.activity ? state.activity.recent_passes : [];
  const width = Math.round(host.clientWidth);
  const key = `${width}|${passes.length}|${passes.at(-1)?.gc_completed_at}|${state.version}|${Math.floor(Date.now() / 30000)}`;
  if (!width || key === ui.chartKey) return;
  ui.chartKey = key;
  if (!passes.length) {
    fill("chart-legend");
    setChildren(host, h("p", { class: "empty" }, "No checks yet. LLM-GC checks after every reply."));
    return;
  }
  const writtenAt = writtenUpTo();
  const pts = passes.map((p, i) => ({ i, p, before: p.tokens_before,
    after: p.gc_status === "completed" ? p.tokens_before - p.tokens_saved : p.tokens_before,
    written: writtenAt && p.newest_turn_index !== null && p.newest_turn_index !== undefined ? writtenAt(p.newest_turn_index) : null }));
  const withWritten = pts.some((d) => d.written !== null);
  const cuts = pts.filter((d) => d.after < d.before);
  const skipped = pts.filter((d) => d.p.abandoned_cut);
  const failed = pts.filter((d) => d.p.gc_status === "bypassed_on_error");

  fill("chart-legend",
    h("span", {}, h("span", { class: "line-key", style: "background:var(--accent)" }), "Sent to the model"),
    withWritten ? h("span", {}, h("span", { class: "line-key", style: "background:var(--written)" }), "The conversation as written") : null,
    // Each mark in the legend is the chart's own marker.
    cuts.length ? h("span", {}, svg("svg", { width: 12, height: 12, viewBox: "0 0 12 12" }, svg("circle", { cx: 6, cy: 6, r: 4.5, fill: "var(--accent)" })), "clean-up") : null,
    skipped.length ? h("span", {}, svg("svg", { width: 12, height: 12, viewBox: "0 0 12 12" }, svg("circle", { cx: 6, cy: 6, r: 4, fill: "var(--surface)", stroke: "var(--ink-3)", "stroke-width": 1.5 })), "skipped: would not have shrunk the prompt") : null,
    failed.length ? h("span", {}, h("span", { class: "dot critical" }), "failed") : null);

  const narrow = width < 520;
  const M = { l: 48, r: narrow ? 14 : 112, t: 16, b: 30 };
  const pw = Math.max(40, width - M.l - M.r), ph = CHART_HEIGHT - M.t - M.b;
  const maxValue = Math.max(1, ...pts.map((d) => Math.max(d.before, d.written ?? 0)));
  const step = niceStep(maxValue, 4);
  const top = Math.ceil(maxValue / step) * step;
  const x = (i) => M.l + (pts.length === 1 ? pw / 2 : (i / (pts.length - 1)) * pw);
  const y = (v) => M.t + ph - (v / top) * ph;

  const grid = [];
  for (let v = 0; v <= top + 1e-9; v += step) {
    grid.push(svg("line", { class: v === 0 ? "axis" : "grid", x1: M.l, x2: M.l + pw, y1: y(v), y2: y(v) }),
      svg("text", { class: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end" }, compact(v)));
  }
  let sentPath = "";
  pts.forEach((d, n) => { sentPath += `${n ? "L" : "M"}${x(d.i)},${y(d.before)}`; if (d.after < d.before) sentPath += `L${x(d.i)},${y(d.after)}`; });
  const areaPath = `${sentPath}L${x(pts.at(-1).i)},${y(0)}L${x(0)},${y(0)}Z`;
  const writtenPts = pts.filter((d) => d.written !== null);
  const writtenPath = writtenPts.map((d, n) => `${n ? "L" : "M"}${x(d.i)},${y(d.written)}`).join("");

  const last = pts.at(-1);
  const ends = [];
  if (!narrow) {
    ends.push(svg("text", { class: "end", x: M.l + pw + 10, y: y(last.after) + 4 }, `sent ${fmt(last.after)}`));
    if (last.written !== null && Math.abs(y(last.written) - y(last.after)) >= 16) {
      ends.push(svg("text", { class: "end", x: M.l + pw + 10, y: y(last.written) + 4, style: "fill:var(--ink-3)" }, `written ${fmt(last.written)}`));
    }
  }
  const cross = svg("line", { class: "cross", x1: 0, x2: 0, y1: M.t, y2: M.t + ph, style: "display:none" });
  const root = svg("svg", { height: CHART_HEIGHT, viewBox: `0 0 ${width} ${CHART_HEIGHT}`, role: "img",
    "aria-label": `Prompt size over the last ${passes.length} checks: now ${fmt(last.after)} tokens sent${last.written !== null ? `, ${fmt(last.written)} as written` : ""}, ${plural(cuts.length, "clean-up")}.` },
    grid,
    svg("path", { class: "a-sent", d: areaPath }),
    writtenPath ? svg("path", { class: "l-written", d: writtenPath }) : null,
    svg("path", { class: "l-sent", d: sentPath }),
    cuts.map((d) => [svg("circle", { class: "mk-cut", cx: x(d.i), cy: y(d.after), r: 5 }),
      cuts.length <= 8 ? svg("text", { class: "cut-label", x: x(d.i) + 7, y: y(d.after) + 15 }, `−${fmt(d.before - d.after)}`) : null]),
    skipped.map((d) => svg("circle", { class: "mk-skip", cx: x(d.i), cy: y(d.before), r: 4.5 })),
    failed.map((d) => svg("circle", { class: "mk-error", cx: x(d.i), cy: y(d.before), r: 5 })),
    ends,
    svg("text", { class: "tick", x: M.l, y: CHART_HEIGHT - 8 }, agoTs(pts[0].p.gc_completed_at)),
    svg("text", { class: "tick", x: M.l + pw, y: CHART_HEIGHT - 8, "text-anchor": "end" }, `latest · ${plural(passes.length, "check")}`),
    cross);
  // One hit layer for the whole plot: the pointer snaps to the nearest check.
  const hit = svg("rect", { class: "hit", x: M.l - 8, y: M.t, width: pw + 16, height: ph });
  hit.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const scaleX = box.width / width;
    const at = (event.clientX - box.left) / scaleX;
    const i = Math.max(0, Math.min(pts.length - 1, Math.round(pts.length === 1 ? 0 : ((at - M.l) / pw) * (pts.length - 1))));
    const d = pts[i];
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.style.display = "";
    const figure = (label, value) => h("div", { class: "tip-figure" }, h("span", {}, label), h("b", {}, value));
    tooltipContent(`Check ${i + 1} of ${pts.length} · ${agoTs(d.p.gc_completed_at)}`, [
      statusChip(d.p.gc_status),
      figure("sent to the model", d.after < d.before ? `${fmt(d.before)} → ${fmt(d.after)}` : fmt(d.before)),
      d.written !== null ? figure("as written", fmt(d.written)) : null,
      d.p.abandoned_cut ? figure("a clean-up would have made it", fmt(d.p.abandoned_cut.context_tokens_after)) : null,
      d.after < d.before ? figure("summarised · facts", `${d.p.compact_count} · ${d.p.archive_count}`) : null,
      d.p.degradations.length ? h("div", { class: "tip-row" }, h("span", { class: "st warn" }, "⚠ "), plural(d.p.degradations.length, "problem")) : null,
    ]);
    const cx = box.left + x(i) * scaleX;
    placeTooltip({ left: cx + 14, right: cx + 14, top: box.top + M.t, bottom: box.top + M.t + 40 });
  });
  hit.addEventListener("pointerleave", () => { cross.style.display = "none"; hideTooltip(); });
  root.appendChild(hit);
  setChildren(host, root);
}

export function renderLastCheck() {
  const pass = state.activity && state.activity.gc_breakdown;
  fill("last-check", pass ? h("div", { style: "text-align:right" }, h("div", { class: "eyebrow", style: "display:flex;gap:6px;justify-content:flex-end;align-items:center" }, "Last check", info("check")),
    h("div", { style: "margin-top:4px" }, statusChip(pass.gc_status)),
    h("div", { class: "note", style: "margin-top:2px" }, [agoTs(pass.gc_completed_at), `${pass.duration_ms.toFixed(0)} ms`,
      pass.failure_reason ? `${pass.failure_stage}: ${pass.failure_reason}` : null].filter(Boolean).join(" · "))) : null);
}

// ----------------------------------------------------------------- the engine: relevance and settings

export function renderRelevance(k) {
  const c = state.config;
  const a = c ? c.archive_threshold : 0.3, keep = c ? c.keep_threshold : 0.7;
  fill("rel-title", "How the next clean-up rates each recent message", info("relevance"));
  fill("rel-sub", `Each dot is a message's relevance. Below ${a} it would be reduced to facts, from ${keep} quoted whole, and in between summarised. A ring means a rule keeps it word for word whatever its score.`);
  const rows = k.turns.filter((t) => t.fidelity === "verbatim" && t.score !== null);
  const plot = $("relevance");
  if (!rows.length) { setChildren(plot, h("p", { class: "empty" }, "No recent messages to rate.")); return; }
  plot.style.setProperty("--a", `${a * 100}%`);
  plot.style.setProperty("--k", `${keep * 100}%`);
  const why = (t) => !t.forced_keep ? CLASS_LABEL[t.classification]
    : t.forced_keep.includes("too short") ? "kept: too short to condense"
    : t.forced_keep.includes("last_n") ? `kept: among the newest ${keepLast()}` : "kept: by a rule";
  setChildren(plot,
    h("div", { class: "rel-row rel-axis", "aria-hidden": "true" }, h("div"),
      h("div", { class: "rel-track" },
        h("span", { class: "zone", style: `left:${(a / 2) * 100}%` }, `facts · < ${a}`),
        h("span", { class: "zone", style: `left:${((a + keep) / 2) * 100}%` }, "summarise"),
        h("span", { class: "zone", style: `left:${((keep + 1) / 2) * 100}%` }, `quote · ≥ ${keep}`)), h("div")),
    rows.map((t) => h("div", { class: "rel-row" },
      h("div", { class: "rel-lab num" }, `#${t.turn_index} · ${speaker(t.role)}`),
      h("div", { class: "rel-track", ...hoverTip(`Message ${t.turn_index}: relevance ${t.score.toFixed(2)}`,
          t.forced_keep ? why(t) : `rated to ${CLASS_LABEL[t.classification]}`,
          t.becomes ? `the next clean-up: ${BECOMES[t.becomes]}` : "not part of the next clean-up", tipQuote(preview(t.text, 120))) },
        h("span", { class: `rel-dot ${t.forced_keep ? "forced" : `f-${CLASS_TO_CAT[t.classification]}`}`, style: `left:${t.score * 100}%` })),
      h("div", { class: "rel-why" }, h("div", {}, h("b", {}, t.score.toFixed(2)), ` · ${why(t)}`),
        t.becomes ? h("div", { class: "then" }, `→ next clean-up: ${BECOMES[t.becomes]}`) : null))),
    h("div", { class: "rel-legend" },
      h("span", {}, h("span", { class: "rel-dot f-verbatim" }), "quote whole"),
      h("span", {}, h("span", { class: "rel-dot f-summarised" }), "summarise"),
      h("span", {}, h("span", { class: "rel-dot f-facts" }), "reduce to facts"),
      h("span", {}, h("span", { class: "rel-dot forced" }), "kept word for word by a rule")));
}

export function renderSettings() {
  const c = state.config;
  if (!c) return;
  // The flag is how to change a setting while watching here; the rest are GCConfig fields.
  const rows = [
    ["Clean-up size", "min_cut_tokens", c.min_cut_tokens, "tokens ready to condense before a clean-up runs", "--min-cut-tokens"],
    ["Always word for word", "last_n_turns_to_keep", c.last_n_turns_to_keep, "newest messages never condensed", "--keep-last"],
    ["Quote whole from", "keep_threshold", c.keep_threshold, "relevance at or above this is kept word for word"],
    ["Facts below", "archive_threshold", c.archive_threshold, "relevance below this is reduced to facts"],
    ["Too short to condense", "min_compactable_tokens", c.min_compactable_tokens, "messages this short, in tokens, are kept whole"],
    ["Facts per block", "max_memory_facts", c.max_memory_facts, "most facts the condensed block carries"],
    ["Fact budget", "max_memory_tokens", c.max_memory_tokens, "most tokens of facts in the condensed block"],
    ["The model's window", "context_window", c.context_window, "its capacity, in tokens", "--context-window"],
  ];
  fill("settings", rows.map(([label, name, value, desc, flag]) => h("div", { class: "setting" },
    h("div", { class: "name" }, label, h("code", {}, name)), h("div", { class: "value" }, fmt(value)),
    h("div", { class: "desc" }, desc, flag ? h("div", {}, "change with ", h("code", { class: "flag" }, `run_proxy.py ${flag}`)) : null))));
}

// ----------------------------------------------------------------- the engine: checks, problems, history

export function renderChecks() {
  const passes = state.activity ? state.activity.recent_passes.slice().reverse() : [];
  // Consecutive checks that found nothing to do are one row: they are the normal case.
  const groups = [];
  for (const p of passes) {
    const quiet = p.gc_status !== "completed" && p.gc_status !== "bypassed_on_error" && !p.abandoned_cut && !p.degradations.length;
    const lastGroup = groups.at(-1);
    if (quiet && lastGroup && lastGroup.quiet && lastGroup.status === p.gc_status) lastGroup.items.push(p);
    else groups.push({ quiet, status: p.gc_status, items: [p] });
  }
  setChildren($("checks"),
    h("thead", {}, h("tr", {}, ["When", "Result", "Prompt, tokens", "Took"].map((title) => h("th", {}, title)))),
    h("tbody", {}, groups.length ? groups.map((g) => {
      const p = g.items[0], oldest = g.items.at(-1);
      const cut = p.gc_status === "completed" && p.tokens_saved > 0;
      const durations = g.items.map((x) => x.duration_ms);
      const span = (a, b) => a === b ? a : `${a} – ${b}`;
      const took = span(`${Math.min(...durations).toFixed(0)}`, `${Math.max(...durations).toFixed(0)}`);
      return h("tr", { class: cut ? "cut" : p.gc_status === "completed" ? "" : "quiet" },
        h("td", { class: "num", style: "white-space:nowrap" }, span(agoTs(oldest.gc_completed_at), agoTs(p.gc_completed_at))),
        h("td", {}, statusChip(p.gc_status), g.items.length > 1 ? h("span", { class: "times" }, `×${g.items.length}`) : null,
          cut ? h("div", { class: "muted", style: "font-weight:400;font-size:12.5px" }, `${p.compact_count} summarised · ${p.archive_count} reduced to facts`) : null,
          p.degradations.length ? h("div", { style: "font-weight:400" }, h("span", { class: "st warn" }, "⚠ "), plural(p.degradations.length, "problem")) : null),
        h("td", { class: "num", style: "white-space:nowrap" }, cut ? `${fmt(p.tokens_before)} → ${fmt(p.tokens_before - p.tokens_saved)}`
          : p.abandoned_cut ? [fmt(p.tokens_before), h("div", { class: "muted", style: "font-size:12px" }, `would have been ${fmt(p.abandoned_cut.context_tokens_after)}`)]
          : span(fmt(oldest.tokens_before), fmt(p.tokens_before))),
        h("td", { class: "num", style: "white-space:nowrap" }, `${took} ms`));
    }) : h("tr", {}, h("td", { colspan: 4, class: "empty" }, "No checks yet."))));
}

export function renderProblems() {
  const problems = state.activity ? state.activity.recent_degradations.slice().reverse() : [];
  fill("problems", problems.length ? problems.map((r) => h("li", {},
    h("div", { class: "event-head" }, h("span", { class: "st warn" }, "⚠"), h("b", {}, r.degradation.reason.replaceAll("_", " ")), h("span", { class: "when" }, agoTs(r.occurred_at))),
    h("div", { class: "event-lines" }, `Messages ${ranges(r.degradation.turn_indices)} stayed word for word. ${r.degradation.detail}`)))
    : h("li", { class: "empty" }, h("span", { class: "st good" }, "✓ "), "None: every clean-up did what it set out to."));
}

export function renderHistory() {
  const transitions = state.activity ? state.activity.generation_lifecycle.recent_transitions : [];
  // Moves made by one clean-up arrive together; group them by time.
  const groups = [];
  for (const t of transitions) {
    const lastGroup = groups.at(-1);
    if (lastGroup && t.occurred_at - lastGroup.at < 2) lastGroup.items.push(t);
    else groups.push({ at: t.occurred_at, items: [t] });
  }
  fill("history", groups.length ? groups.reverse().map((g) => {
    const byType = {};
    for (const t of g.items) (byType[t.transition_type] ||= []).push(t);
    return h("li", {},
      h("div", { class: "event-head" }, h("b", {}, "Clean-up"), h("span", { class: "when" }, agoTs(g.at))),
      h("div", { class: "event-lines" }, Object.entries(byType).map(([type, items]) => h("div", {},
        h("span", { class: "k" }, TRANSITION[type] || type),
        type === "superseded" ? items.map((t) => `“${t.topic_label}” from #${t.turn_index}`).join(", ") : `messages ${ranges(items.map((t) => t.turn_index))}`))));
  }) : h("li", { class: "empty" }, "Nothing has moved yet."));
}
