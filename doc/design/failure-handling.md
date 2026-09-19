# Failure Handling — The Circuit Breaker

> **Not yet built.** This doc records the intended design so the decision is
> settled before the code exists. It describes what the breaker will do, not
> what it does today.

## How it works

The breaker is a **per-session three-state machine** wrapping `gc.collect`.

```
closed     ──[N consecutive failures within window]──►  open
open       ──[cooldown elapsed]──────────────────────►  half-open
half-open  ──[probe succeeds]────────────────────────►  closed
half-open  ──[probe fails]───────────────────────────►  open (cooldown resets)
```

Definitions:

- **Failure** — `gc.collect` exceeds `collect_budget_ms` (default 200), or an
  unhandled exception is caught at the orchestrator boundary.
- **N** — `failure_threshold`, default 3.
- **Window** — `failure_window_ms`, default 60000. Failures older than the
  window stop counting toward N.
- **Cooldown** — `open_cooldown_ms`, default 30000.
- **Probe** — the first `gc.collect` after the cooldown runs the full pipeline
  and the breaker watches the outcome.

Behaviour by state:

- **closed** — the pipeline runs. Failures are counted; the Nth in-window
  failure opens the breaker and emits `CIRCUIT_BREAKER_OPENED`.
- **open** — `gc.collect` short-circuits, returning the original `messages` with
  `GCResult.bypassed = True` and emitting `CIRCUIT_BREAKER_BYPASS`. The pipeline
  is not invoked. The caller sees normal API behaviour, just no GC.
- **half-open** — the next collect runs as a probe. Other concurrent collects on
  the same session keep bypassing while it is in flight, so they cannot pile in
  and corrupt the test.

`gc.update` failures do **not** feed the breaker. They emit `UPDATE_FAILED` and
show in the dashboard, but they do not trip it.

Breaker state is in-process and per-session, and is not persisted across
restarts.

## Why it exists

The *never make things worse* principle says the system must not degrade the
experience below the baseline of not running LLM-GC at all. A hard budget and
exception catching get part of the way, but without an open state the engine
retries a failing pipeline on every single call — paying the full budget each
time for the same outcome. The user gets consistent added latency and no
working GC until the underlying cause resolves on its own. The open state
amortises the cost of failure across the cooldown.

Writing the state machine down is itself the point. Named states and named
transitions are testable; "catch exceptions and hope" drifts into ad-hoc retry
loops scattered through the engine, with no way to prove the behaviour the
principle demands.

## Why this shape

- **Per-session, not global.** A bug triggered by one session's data — a
  malformed permanent-gen entry, a strange embedding — would trip a global
  breaker for every user. Keying by session contains the blast radius to the
  session that actually has the problem. Rejected alternative: one breaker for
  the engine.
- **Collect trips it; update does not.** The update path's failure modes are LLM
  timeouts and network blips, and none of them predict whether the next collect
  will succeed — collect reads a cached plan and is independent of whether the
  last update finished. Tripping on update failures would bypass collects for no
  reason. Rejected alternative: count both paths.
- **Fixed cooldown, not exponential backoff.** Deferred rather than rejected:
  a fixed cooldown is easier to reason about, and the data needed to choose good
  backoff parameters does not exist yet. Revisit if flapping turns out to be
  real.
- **State dies with the process.** A restart is a meaningful event — config,
  dependencies and state may all have changed. Carrying breaker state across it
  risks bypassing long after the cause was repaired. Rejected alternative:
  persist breaker state.
- **Every transition emits an event.** A breaker that opens quietly is exactly
  the failure this project exists to prevent. The dashboard shows the current
  state per session and the post-session report shows the transitions over the
  session's life, so "GC was off, for this long, because of this" is visible
  rather than implicit.

## Consequences

The engine carries one state machine per active session. The cost is small — a
counter, a window of timestamps, a state — but it is real, and it is one more
thing tied to the session lifecycle in [session/overview.md](session/overview.md).
The breaker is released when the session is.

A session with a permanent rather than transient problem cycles between
half-open and open until it ends or the cause is fixed externally. That is the
intended behaviour: "the issue might be fixed" is a real possibility, and the
cost of periodic probes is bounded by the cooldown.

The breaker moves a correctness obligation from the caller to the engine.
Callers never handle GC errors — every `gc.collect` returns either a clean
result or a bypassed one with the original `messages` intact. That is what makes
"GC is best-effort" a contract instead of a hope, and it is a real
simplification for the proxy and the SDK wrappers.
