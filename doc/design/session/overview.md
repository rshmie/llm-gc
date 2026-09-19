# Session Management

## How it works

A **session** is one continuous conversation's remembered state, keyed by a
`session_id`. It is what lets the engine carry state between turns instead of
being handed the whole message list every call.

Three components, in increasing responsibility:

- **`SessionState`** — the remembered state for one conversation: its
  `session_id`, its live list of young-generation `messages`, and the timestamps
  (`created_at`, `last_updated_at`) the idle sweeper reads. A plain mutable
  container, appended to and re-timestamped each turn.
- **`SessionStore`** — the registry of active sessions, keyed by id. Hands out a
  `SessionState` on demand, creating one on first sight of an id. In-memory for
  now; a file-backed store implementing the same small interface swaps in later.
- **`SessionManager`** — wraps the store and adds the concurrency contract and
  the lifecycle. All access to one session's state happens inside `async with
  acquire(session_id)`, which holds a per-session `asyncio.Lock` for the whole
  read-modify-write. It also owns session end: `close_session` drops a session's
  state (keeping its lock), and `run_idle_sweeper` is a background task that
  closes sessions gone quiet past the idle timeout.

The two engine operations that use a session are the ager (`gc.update`) and the
assembler (`gc.collect`) — the hot/cold split described in
[generational-memory/overview.md](../generational-memory/overview.md). Both run
under the session's lock.

**The ager is the pass that reports.** `gc.update` is where the real work
happens — scoring, sweeping, and moving turns between generations — so it is
the operation that emits the terminal GC event, carrying both the run's result
and the session id it belongs to. The assembler emits nothing: it changes no
state, and a read announcing itself as a completed run would overwrite the last
real run's breakdown with a no-op. Both the result the ager reports as the
session's resulting context and the list the assembler returns come from a
single shared assembly step, so the number a dashboard shows and the payload a
caller forwards cannot drift apart. That context is measured whole —
old-generation summaries included — because those summaries are real tokens in
the real prompt, and a budget that counted only young turns would under-report
pressure in the reassuring direction.

**Aging is triggered by pressure, not by arrival.** Every update records the new
turn — that is bookkeeping — but the ager only scores, sweeps, and moves turns
when the assembled context has crossed the configured GC threshold. Below it the
pass reports itself bypassed and changes nothing. Aging is not free: it replaces
verbatim text with a summary, and once the compactor is a real LLM it also costs
a call per promotion. With plenty of window left there is nothing to buy with
that cost, so the right amount of aging is none. This is the same guard the
stateless collector already applied, measured on the same assembled context the
caller will actually send.

*Rejected: aging on every update.* Uniform per-pass work and no burst when the
threshold is crossed, but it compacted turns nobody needed compacted — observed
promoting a turn at 13% context pressure, discarding verbatim text with most of
the window still free, and (once the compactor is real) spending money to do it.

Aging failures are contained at two levels. A single turn that fails to age
(a compactor error, say) stays in the young generation for the next pass and
never leaves a summary standing in for an original that is also still live; the
rest of the pass continues. A failure in the mark stage, before anything has
moved, is reported as a bypassed run rather than raised — the ager is called
fire-and-forget, so an exception escaping it would be invisible to logs and
dashboard alike.

## Why it exists

Without sessions, the engine is stateless: every call re-derives everything from
a full message list, and there is nowhere to keep a session's old-gen summaries,
permanent-gen facts, or (later) a pre-computed plan between turns. A real proxy
serving many conversations at once needs each one's memory to persist across
requests — that persistence is the session.

The per-session lock exists because the ager and assembler both read state, do
work, and write state back, and that span contains `await` points. Under a
single event loop, two coroutines for the same session can interleave at an
`await` and lose each other's writes (a lost update). The lock closes that
window.

## Why this shape

- **Per-session lock, not one global lock.** Different conversations never touch
  each other's state, so serialising them behind one lock would cap throughput
  for no correctness gain. Keying the lock by `session_id` makes one session
  single-writer while letting unrelated sessions run fully in parallel. Rejected
  alternative: a single manager-wide lock — simpler, but it makes every session
  wait on every other.
- **Bounded wait, not infinite.** `acquire` waits at most
  `session_lock_timeout_ms` (default 100) for the lock, then raises
  `SessionLockTimeout` rather than blocking forever. A stuck holder (e.g. a
  hung LLM call inside an in-flight update) must not be able to freeze every
  later request on that session. On timeout the caller is expected to fall back
  to last-persisted state and surface the staleness, never to hang. Rejected
  alternative: block until acquired — one hung update takes the whole session
  down.
- **Close keeps the lock; it does not remove it (Approach 3).** `close_session`
  acquires the lock (draining an in-flight update up to `close_drain_ms`),
  removes the state from the store, and **leaves the lock object in the
  registry**. Every `acquire` for an id must return the *same* lock, so a
  request arriving during or after a close serialises on it. Two rejected
  alternatives, both racy: *pop the lock* — a coroutine waiting on the old lock
  and a newcomer that mints a fresh one would both run inside the session at
  once (the two-lock race); *acquire-then-pop* — same window survives for a
  coroutine that was already queued on the old lock. Keeping the lock stable is
  the only race-free option. Its cost is a lingering lock per closed session
  (one per id ever seen); that is an accepted v1.0 leak — bounded lock cleanup
  is a future refinement, not a correctness gap.
- **Session ends on idle timeout OR explicit close, and reopens on demand.** A
  background sweeper (`run_idle_sweeper`, every `sweep_interval_s`) closes
  sessions idle longer than `session_timeout_s`; a caller can also close one
  directly. Closing releases in-memory state but the next request for that id
  re-creates fresh state (a reopen). Rejected alternatives: *no idle timeout* —
  the active set and its memory grow without bound for the common case (a user
  walks away); *idle timeout only, no explicit close* — a library caller (a test,
  a benchmark) has no synchronous way to end a session and get its report now.
- **State is a plain dataclass, not a validated model.** Session state is
  internal machinery that mutates every turn and holds live sub-objects, not a
  fixed shape of external data validated at a boundary. A dataclass fits that;
  a validation model would fight it.
- **`SweepClassification` lives in `models/`, not `engine/`.** Session state
  and the engine both speak in classifications, so the type sits in the shared
  leaf package both can depend on without a cycle — the same reason `Message`
  lives there.

## Deferred (planned, not yet built)

- **Stale-state fallback** — the `acquire` timeout currently raises; the
  proceed-on-old-state-and-emit path lands with the proxy.
- **Bounded lock cleanup** — closed sessions leave their lock in the registry;
  a safe, bounded reclaim of long-dead locks is a future refinement.
- **Persistence** — the in-memory store is replaced by a file-backed one.
- **Session identity** — a session id is supplied by the caller today; deriving
  it (by fingerprinting a request's message prefix, or by a time window) needs
  a real request to fingerprint and lands with the proxy.
- **Background scheduling** — the ager is designed to be fire-and-forget, but
  nothing schedules it yet; the caller that does is the proxy.
- **Pre-computation** — the assembler reads what the ager maintained; it does
  not yet re-rank young turns against the incoming query, nor inject relevant
  facts from the permanent generation. Both need the per-query relevance path,
  and a latency budget to justify the cost against.
