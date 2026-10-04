# Session Management

## How it works

A **session** is one continuous conversation's remembered state, keyed by a
`session_id`. It is what lets the engine carry state between turns instead of
being handed the whole message list every call.

### What a session owns

**Everything a conversation has produced lives on its `SessionState`, and
nowhere else.** That is all three generations — verbatim young turns, old-gen
summaries, and permanent-gen facts — held together as one **snapshot**. The
services that do the work (the scorer, sweeper, compactor, knowledge extractor,
and `GenerationalMemory`) hold *how* to do things and no conversation data at
all. They are shared across every session, which is safe precisely because they
remember no one.

```
             shared, stateless                       one per conversation
  ┌──────────────────────────────────┐     ┌─────────────────────────────────────┐
  │ GcService                        │     │ SessionState  ("alice")             │
  │  ├ RelevanceScorer               │     │   snapshot ──▶ GenerationSnapshot    │
  │  ├ Sweeper                       │ ──▶ │                 young  (turns)      │
  │  ├ GenerationalMemory            │     │                 old    (summaries)  │
  │  │   ├ KnowledgeExtractor        │     │                 knowledge (facts)   │
  │  │   └ Compactor (LLM)           │     │                 version             │
  │  └ SessionManager ── store ──────┼──▶  │   timestamps, aging_in_flight       │
  └──────────────────────────────────┘     └─────────────────────────────────────┘
                                           │ SessionState  ("bob")   ...         │
```

Three components manage sessions:

- **`SessionState`** — one conversation: its id, its current snapshot, the
  timestamps the idle sweeper reads, and an `aging_in_flight` flag. The flag is
  process-local coordination rather than conversation data, which is why it
  sits beside the snapshot rather than in it.
- **`SessionStore`** — the registry of sessions by id. In memory for now; a
  persistent store implementing the same small interface swaps in later.
- **`SessionManager`** — lifecycle: hands out a session's state (creating it on
  first sight), reports whether a given state is still the live one for its id,
  closes sessions, and runs the idle sweeper.

### The snapshot, and the operations on it

A **snapshot** is immutable. Every change produces a new snapshot with the next
`version` number, and the session's reference is replaced. A reader that has
taken a snapshot therefore holds a consistent view for as long as it likes —
across any number of `await`s — without a lock, because nothing can change it.

The engine performs three operations on a session that change or use it, each
with its own guarantee, and one that only looks:

| Operation | Who calls it | Guarantee |
|---|---|---|
| **collect** (assemble the context) | the proxy, before forwarding a request | Never waits. Reads the current snapshot once. |
| **record** turns | the proxy, from the client's history on each request | Immediate. The next collect sees them. |
| **age** (score, sweep, cut) | after each response, in the background | At most one per session at a time; commits onto whatever the snapshot is *now*. |
| **inspect** | the dashboard | Read-only. Never creates a session, never resets its idle clock, announces nothing. |

Besides the three generations, a snapshot records how many cuts the session has
had and, after a cut, the first turn whose thinking blocks were produced against
the current front (see [the cut](../generational-memory/the-cut.md)).

Aging is the slow one: compaction is a model call that can take seconds. It is
split so that the slow part holds nothing:

```
time ─────────────────────────────────────────────────────────────────────▶

 aging:    read v7 ──── plan: score, sweep, ask the LLM to summarise ──── commit
           young 3,4,5,10   (seconds, nothing held)                         │
                                                                            ▼
 record:              turn 11 ─▶ v8                                   v9 = v8 + delta
                                young 3,4,5,10,11                     young 10,11
                                                                      old   summary(3)
 collect:                         reads v8 ─▶ turns 3,4,5 verbatim, plus 11
                                  (returns at once; complete, only less compressed)
```

1. **Plan.** The pass reads the current snapshot (the *base*) and works out what
   to change: which turns leave the young generation, which summaries and facts
   are added, which cold summaries are evicted. Nothing reaches session state.
2. **Delta.** The plan is expressed as a **delta** — things to remove and values
   to add — never as replacement lists. A replacement list built from the base
   would erase any turn recorded while the pass was waiting.
3. **Commit.** The delta is applied to the *current* snapshot, which may have
   gained turns since the base was read, in one step with no `await` inside it.
   Turns recorded in the meantime survive, because the delta only removes what it
   names.

Two checks guard the commit:

- **One aging pass per session at a time.** A second trigger while a pass is in
  flight is reported (`bypassed_aging_in_flight`) and skipped, not queued.
  Pressure that persists triggers aging again on the next turn. Without this, two
  passes reading the same base would compact the same turns twice — two model
  calls, two summaries of one run, and two deltas each removing the same turns.
- **The session must still be live.** If the session was closed while the pass
  was running — including when a new conversation has since reopened the same
  id — the result is discarded (`discarded_session_closed`) and nothing is
  announced. A closed session stays closed, and a reopened id never inherits the
  previous conversation's work.

A delta that names a turn or summary the snapshot no longer holds is a
programmer error and raises: only one pass ages a session, so it means that rule
broke, and merging anyway would silently drop or duplicate content.

### Events describe committed changes

The ager is the pass that reports. Events for promotions, archives, added facts,
and supersessions are emitted **after the commit**, each carrying the
`session_id`, so a subscriber that reads state on hearing an event sees the
change it was told about, and a discarded pass announces nothing. The terminal
GC event is emitted exactly once per update, whatever the outcome. The assembler
emits nothing: it changes no state.

Both the context the ager reports and the list the assembler returns come from a
single assembly step, so the number a dashboard shows and the payload a caller
forwards cannot drift apart. That context is measured whole — the front
included — because it is real tokens in the real prompt.

### Aging cuts the conversation in batches

An aging pass changes the prompt only by a **cut**: the oldest turns are replaced
with one front message holding their summaries, quotes and facts, and every later
turn stays verbatim. Classification decides what a cut compacts; a cut is made only
when the turns classified for compaction add up to a batch worth it, because each
cut resets the provider's prompt cache once. Below that, the pass reports itself
bypassed and changes nothing. The mechanism, its rules, and the rejected
alternatives are in [the-cut.md](../generational-memory/the-cut.md).

### Failures are contained

A run that fails to compact, or a turn that fails to archive, stops the cut just
before it: those turns stay verbatim for a later pass, and the failure is reported
as a degradation. A failure anywhere else in planning leaves the session exactly as
it was, and is reported as a bypassed run rather than raised: the ager is called
fire-and-forget, so an exception escaping it would be invisible.

## Why it exists

Without sessions, the engine is stateless: every call re-derives everything from
a full message list, and there is nowhere to keep a conversation's summaries and
facts between turns. A proxy serving many conversations needs each one's memory
to persist across requests, and to stay strictly its own.

**Isolation is the first requirement.** When old and permanent generations were
held by one shared `GenerationalMemory`, a proxy serving two users injected one
user's archived facts into the other's prompt, labelled as the second user's own
memory. That is a privacy failure, a correctness failure, and a false statement
by the transparency layer, all at once. It went unnoticed because every engine
test drove a single session. Session-owned state makes isolation structural: an
operation can only touch the session it was handed.

**The hot path must never wait on the cold path.** Using LLM-GC must not make
a user's experience worse than not using it. Collect runs on the request path,
before the user's request is even forwarded; aging runs afterwards and can spend
seconds on a model call. A design in which the first can wait for the second puts
LLM-GC's background work directly into the user's latency.

## Why this shape

- **Session-owned data, stateless services.** The usual backend shape: handlers
  hold behaviour, the store holds state. It makes deletion a single operation
  (removing a session removes everything derived from it), makes a session
  persistable as plain data, and lets a later multi-process deployment move
  sessions to an external store without touching the services. Rejected
  alternatives: *one `GenerationalMemory` instance per session* — session data
  would then carry live services (an LLM client, the event bus) and could not be
  persisted, and deletion would have to reach inside a service object; *a
  session-keyed map inside `GenerationalMemory`* — two owners for one session's
  data, so expiring a session from the store would leave its facts behind, a
  memory leak and a retention breach.
- **Immutable snapshots, not a lock.** Readers never wait and cannot observe a
  half-applied change, and that holds by construction rather than by discipline:
  no future write path can mutate state across an `await` because state cannot be
  mutated at all. *Rejected: a per-session lock held for the whole update.* It
  made collect queue behind compaction; measured in an agent-style loop (next
  request 200ms after the reply, compaction about a second), 18 of 40 requests
  timed out on the lock and were forwarded without GC. *Rejected: keeping a lock
  between writers and committing at the end.* Recording a turn then waited behind
  the model call, so a collect in that window returned a context missing the most
  recent turn — a plausible-but-wrong prompt with no signal.
- **A delta, not a recomputed result.** Describing *changes* rather than
  *outcomes* is what makes a slow computation safe to merge onto newer state, in
  the way a patch applies on top of commits made after branching where an
  overwrite discards them. Young turns are named by turn index; summaries by
  value, because a regenerated turn re-sends its index and two summaries can share
  one.
- **Skip a second aging trigger rather than queue it.** Queuing would let aging
  fall arbitrarily far behind and spend model calls on plans computed against
  stale state. Skipping costs, at most, a context slightly above target for a
  turn or two — tokens, never waiting.
- **A version number on every snapshot.** It orders snapshots for debugging and
  events, and it is what an external store needs to commit with compare-and-set
  ("write version 9 only if the stored version is still 8"), so moving to one
  needs no redesign.
- **Session ends on idle timeout or explicit close, and reopens on demand.**
  Without an idle timeout the active set grows without bound when users walk
  away; without explicit close a library caller has no way to end a session now.
- **Looking does not count as activity.** Inspection reads the store without
  touching the session, so a dashboard left open on a conversation does not keep
  it alive, and an inspection of a closed session reports it absent rather than
  creating an empty one.
- **`GenerationSnapshot` and `SweepClassification` live in `models/`.** Session
  state and the engine both depend on them, so they sit in the shared leaf
  package both can import without a cycle.

### Accepted costs

- **Copying on every change.** A commit copies the young, old, and knowledge
  tuples: O(turns + summaries + facts). Young and old are bounded by aging; facts
  grow over a session but are small. Revisit at around ten thousand facts per
  session, with a persistent data structure or an index by topic.
- **A context can be one pass behind.** While aging is in flight, collect returns
  the pre-pass context: complete and correct, only less compressed.
- **Single event loop.** "No `await` inside the commit" is atomic only on one
  event loop. If aging ever moves to threads or processes, the commit must become
  a real compare-and-set on `version`.

## Deferred (planned, not yet built)

- **Persistence** — the in-memory store is replaced by a persistent one; a commit
  becomes a conditional write on `version`, and the liveness check becomes what
  stops a late commit from resurrecting a closed session.
- **Work on the collect path** must stay within a latency budget, and CPU-bound
  work there must move off the event loop: an asyncio timeout cannot interrupt
  code that never awaits.
