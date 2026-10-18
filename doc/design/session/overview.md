# Session Management

## How it works

A **session** is one continuous conversation's remembered state, keyed by a `session_id`. It is what lets the engine carry state between turns instead of being handed the whole message list every call.

### What a session owns

**Everything a conversation has produced lives on its `SessionState`, and nowhere else.** That is all three generations — verbatim young turns, old-gen spans (each holding its turns, shown at a detail level), and permanent-gen facts — held together as one **snapshot**. The services that do the work (the span scorer, the compactor, the fact reader, and `GenerationalMemory`) hold *how* to do things and no conversation data at all. They are shared across every session, which is safe precisely because they remember no one. The span scorer does remember each span's importance, but by a hash of its text: it holds none of the text, and the answer is the same whichever session asks.

```
             shared, stateless                       one per conversation
  ┌──────────────────────────────────┐     ┌─────────────────────────────────────┐
  │ GcService                        │     │ SessionState  ("alice")             │
  │  ├ SpanScorer                    │     │   snapshot ──▶ GenerationSnapshot    │
  │  ├ GenerationalMemory            │ ──▶ │                 young  (turns)      │
  │  │   ├ FactReader (LLM)          │     │                 old    (spans)      │
  │  │   └ Compactor (LLM)           │     │                 knowledge (facts)   │
  │  │                               │     │                 version             │
  │  └ SessionManager ── store ──────┼──▶  │   timestamps, aging_in_flight       │
  └──────────────────────────────────┘     └─────────────────────────────────────┘
                                           │ SessionState  ("bob")   ...         │
```

Three components manage sessions:

- **`SessionState`** — one conversation: its id, its current snapshot, the timestamps the idle sweeper reads, and an `aging_in_flight` flag. The flag is process-local coordination rather than conversation data, which is why it sits beside the snapshot rather than in it.
- **`SessionStore`** — the registry of sessions by id. In memory for now; a persistent store implementing the same small interface swaps in later.
- **`SessionManager`** — lifecycle: hands out a session's state (creating it on first sight), reports whether a given state is still the live one for its id, closes sessions and announces each close on the event bus, and runs the idle sweeper.

### The snapshot, and the operations on it

A **snapshot** is immutable. Every change produces a new snapshot with the next `version` number, and the session's reference is replaced. A reader that has taken a snapshot therefore holds a consistent view for as long as it likes — across any number of `await`s — without a lock, because nothing can change it.

The engine performs four operations on a session that change or use it, each with its own guarantee, and one that only looks:

| Operation | Who calls it | Guarantee |
|---|---|---|
| **collect** (assemble the context) | the proxy, before forwarding a request | Never waits. Reads the current snapshot once and changes nothing. |
| **record** turns | the proxy, from the client's history on each request; a library caller, one turn at a time | Immediate. The next collect sees them. A library caller's turn must be the next by number. |
| **switch** a staged cut in | the proxy, just before collect | Immediate: no model call, no `await`. Commits the staged cut onto the current snapshot, when the switch policy says so. |
| **age** (prepare a cut: draft, plan, summarise, stage) | after each response, in the background | At most one per session at a time; stages onto whatever the snapshot is *now*. |
| **inspect** | the dashboard | Read-only. Never creates a session, never resets its idle clock, announces nothing. |

Besides the three generations, a snapshot records how many cuts the session has had, the cut a pass has staged and the request path has not switched in yet, and, after a switch, the first turn whose thinking blocks were produced against the current front (see [the cut](../generational-memory/the-cut.md)).

Aging is the slow one: a summary is a model call that can take seconds. It is split so that the slow part holds nothing, and it never changes what the model is sent; only a switch does:

```
time ──────────────────────────────────────────────────────────────────────────▶

 aging:    read v7 ──── draft, plan, ask the LLM to summarise ──── re-plan on now, stage
           young 0–9        (seconds, nothing held)                  │
                                                                     ▼
 record:              turn 10 ─▶ v8                            v9 = v8 + staged cut of 0–5
                                young 0–10                     young 0–10, still sent whole
 collect:                         reads v8 ─▶ turns 0–10 verbatim
                                  (returns at once; complete, only less condensed)

 next request:   switch ─▶ v10 = v9 with the staged cut applied: old spans 0–1, 2–3, 4–5
                                                                 young 6–10
                 collect ─▶ the front, then turns 6–10
```

1. **Plan.** The pass reads the current snapshot (the *base*) and works out what to change: which turns leave the young generation, the spans they become, the facts read from them, and which older spans change level. Nothing reaches session state.
2. **Condense.** It reads the facts and writes the summaries. Nothing is held while it waits.
3. **Re-plan and stage.** Against the snapshot as it is *now*, in one step with no `await` inside it, it plans again with what it condensed and stores the result as a **delta** - things to remove and values to add, never replacement lists - held as the session's staged cut. Spans whose turns are no longer the oldest young turns are left out: switched in while it waited, or rewound. A replacement list built from the base would erase any turn recorded while the pass was waiting.
4. **Switch.** On a later request, the delta is applied to the current snapshot in one step with no `await` inside it. Turns recorded in the meantime survive, because the delta only removes what it names.

Three checks guard it:

- **One aging pass per session at a time.** A second trigger while a pass is in flight is reported (`bypassed_aging_in_flight`) and skipped, not queued. The next response triggers aging again. Without this, two passes reading the same base would summarise the same turns twice — two model calls per span, two spans for one exchange, and two deltas each removing the same turns.
- **The session must still be live.** If the session was closed while the pass was running — including when a new conversation has since reopened the same id — the result is discarded (`discarded_session_closed`) and nothing is announced. A closed session stays closed, and a reopened id never inherits the previous conversation's work.
- **A staged cut follows the history.** A rewind that discards one of its turns drops it with the abandoned branch; a pass whose prepared turns a rewind has changed stages nothing (`discarded_history_changed`).

A switch does not wait for a running pass, and a pass does not hold switches back: in a long burst a pass is nearly always running, and holding switches would stop the stale-weight bound from ever being enforced. The pass re-planning on the current snapshot is what makes the two safe together, and a span condensed before a switch is never paid for twice.

A delta that names a turn or span the snapshot no longer holds is a programmer error and raises: only one pass ages a session, so it means that rule broke, and merging anyway would silently drop or duplicate content. So is a delta in which a departing turn does not land in exactly one new span: a turn would be lost or shown twice.

### Events describe committed changes

Events for spans summarised, spans archived or lowered, added facts, and supersessions are emitted **after a switch commits**, each carrying the `session_id`, so a subscriber that reads state on hearing an event sees the change it was told about, and a cut that is staged and never switched in announces nothing. The terminal GC event is emitted once per pass, whatever its outcome, and once per switch, as the collection. The assembler emits nothing: it changes no state.

Both the context the ager reports and the list the assembler returns come from a single assembly step, so the number a dashboard shows and the payload a caller forwards cannot drift apart. That context is measured whole — the front included — because it is real tokens in the real prompt.

### Aging prepares a cut; the request path switches to it

An aging pass prepares a **cut** (a *collection*, once it takes effect): the oldest turns become spans in one front message, every later turn stays verbatim, and every span's level is planned again. Preparing changes nothing the model is sent; the result is held as the session's staged cut. The prompt changes only when a **switch** on the request path commits it, and only when the switch policy says so: the cache is cold, the saving repays the rewrite, or the stale weight has reached its bound.

The plan decides how each span is shown. A pass prepares a cut only when the new turns the plan would not keep word for word add up to `min_cut_tokens`, because each pass pays for model calls, and only when the cut pays for itself. Otherwise the pass reports itself bypassed and changes nothing. The mechanism, its rules, and the rejected alternatives are in [the-cut.md](../generational-memory/the-cut.md); when a cut is switched in is in [when-to-collect.md](../generational-memory/when-to-collect.md).

### Failures are contained

```
 a summary fails     ──► the span is made without one: kept whole rather than summarised
 reading facts fails ──► the span is made without facts: archived, it shows its opening line
 anything else fails ──► the session is left exactly as it was; the pass is reported bypassed
```

The first two are reported as degradations, and nothing is lost: the span holds its turns. The third is reported rather than raised: the ager is called fire-and-forget, so an exception escaping it would be invisible.

## Why it exists

Without sessions, the engine is stateless: every call re-derives everything from a full message list, and there is nowhere to keep a conversation's summaries and facts between turns. A proxy serving many conversations needs each one's memory to persist across requests, and to stay strictly its own.

**Isolation is the first requirement.** When old and permanent generations were held by one shared `GenerationalMemory`, a proxy serving two users injected one user's archived facts into the other's prompt, labelled as the second user's own memory. That is a privacy failure, a correctness failure, and a false statement by the transparency layer, all at once. It went unnoticed because every engine test drove a single session. Session-owned state makes isolation structural: an operation can only touch the session it was handed.

**The hot path must never wait on the cold path.** Using LLM-GC must not make a user's experience worse than not using it. Collect runs on the request path, before the user's request is even forwarded; aging runs afterwards and can spend seconds on a model call. A design in which the first can wait for the second puts LLM-GC's background work directly into the user's latency.

## Why this shape

- **Session-owned data, stateless services.** The usual backend shape: handlers hold behaviour, the store holds state. It makes deletion a single operation (removing a session removes everything derived from it), makes a session persistable as plain data, and lets a later multi-process deployment move sessions to an external store without touching the services. Rejected alternatives: *one `GenerationalMemory` instance per session* — session data would then carry live services (an LLM client, the event bus) and could not be persisted, and deletion would have to reach inside a service object; *a session-keyed map inside `GenerationalMemory`* — two owners for one session's data, so expiring a session from the store would leave its facts behind, a memory leak and a retention breach.
- **Immutable snapshots, not a lock.** Readers never wait and cannot observe a half-applied change, and that holds by construction rather than by discipline: no future write path can mutate state across an `await` because state cannot be mutated at all. *Rejected: a per-session lock held for the whole update.* It made collect queue behind compaction; measured in an agent-style loop (next request 200ms after the reply, compaction about a second), 18 of 40 requests timed out on the lock and were forwarded without GC. *Rejected: keeping a lock between writers and committing at the end.* Recording a turn then waited behind the model call, so a collect in that window returned a context missing the most recent turn — a plausible-but-wrong prompt with no signal.
- **A delta, not a recomputed result.** Describing *changes* rather than *outcomes* is what makes a slow computation safe to merge onto newer state, in the way a patch applies on top of commits made after branching where an overwrite discards them. Departing turns are named by turn index; a span whose level changes is named by value, so a delta can only change the span it was planned against.
- **Skip a second aging trigger rather than queue it.** Queuing would let aging fall arbitrarily far behind and spend model calls on plans computed against stale state. Skipping costs, at most, a context slightly above target for a turn or two — tokens, never waiting.
- **A version number on every snapshot.** It orders snapshots for debugging and events, and it is what an external store needs to commit with compare-and-set ("write version 9 only if the stored version is still 8"), so moving to one needs no redesign.
- **Session ends on idle timeout, the cap, or explicit close, and reopens on demand.** A session idle for a day is closed, and past 64 held sessions the least recently active is closed; without an explicit close a library caller has no way to end a session now. The limit is long because a person's break is not the end of a conversation: closing it throws away the recap, so the next message is sent whole and a new collection pays again for summaries already made, and it does so after a break, when the provider's cache is cold and switching a recap in would cost nothing. The dashboard can only show a session that is held. The cap, not the timer, bounds the memory: a session holds its conversation's text, about half a megabyte for a 100k-token conversation. *Rejected: a 30-minute timeout* — a lunch break closed a live Claude Code conversation, emptied its dashboard, and cost a resend and a fresh collection. *Rejected: keeping a read-only view after closing* — the dashboard would show it, but every break would still cost the resend. *Revisit* if the proxy serves many people at once, or once sessions are stored outside the process.
- **A close is announced on the event bus, and whatever keeps a copy of a session's data drops it there.** The dashboard keeps its own copy of each conversation as the client sent it, and drops it on `session_closed`, so the copy goes when the session does. The close is announced after the session is gone and only for a session that was held, so a listener never finds the session still there and never hears one close twice. It carries only the session's id: a field such as why the session closed can be added later without breaking a listener, while one that listeners read cannot be taken out. *Rejected: a close callback on `SessionManager`* — a second publish/subscribe mechanism beside the bus, which would need its own guard so a failing listener cannot break the sweep. *Rejected: checking the store when the copy is read* — a closed conversation nobody opens again stays in memory for the life of the process. *Rejected: keeping the copy in the session's state* — the copy is the dashboard's, kept only while the dashboard runs, and the library would hold the application's data.
- **A library caller's turn must be the next one; a missing count is filled in.** Cuts and facts are filed by turn number, so a turn numbered out of order would run two turns together under one number, and a turn counted as nothing keeps every batch below its minimum, so nothing is ever collected. The proxy numbers and counts the turns it records; the library's one-call loop takes both from the caller's code, so it checks them. A count has one right answer, read from the turn's own text, so a missing one is filled in the way the proxy counts and a given one is kept. A position cannot be read from the turn, and when the caller and the session disagree about it neither can be trusted, so the turn is refused and nothing is recorded. A conversation resumed after a restart is replayed from its first turn: a session that began at turn 40 would hold none of turns 0–39 and send the model a conversation without its start. *Rejected: renumbering the turn* — the caller's numbers and the session's would drift apart unseen, so a fact reported at one turn points at another in the caller's records, and a dropped or repeated turn goes unnoticed. *Rejected: numbering a turn only when the caller gave no number* — `Message(...)` and `Message(..., turn_index=0)` would then behave differently. *Revisit* if replaying a long conversation one pass per turn proves too slow: recording many turns at once would be an addition.
- **Looking does not count as activity.** Inspection reads the store without touching the session, so a dashboard left open on a conversation does not keep it alive, and an inspection of a closed session reports it absent rather than creating an empty one.
- **`GenerationSnapshot` and `Span` live in `models/`.** Session state and the engine both depend on them, so they sit in the shared leaf package both can import without a cycle.

### Accepted costs

- **Copying on every change.** A commit copies the young, old, and knowledge tuples: O(turns + spans + facts) references, never the text. Young is bounded by aging; spans and facts grow over a session, because every span keeps its turns. Revisit at around ten thousand facts per session, with a persistent data structure or an index by topic.
- **Every turn is held for the session's life.** The price of being able to show any span in more detail again; dropped with the session.
- **A context can be one pass behind.** While aging is in flight, collect returns the pre-pass context: complete and correct, only less compressed.
- **Single event loop.** "No `await` inside the commit" is atomic only on one event loop. If aging ever moves to threads or processes, the commit must become a real compare-and-set on `version`.

## Deferred (planned, not yet built)

- **Persistence** — the in-memory store is replaced by a persistent one; a commit becomes a conditional write on `version`, and the liveness check becomes what stops a late commit from resurrecting a closed session.
- **Work on the collect path** must stay within a latency budget, and CPU-bound work there must move off the event loop: an asyncio timeout cannot interrupt code that never awaits.
