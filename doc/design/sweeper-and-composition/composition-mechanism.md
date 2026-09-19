# Composer — Composition Mechanism

## Why Does This Exist?

The Sweeper produces a `SweepResult` — every message labelled KEEP, COMPACT,
or ARCHIVE. But a list of *labels* is not yet a context that can be sent to
a model. Something has to walk the labels in order, *act* on each one
(passing KEEP through, calling the compactor for COMPACT runs, depositing
ARCHIVE into memory), and produce the actual list of messages the LLM will
see.

That is the Composer's job — and only the Composer's job. It is the
**single seam** between the live transcript and `GenerationalMemory`.
Nothing else in the engine writes to memory; nothing else assembles the
final message list.

This is the **Compose** phase of the Mark-Sweep-Compact pipeline. In Java
GC terms: marking identifies liveness, sweeping classifies, compacting
rewrites the live set into a contiguous form. Here, the analogue is
literal: the Composer rewrites the conversation as a contiguous, in-order
sequence of KEEP messages and compactor summaries, with ARCHIVE removed
and remembered elsewhere.

## What the Composer Does — and Does Not — Do

The Composer's contract is narrow on purpose:

| Does                                                              | Does not                                                                |
|-------------------------------------------------------------------|-------------------------------------------------------------------------|
| Iterate `SweepResult.sweep_entries` in turn order                 | Score messages (the Scorer did that)                                    |
| Pass KEEP entries through verbatim                                | Decide which messages get compacted (the Sweeper did that)              |
| Group **adjacent COMPACT** entries into a *compact run*           | Implement compaction itself (the injected `BaseCompactor` does that)    |
| Delegate each compact run to the injected `BaseCompactor`         | Decide *how* compaction works (NoOp / LLM / learned — Compactor's job)  |
| Route ARCHIVE entries to `GenerationalMemory.archive_message(...)`| Extract knowledge from archived messages (Generational memory does that)|
| Emit `CONTEXT_COMPOSED` with metrics                              | Decide *which* events flow downstream (consumers subscribe themselves)  |

The Composer **never transforms message content**. The phrase appears
verbatim in `ContextComposer`'s class docstring and is the line that
prevents the anti-pattern the compaction design explicitly forbids — inline summary
formatting drifting back into the composer over time.

## What "Adjacent" Means

This is the spec point that is most easily misread, so it is pinned here.

**"Adjacent"** in the compactor's contract means **consecutive turn indices
where every entry in the run is classified COMPACT and there is no
non-COMPACT entry between them**. The moment the Composer sees a KEEP or
ARCHIVE entry, the current compact run is closed off and handed to the
compactor.

Concretely: with a sweep result like

```
turn  5: KEEP
turn  6: COMPACT
turn  7: COMPACT
turn  8: KEEP
turn  9: ARCHIVE
turn 10: COMPACT
turn 11: KEEP
turn 12: COMPACT
turn 13: COMPACT
turn 14: COMPACT
turn 15: KEEP
```

three separate compact runs are formed, each handed to the compactor as
its own `compact()` call:

- Run A: turns 6–7
- Run B: turn 10 (a singleton run — still a valid run)
- Run C: turns 12–14

A "merge everything COMPACT into one summary" interpretation is rejected.
That would lose temporal grounding (the model could no longer tell that
KEEP turn 11 happened *between* turn 10 and turns 12–14), and the
provenance marker `[Compacted (compaction_strategy=<name>) turns N-M]`
would no longer describe a contiguous span.

## Worked Example — Final Message List

Using the classification above, the Composer's output (with the default
`NoOpCompactor`) is:

```text
final_messages = [
  Message(turn=5,  role=...,         content=<verbatim turn 5>),
  Message(turn=6,  role="assistant", content="[Compacted (compaction_strategy=noop) turns 6-7]\n<turn 6 content>\n<turn 7 content>"),
  Message(turn=8,  role=...,         content=<verbatim turn 8>),
  # turn 9 ARCHIVE → not present in final_messages; deposited in PermanentGeneration via GenerationalMemory.archive_message(...)
  Message(turn=10, role="assistant", content="[Compacted (compaction_strategy=noop) turns 10-10]\n<turn 10 content>"),
  Message(turn=11, role=...,         content=<verbatim turn 11>),
  Message(turn=12, role="assistant", content="[Compacted (compaction_strategy=noop) turns 12-14]\n<turn 12>\n<turn 13>\n<turn 14>"),
  Message(turn=15, role=...,         content=<verbatim turn 15>),
]
```

A few invariants this example demonstrates:

1. **Order is preserved.** Every emitted message carries a `turn_index`
   and the list is in increasing turn-index order. The compactor's
   summary takes the *first* turn in its run as its own `turn_index`
   (per `BaseCompactor.compact()`'s post-condition), which is what makes
   chronological interleaving with KEEP work.
2. **ARCHIVE leaves a gap in turn numbering.** Turn 9 is missing from
   `final_messages`. That is intentional — its content is no longer in
   the live context; what mattered about it lives in `PermanentGeneration`
   as `KnowledgeEntry` objects.
3. **Singleton compact runs are still compacted.** Turn 10 is alone but
   still goes through `BaseCompactor.compact([turn 10])`. The Composer
   does not "skip compaction if the run is short" — that is the
   Sweeper's job (it has `min_compactable_tokens` for exactly this
   reason). Once the Sweeper has decided COMPACT, the Composer respects
   the decision.
4. **The marker is provenance, not summary text.** The `compaction_strategy`
   in the marker tells a downstream reader (or a dashboard) *which*
   compactor produced this summary — `noop`, `llm` or other compactors we may add in the future.
   Without the marker, there would be no  way to look at a final message list and tell which lines are
   verbatim and which were transformed by which compactor.

## Where Each Message Ends Up

After `compose()` returns, every message originally in the conversation
is in **exactly one** of three places:

| Location                    | Which messages                                              | Visible to the LLM? |
|-----------------------------|-------------------------------------------------------------|---------------------|
| `final_messages` (returned) | KEEP entries verbatim, plus one summary message per compact run | Yes              |
| `PermanentGeneration`       | `KnowledgeEntry` records from ARCHIVE messages — extracted, or a verbatim `RAW` entry when extraction finds nothing | No (queried separately later phases) |
| Nowhere                     | (intentionally empty — no message is silently dropped)      | —                   |

The "nowhere" row is the invariant. **Every classified message is
accounted for.** If a future change ever produces a code path where a
message could be silently lost, that is a regression — the composer's
test suite is built around this invariant.

## Event Emission

The Composer emits one event per `compose()` call:

```python
EventType.CONTEXT_COMPOSED
data = {
    "messages_kept": int,
    "messages_compacted": int,            # input messages that went into compact runs
    "messages_archived": int,
    "compact_runs_created": int,          # number of summaries produced
    "tokens_before": int,                 # total input tokens (KEEP + COMPACT + ARCHIVE)
    "tokens_after": int,                  # total tokens in final_messages
    "final_message_count": int,
}
```

The injected `BaseCompactor` separately emits one `MESSAGE_COMPACTED`
event per run it processes, with the `compaction_strategy`, the turn
range, and the pre/post token counts. A subscriber that wants per-run
detail listens to `MESSAGE_COMPACTED`; a subscriber that wants the
overall outcome listens to `CONTEXT_COMPOSED`. Neither event subsumes
the other.

> **Known semantic asymmetry:** `tokens_before` includes ARCHIVE tokens,
> but ARCHIVE messages are not in `final_messages`, so
> `tokens_before - tokens_after` overstates raw savings. The field is
> expected to be renamed `tokens_in_final`, with a separate field for the
> archived bytes, once the dashboard consumes it.

## Composer's Position in the Pipeline

```
                    Scorer ─► SweepResult ─► COMPOSER ─► final_messages ─► LLM
                                                │
                                                └─► GenerationalMemory.archive_message(...)
                                                          │
                                                          ▼
                                                    PermanentGeneration
                                                  (KnowledgeEntry records)
```

The Composer is the only arrow with a fork — every other component in
the pipeline produces a single output. That fork is the reason it earns
its own ADR: it is the place where the engine's data-flow stops being
linear, and the rule "*only* the composer writes to memory" is what keeps
that fork from spreading. That is also why it is documented apart from
[compaction-mechanism.md](compaction-mechanism.md), which covers the
component it delegates to.

## File Map

```
src/llm_gc/
  engine/
    context_composer.py           # The Composer itself
    compaction/
      base_compactor.py           # ABC the Composer delegates to
      noop_compactor.py           # Default concrete compactor
      compaction_result.py        # Return type from BaseCompactor.compact()
      compaction_strategy.py      # Enum used in the provenance marker
    generations/
      generational_memory.py      # The only writer the Composer calls
```
