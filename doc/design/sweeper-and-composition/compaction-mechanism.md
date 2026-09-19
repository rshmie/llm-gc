# Compactor — Compaction Mechanism

## How it works

The sweeper classifies every message as KEEP, COMPACT or ARCHIVE. A
**compactor** is the component that takes a run of adjacent COMPACT messages
and produces a single shorter message that preserves what mattered.

`BaseCompactor` is the ABC every compactor implements, with one method:

```
compact(messages: list[Message]) -> CompactionResult
```

`NoOpCompactor` is the default. It returns the COMPACT-classified messages
unchanged — the same effect as classifying them KEEP — and stamps the result
`method=noop`. A real `LLMCompactor` implements the same interface once an
async HTTP layer exists to call a provider from.

Two rules bind every compactor, present and future:

- **Compaction runs only from `gc.update`, never from `gc.collect`.**
  `gc.update` compacts COMPACT-classified runs, writes the resulting summaries
  into the old generation, and stamps each with a provenance marker.
  `gc.collect` only *reads* those cached summaries. If a COMPACT-classified
  turn has no cached summary — a true cold start — the composer passes the turn
  through verbatim and emits `COMPACT_CACHE_MISS`. It does not compact inline.
- **Summaries are surfaced to the model with a visible marker**, carrying the
  turn range and the method that produced them
  (`[Compacted summary of turns N-M, method=llm]: ...`). The model is never
  shown a transformed turn pretending to be an original.

## Why it exists

The COMPACT bucket is worthless without something to act on it, but real
semantic compaction needs an LLM call, and an LLM call needs an async HTTP
layer. The compactor therefore has to exist as an *interface* before it can
exist as a real implementation, so the engine has a working end-to-end pipeline
— and so the orchestrator, health monitor, sessions and benchmarks all have
something to call — before the provider layer lands.

`NoOpCompactor` is what makes that possible without lying. It reduces nothing,
and it says so.

## Why this shape

- **A pass-through default, not a heuristic one.** A regex "key sentence"
  extractor or a bracket-summary placeholder would cut token count immediately,
  which is tempting. It would also lose meaning silently: the caller cannot tell
  that something was dropped. That is the worst failure mode this project has,
  because it breaks quality preservation and epistemic transparency at the same
  time, and it breaks them invisibly. A compactor that does nothing and reports
  `method=noop` is honest; one that guesses and reports success is not. Rejected
  alternative: heuristic compaction as the default.
- **An ABC, not a `Protocol`.** Compactors share real behaviour — the provenance
  stamping and result shape are identical across implementations — so a base
  class that supplies them is worth the coupling. (Scorers went the other way,
  and `scoring/overview.md` explains why.)
- **`gc.update`-only execution, not inline.** Compaction inside `gc.collect`
  would put a multi-hundred-millisecond LLM call on the request hot path, which
  is the exact latency problem the hot/cold split exists to avoid. This rule is
  binding on every future compactor: one that cannot finish inside the async
  update window cannot be the default. Rejected alternative: compact on demand
  during collect, with a cache in front — the cold-start case still lands on the
  hot path.
- **Ship the interface before the implementation, async.** Rejected
  alternative: build a synchronous `LLMCompactor` now and rewrite it async
  later. Synchronous provider calls inside the pipeline block the request
  thread, and building it twice is more work than waiting to build it once.
- **`COMPACT_CACHE_MISS` is an event, not a silent fallback.** A turn classified
  COMPACT that had to pass through verbatim is a degradation, so it is visible.

## Consequences

Token savings from COMPACT are zero while the default is in use — KEEP and
ARCHIVE do all the real work. Any benchmark number produced before the
LLM-backed compactor lands is therefore not representative of the full system,
and has to be reported with that caveat attached.

## File map

```
src/llm_gc/
  engine/
    compaction/
      base_compactor.py           # the ABC
      noop_compactor.py           # default concrete compactor
      compaction_result.py        # return type, carries provenance
      compaction_strategy.py      # enum used in the provenance marker
```
