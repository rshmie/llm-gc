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


- **The run reaches the model as data, not as turns.** The obvious shape is to
  pass the run through as the model's own message list. It is rejected: every
  character of the run is untrusted third-party content, and as real messages a
  turn reading "ignore your instructions" sits in the same structural position as
  a genuine one, with nothing to distinguish them. The run is rendered into a
  single delimited block instead, and the system prompt states that the block is
  material to summarise and never direction. Roles survive as labels, because
  "the user decided" and "the assistant suggested" are different facts. Cost
  accepted: the model reads labelled text rather than native roles, which is
  marginally worse for summary quality, and the delimiting is a mitigation rather
  than a guarantee.


- **The output cap is a fraction of the run, with a floor.** A fixed cap is
  either too tight for a long run — truncation, refusal, a wasted call — or too
  loose for a short one, where the "summary" can legally be as long as its input
  and nothing has been saved. Proportional makes the saving structural instead of
  a hope about the model's brevity. The floor exists because a proportional cap on
  a short run produces a budget too small for one sentence.


- **A summary that is not an improvement is refused, not filed.** Three cases:
  output that stopped at the token cap and is therefore missing its tail, output
  that is empty, and output no shorter than the run it would replace. The caller's
  correct response to all three is to do nothing, so the compactor raises and the
  run stays verbatim for the next pass — one wasted call, nothing lost. Returning
  a degraded result instead would delete the originals and file the bad summary,
  which is the one outcome this project must never produce silently. Rejected
  alternative: fall back to the pass-through marker, which turns a refusal into a
  guaranteed expansion.


- **Adjacent cooled turns are summarised as one run, not one at a time.** This
  began as an efficiency note and turned out to be a precondition. Every summary
  carries a fixed-size provenance marker, so a single short turn can never be
  replaced by something smaller than itself, and a real compactor refuses every
  such promotion — the aging path files nothing at all. Grouping also produces
  better summaries, because a run has a thread to follow where one turn has only
  itself. A run breaks at the first non-COMPACT turn, so a summary never spans
  turns that were separated by a kept one. Promotion is all-or-nothing per run:
  filing the summary while leaving one of its turns young would send both the
  summary and the original.

## Consequences

With the pass-through default in use, COMPACT does not merely save nothing — it
costs. The marker it prepends makes every "summary" longer than the run it
replaced, so KEEP and ARCHIVE do all the real work and compaction works against
them. Any benchmark number produced with that default is not representative of
the system and has to be reported with the caveat attached.

With the LLM-backed compactor, compaction becomes the second real source of
saving alongside ARCHIVE, and the refusal guard makes its failures visible rather
than silent: a refused promotion leaves the turn verbatim, so the context grows
instead of degrading. That is the intended direction of failure — a prompt that
is too large is a cost problem, and a prompt that quietly lost its meaning is not
recoverable.

Two costs come with it. Compaction now makes a network call inside the async
update window, so a slow or unavailable provider stops the aging path (safely:
turns stay young). And it spends money per promotion, which is why aging is
pressure-triggered rather than unconditional.

## File map

```
src/llm_gc/
  engine/
    compaction/
      base_compactor.py           # the ABC
      noop_compactor.py           # default concrete compactor
      compaction_result.py        # return type, carries provenance
      compaction_strategy.py      # enum used in the provenance marker
      llm_compactor.py            # summarises by calling a model
  llm/
    prompts/
      compaction.md               # the summarisation system prompt
```

The prompt lives in a file rather than in the class, because it is the part most
likely to be changed and by someone other than its author, and a diff should show
the wording change on its own line. It is loaded through `importlib.resources` so
it still resolves when the package is not a directory of loose files on disk.
