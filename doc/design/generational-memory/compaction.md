# Compaction: How a Span Gets Its Summary

## How it works

When a collection makes a span, it asks a **compactor** for the span's summary, once. The summary is what the span's summarised level shows beneath its opening line and facts.

```
 a span's turns ──► worth_compacting(turns)? ── no ──► no summary: the span can be
                         │                             kept or archived, never summarised
                         │ yes
                         ▼
                     compact(turns) ──► CompactionResult
                         │                 summary        the compactor's message, with its marker
                         │                 summary_text   what it says, without the marker
                         │                 original_token_count, compaction_strategy
                         │
                         └── refused or failed ──► the span is made without a summary,
                                                   and the failure is reported
```

`BaseCompactor` is the abstract base class every compactor implements:

```
compact(messages: list[Message]) -> CompactionResult
worth_compacting(messages) -> bool       # could a summary of these turns be shorter?
```

- **`LLMCompactor`** summarises by calling a model through any `LLMProvider`.
- **`NoOpCompactor`** is what a deployment without a summary provider gets. Its "summary" is the turns themselves, never shorter, so the planner never chooses the summarised level: spans are kept or archived. It reduces nothing, and says so.

**The front owns the marker the model sees.** A compactor returns its summary wrapped in its own provenance marker (`[Compacted (compaction_strategy=llm) turns 6-7] …`); the front renders `summary_text` under its own `[Summarised] (turns 6–7)` heading instead, so every level's marker is written in one place. Which compactor made a summary travels in the result and in events, for the dashboard.

## Why it exists

The summarised level is the middle of the ladder: far smaller than a span's words, far richer than its opening line. Making a summary needs a model, so it is the one part of condensing that costs money and time, and the one that can fail.

## Why this shape

- **Summarise when the span is made, once.** Choosing a level later must never wait on a model, and a span's boundaries never change, so its summary never needs remaking. *Accepted cost:* a span is summarised even if the plan archives it straight away, because the material for every level must exist before it is needed. Facts are read in a call of their own, once per collection: an exchange too short to summarise can still change a fact (see `overview.md`).
- **Ask whether a summary could be shorter before calling.** A run too small to beat the compactor's marker is never sent: refused before the call, it would be a paid answer thrown away. `worth_compacting` lets a collection skip it without a refusal to report.
- **An abstract base class, not a `Protocol`.** Compactors share real behaviour — the marker and the result's shape — so a base class that supplies them is worth the coupling. (Scorers go the other way; see `scoring/overview.md`.)
- **Compaction runs only on the cold path.** A model call on the request path would put hundreds of milliseconds in front of every forwarded request.
- **The turns reach the model as data, not as turns.** Every character of a span is untrusted third-party content; as real messages, a turn reading "ignore your instructions" would sit where a genuine instruction sits. The turns are rendered into one delimited block, and the system prompt says the block is material to summarise and never direction. Roles survive as labels, because "the user decided" and "the assistant suggested" are different facts. *Accepted cost:* labelled text summarises marginally worse than native roles, and delimiting is a mitigation, not a guarantee.
- **The summary's length is a share of the span, with a floor, stated in words.** A fixed length is too tight for a long span and too loose for a short one. The floor exists because a proportional target on a short span is less than a sentence. Words, because models follow a word count far better than a token count. The target never exceeds what fits inside the span once the marker is added.
- **The output budget is that target plus room to think.** Current models reason before answering and reasoning counts against the output cap; a cap sized for the summary alone is spent before the summary starts, and every summary comes back truncated. *Rejected:* switching thinking off in the request — accepted by some models and a 400 on others.
- **A summary that is not an improvement is refused, not used.** Output that stopped at the cap, empty output, and output no shorter than the turns it would stand for are all refused. The span is then made without a summary and kept whole if its score asks for detail: one wasted call, nothing lost. Returning a degraded summary instead would put something the model cannot trust into the front, silently.

## Consequences

A provider that is slow or down stops spans from being summarised, safely: the spans are made without summaries, kept or archived by their score, and each failure is reported. A span made then is not summarised later.

Every summary costs money, which is one reason a pass waits for a batch worth making, and why it checks that its cut would pay for itself before buying any summary (see [the-cut.md](the-cut.md) and [when-to-collect.md](when-to-collect.md)).

## Alternatives rejected

- **A heuristic "key sentence" compactor as the default.** It would cut tokens immediately and lose meaning silently — the caller could not tell what was dropped. A compactor that does nothing and says so is honest; one that guesses and reports success is not.
- **Summarising runs of adjacent turns that a sweeper had classified "compact".** This was the earlier unit. A run's boundaries depended on how neighbouring turns happened to score, so a summary could start mid-exchange; spans follow exchanges instead, and every span gets a summary, whatever its level.
- **Leaving a refused run in the tail for a longer run later.** With runs gone, spans have fixed boundaries; a refused summary would be refused again.

## File map

```
src/llm_gc/
  engine/
    compaction/
      base_compactor.py           # the abstract base: compact, worth_compacting, the marker
      noop_compactor.py           # turns kept as they are
      llm_compactor.py            # summarises by calling a model
      compaction_result.py        # the summary, its text, and provenance
      compaction_strategy.py      # which compactor made it
  llm/
    prompts/
      compaction.md               # the summarisation system prompt
```

The prompt lives in a file rather than in the class, because it is the part most likely to be changed, and a diff should show a wording change on its own line. It is loaded through `importlib.resources`, so it resolves when the package is installed as an archive rather than as loose files.
