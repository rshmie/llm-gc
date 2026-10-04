# The Cut: How Compaction Reaches the Prompt

## How it works

Generational memory decides *what* happens to each turn: scoring classifies it,
and a cooled turn moves to the next generation — summarised (old generation) or
reduced to facts (permanent generation). **The cut is how those decisions reach
the prompt.** It is the one moment LLM-GC changes what the model is sent.

A cut replaces the oldest part of the conversation with a single message, the
**front**, and keeps every later turn exactly as the client sent it:

```
before any cut:   t0 t1 t2 … t39 t40 … t57                    (the client's history, untouched)

the cut:          [FRONT] t40 … t57
                   └ summaries, quotes and facts for t0–t39, one user-role message

between cuts:     [FRONT] t40 … t57 t58
                  [FRONT] t40 … t58 t59          ← same front bytes; the prompt only grows at the end
                  [FRONT] t40 … t59 t60

the next cut:     [FRONT′] t52 … t60             ← a new front, covering more
```

### What the front contains

Every turn the cut removes is represented in the front according to its
classification — so scoring still decides how much of each turn survives:

| Classification | In the front |
|---|---|
| **KEEP** | Quoted word for word. A turn scored too important to summarise is never lost to a cut. |
| **COMPACT** | Summarised, together with the adjacent COMPACT turns, as one run. |
| **ARCHIVE** | Reduced to extracted facts. |

The front is one user-role message: a header saying it is condensed history, the
facts section, then the summaries and quotes in conversation order. It opens the
conversation as a user turn, which the provider requires.

**Facts are chosen when the front is made, not for each question.** Without a
question to rank against, they are ranked by what they are, then by recency:
decisions first, then facts, then preferences, and raw excerpts — turns extraction
found nothing in — only in the room left over. Ranked by recency alone, a front
kept two late acknowledgements ("Understood, noted") and dropped the turn-2
decision the conversation depended on.

### When a cut happens

Classification decides *what* a cut compacts; **the batch decides *when***. After
each response, in the background, the session's verbatim turns are scored and
swept, and the latest safe cut point is found. A cut is made only when:

1. **The batch is worth a cut.** The turns before the cut point that are classified
   COMPACT or ARCHIVE add up to at least `min_cut_tokens`. A batch of only KEEP
   turns is never cut: quoting compacts nothing.
2. **The cut does not split a tool round.** No tool result kept after the cut may
   answer a tool call removed by it; the provider rejects an orphaned result.
3. **The most recent turns stay verbatim** — at least `last_n_turns_to_keep`.
4. **The cut makes the context smaller.** Measured on the context the cut would
   produce, not summed: the front's headers, quote markers and facts can make it
   larger than the turns it replaces, and a cut that grows the prompt would cost
   a cache reset *and* tokens.

This is how a generational garbage collector works too: a young-generation
collection does not run when one object dies, but when the young generation has
filled enough to be worth a pass.

### When a summary or an extraction fails

The cut **retreats to just before the failure**. Those turns stay verbatim in the
tail, so the next pass meets them in a longer run — which is the retry that can
actually succeed, since repeating the identical call reproduces the identical
refusal. The pass is reported as completed, with the sweeper's intended
compactions in its counts and the failure recorded as a degradation: status says
the pass ran, degradations say what it could not do. If the retreat leaves too
little to be worth a cut, nothing is committed.

### Thinking blocks and the provider's cache

Current models attach their reasoning to each reply as thinking blocks, and each
block is bound to the exact history before it. When a cut replaces that history,
the blocks in the turns the cut keeps no longer match it, so the proxy strips them
from those turns — once, at the cut, which is the provider's prescribed handling
for this shape. Replies produced after the cut keep their thinking, because the
front they were produced against does not change until the next cut.

The same stability is what lets the provider cache the prompt: a cached prefix is
billed at a fraction of the input price and answered faster, but only if it is
byte-identical. Between cuts, every request's conversation is the previous one
plus new turns at the end.

## Why it exists

The first design rebuilt the conversation on every request: summaries slotted
in among the kept turns, archived turns removed from the middle, and a memory
block chosen for each question placed at the top. Each prompt was a new document.
Three things made that untenable:

- **The provider rejects it.** A history rewritten from the middle invalidates
  every later thinking block, and for newer accounts the provider answers with a
  400 before producing any output. The provider's documentation names the three
  shapes that break — keep-tail compaction, background compaction, and removing
  turns from the middle — and the first design used all three.
- **It defeats prompt caching.** A front that changes with every question changes
  the first bytes of every prompt. A conversation could send 70% fewer tokens and
  still cost more than sending all of it cached.
- **It could open with an assistant turn**, which the provider also rejects, once
  the turns before a kept reply had been summarised away.

## Why this shape

- **Change the prompt only at the front, only at a cut.** It is the one rewrite the
  provider's checks accept, and the one that keeps caching effective. *Rejected:
  strip every thinking block on every request and keep rewriting* — accepted by the
  provider, but the model loses its reasoning every turn and nothing ever caches.
  *Rejected: the provider's server-side compaction instead of ours* — sanctioned and
  simple, but it summarises everything alike; what LLM-GC adds is scoring that
  decides how much of each turn survives, extracted facts with supersession, and
  transparency about what the model has lost. It is the baseline the benchmark
  compares against.
- **Batch by compactable tokens, not by window fill.** LLM-GC compacts what has
  cooled, which is its generational premise; batching only stops it cutting for a
  handful of tokens. *Rejected: cut whenever any turn is eligible* — in a growing
  conversation one turn cools on nearly every request, so it would cut nearly every
  request, the most expensive pattern there is. *Rejected: a fill-percentage
  trigger* — a capacity rule, not a generational one.
- **Quote KEEP turns inside the cut.** A cut is a prefix, so it cannot skip an
  important turn; quoting it keeps it word for word without stopping the cut.
- **Retreat on failure rather than quote the failed run.** Quoting loses nothing,
  but the run would never get another chance to be summarised; retreating keeps it
  in the tail, where it grows until the compactor accepts it.
- **Choose facts at the cut.** A front chosen per question would change the prompt
  on every request. *Accepted cost:* when stored facts outnumber the room for them,
  a question about one that was not chosen is answered without it in front of the
  model. Revisit if the benchmark shows decisions being missed.
- **Derive the front from the generations rather than storing it.** Rendering is
  deterministic and the generations change only when a cut commits, so the front is
  frozen between cuts by construction, with no second copy that could drift.

### Accepted costs

- **Less compression on paper.** Only a prefix is compacted, and the newest turns
  always stay verbatim; measured on a 26-turn trace the context fell to 0.65 of the
  raw conversation, where rewriting every request reached 0.45. The comparison that
  matters includes caching, and that is a benchmark, not a trace.
- **A cut resets the cache once.** That is what `min_cut_tokens` amortises.
- **A no-op compactor never cuts.** It wraps text without shortening it, so no cut
  it produced could shrink the context.

## Deferred

- **Defaults are drafts.** `min_cut_tokens`, `last_n_turns_to_keep`, and the
  scoring thresholds are to be tuned by watching real conversations on the
  dashboard.
- **Non-text content inside the front.** A cut turn's images and tool calls survive
  only as the text that described them.
- **The stateless library path** (`GarbageCollector.collect`) still recomposes the
  whole conversation per call; the cut shape applies to the session path the proxy
  uses.
