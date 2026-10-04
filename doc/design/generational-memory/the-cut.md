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

Each summary or quote is stored as a **front entry** that records two things
beyond the message the model sees: the turns it stands in for, and what it says
without its marker. The first lets anyone ask where turn 17 is now — a summary's
message names only its first turn. The second is what facts are extracted from
when the entry is later archived.

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
   answer a tool call removed by it. The provider's rule is stricter than "no
   orphans": a tool result must immediately follow the message holding its call,
   with nothing between them. The front is always the first message, so it can
   never land between the two.
3. **The most recent turns stay verbatim** — at least `last_n_turns_to_keep`.
4. **The cut makes the context smaller.** Measured on the context the cut would
   produce, not summed: the front's headers, quote markers and facts can make it
   larger than the turns it replaces, and a cut that grows the prompt would cost
   a cache reset *and* tokens.

This is how a generational garbage collector works too: a young-generation
collection does not run when one object dies, but when the young generation has
filled enough to be worth a pass.

A pass that finds no batch reports *not enough to compact*; one that compacted a
batch and found the result would not shrink the context reports *no saving* —
two different reasons, so they are two different outcomes. A no-saving pass also
records the cut it abandoned: the turns, what each would have become, and the
context it would have produced, so the shortfall can be explained rather than
only stated. The dashboard's
**Run GC now** is the same pass with rule 1's minimum taken as zero: rules 2–4
still hold.

### The same pass sweeps the front, and can overrule the sweeper

Without a sweep of its own, the front would grow by one summary per cut forever.
So after the cut is planned, every front entry — the ones earlier cuts made *and*
the ones this cut has just made — is scored against the context the cut will
leave, and a cold one is archived into facts. A turn the sweeper quoted can
therefore end the pass as facts:

```
sweeper's decision        t0–t11 quoted or archived, t12 quoted, t13 summarised, t14 quoted
front sweep (same pass)   the quotes of t0–t11 score cold in the context the cut leaves → facts
what the pass reports     outcome: 2 quoted · 1 summarised · 12 reduced to facts
```

A pass therefore reports two different things. Its classification counts are the
**sweeper's decision**; its **outcome** is where the cut's turns actually ended up,
plus how many turns earlier cuts had condensed went to facts in the same pass.
Anything showing what a cut *did* reads the outcome; a decision count shown as
"what happened" once reported "6 reduced to facts" for a clean-up that had reduced
12. *Rejected: changing the counts to mean the outcome* — they sit beside the
sweeper's token totals, which tuning the thresholds reads as the sweeper's profile,
and the two would then describe different turns. An abandoned cut reports its
outcome the same way: what each turn would have become after the front sweep.

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
block is bound to the exact prompt before it; a request that replays a block
whose prompt has since changed is rejected. A cut changes the prompt, so the
blocks in the turns it keeps no longer match. The proxy strips them from every
turn sent with **the first request that carries the new front** — the provider's
prescribed handling for keep-tail compaction. They are removed from a leading
run of turns, oldest first, which is the only removal that leaves later blocks
valid. Replies produced from that request on keep their thinking, because the
front they were produced against does not change until the next cut.

The boundary is recorded when the front is first sent, not when the cut commits:

```
request N    plans the cut in the background    reply N produced against the OLD front
                         └ commit
request N+1  first to carry the new front       sends reply N back  -> its thinking is stale
             boundary = the turns it sends       reply N+1 produced against the NEW front
request N+2  same front                          reply N+1 keeps its thinking
```

The commit cannot see reply N — it is still being written, or has not reached
the proxy yet — and with a slow compaction call the commit can land after
request N+1 has already gone out on the old front. Only the first request that
carries the front knows which turns predate it.

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
- **Front entries carry their coverage, as data.** *Rejected: a field on every
  message* — the core vocabulary type would carry a summary-only concept on every
  turn; *a coverage map beside the old generation* — two collections that every
  eviction must keep aligned, and misattribute silently when one is missed; *a
  per-turn ledger written at each cut* — a second record of state the generations
  already imply, free to drift from them. On the entry, coverage leaves with the
  entry when it is archived and is passed on to the facts it yields.
- **Facts are extracted from what an entry says, not from how it is shown.** A
  short turn stating a decision is too small to summarise, so a cut quotes it;
  when the quote went cold it was archived by parsing the quote as the model
  sees it, and "[Quoted exactly] (turn 2, user) The database is postgres" yields
  no fact. It was filed as a raw excerpt, ranked last for the front, and dropped
  from the prompt — the early decision LLM-GC exists to keep, lost to its own
  marker. *Rejected: strip markers with a pattern in
  the extractor* — it couples extraction to every marker format, present and
  future.
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
- **A cut can become visible in the middle of a tool round.** When the first
  request carrying a new front is a tool result, the reply that made the call
  loses its thinking for that step; the provider advises against compacting
  mid-round. Correct, and costs reasoning once per cut. The fix is to hold a new
  front back until a request that is not a tool result.
- **The stateless library path** (`GarbageCollector.collect`) still recomposes the
  whole conversation per call; the cut shape applies to the session path the proxy
  uses.
