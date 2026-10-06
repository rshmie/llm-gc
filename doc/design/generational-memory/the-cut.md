# A Collection: How the Prompt Changes

## How it works

A **collection** (a *cut*, in the code) is the one moment LLM-GC changes what the
model is sent. It takes the turns that have aged out of the tail, makes spans of
them, and re-plans the level of every span the session holds. Everything it
decides reaches the model through one message at the start of the prompt, the
**front**; every later turn is sent exactly as the client wrote it.

```
before any collection:   t0 t1 t2 … t39 t40 … t57                (the client's history, untouched)

a collection:            [FRONT] t40 … t57
                          └ spans for t0–t39, each at its level, in one user-role message

between collections:     [FRONT] t40 … t57 t58
                         [FRONT] t40 … t58 t59    ← same front bytes; the prompt only grows at the end
                         [FRONT] t40 … t59 t60

the next collection:     [FRONT′] t52 … t60       ← new spans added, every span's level re-planned
```

### What a collection does, step by step

```
 after a response, in the background, one pass per session at a time

 1  CUT POINT   the latest position that keeps at least `last_n_turns_to_keep` turns
                verbatim, where a span may begin, and splits no tool round;
                everything before it is the batch
                      │
 2  DRAFT       group the batch into spans (one exchange each) and take each span's
                opening line — no model call
                      │
 3  PLAN        score every span, old and new, against what the user is discussing
                now, and plan every level, with each new span's summary estimated
                      │
                      ├── the batch's tokens the plan would not keep word for word
                      │   are under `min_cut_tokens`?  ──► stop: not enough to collect
                      │
                      ├── the cut would not save anything, at the estimated costs?
                      │                                ──► stop: no saving (nothing paid for)
                      ▼
 4  CONDENSE    read the new spans' facts (one model call, given the facts on record),
                then summarise each new span (one model call per span)
                      │
 5  RE-PLAN     plan again with the real facts and summaries
                      │
 6  MEASURE     do the new spans, as rendered, cost less than the turns they replace?
                      │  no ──► stop: no saving, and report the cut it would have made
                      ▼
 7  COMMIT      one synchronous step, onto the session's CURRENT snapshot
```

**Steps 2 and 3 cost nothing**, which is why they come first: a collection decides
whether it is worth making before it pays for any model call. Its estimate leaves out
the new spans' facts, which are not read yet; the measure in step 6 includes them. The same draft and
plan is the dashboard's preview of the next collection, so the preview cannot drift
from what the pass does.

### When a collection happens

**The plan decides how each span is shown; the batch decides when.** A collection
is made only when:

1. **The batch is worth one.** The batch's turns that the plan would not keep word
   for word add up to at least `min_cut_tokens`. A batch the plan would keep
   entirely is never collected: moving kept turns into the front saves nothing.
2. **It would save something, at the estimated costs.** A session's first
   collection also pays for the front's header, so a short batch can be worth
   condensing and still not worth a collection.
3. **No tool round is split.** No tool result kept after the cut may answer a tool
   call before it. The provider's rule is stricter than "no orphans": a tool result
   must immediately follow the message holding its call. The front is always the
   first message, so it can never land between the two.
4. **The most recent turns stay verbatim** — at least `last_n_turns_to_keep`.
5. **The cut pays for itself.** Its new spans, rendered, must cost less than the
   turns they replace. Re-planned older spans are not part of this measure:
   restoring a span the conversation returned to costs tokens on purpose, and is
   bounded by the recap budget instead.

This is how a generational garbage collector works too: a young-generation
collection does not run when one object dies, but when the young generation has
filled enough to be worth a pass.

```
 a pass's outcome            why
 ──────────────────────────  ──────────────────────────────────────────────────────
 completed                   rules 1–5 held; the collection is committed
 not enough to collect       rule 1: the batch is below min_cut_tokens
 no saving                   rule 2 (decided before paying) or rule 5 (after) — the
                             cut it would have made is reported, so a reader can see
                             by how much it fell short
 already running             another pass is in flight for this session
```

The dashboard's **Run GC now** is the same pass with `min_cut_tokens` taken as zero;
every other rule still holds.

### What a pass reports

A completed pass reports its **outcome**, by turn: which of the cut's turns were
kept, compacted, archived or folded out of view, and which turns condensed earlier
it showed with less detail or with more. Its token totals describe the cut's turns
by the level the plan gave them. A pass that stopped for
"no saving" reports the outcome it *would* have had, and the context it would have
produced.

### When a summary or the facts fail

The span is **still made**:

```
 summary refused or failed  ──► the span cannot be summarised: the plan keeps it whole
                                 (or archives it, if its score says so)
 facts could not be read    ──► the collection's spans have no facts: archived,
                                 each shows its opening line alone
```

Both are reported as degradations, and nothing is lost: the span holds its turns.
Keeping a span whole costs tokens, so a collection whose only new span lost its
summary usually no longer pays for itself, and is reported as "no saving" with the
failure beside it. A span made while the provider was down keeps no summary; a
later collection does not retry it.

### Thinking blocks and the provider's cache

Current models attach their reasoning to each reply as thinking blocks, and each
block is bound to the exact prompt before it; a request that replays a block whose
prompt has since changed is rejected. A collection changes the prompt, so the
blocks in the turns it keeps no longer match. The proxy strips them from every turn
sent with **the first request that carries the new front** — the provider's
prescribed handling for keep-tail compaction. They are removed from a leading run of
turns, oldest first, which is the only removal that leaves later blocks valid.
Replies produced from that request on keep their thinking, because the front they
were produced against does not change until the next collection.

The boundary is recorded when the front is first sent, not when the collection
commits:

```
request N    plans the collection in the background   reply N produced against the OLD front
                         └ commit
request N+1  first to carry the new front              sends reply N back  -> its thinking is stale
             boundary = the turns it sends              reply N+1 produced against the NEW front
request N+2  same front                                 reply N+1 keeps its thinking
```

The commit cannot see reply N — it is still being written, or has not reached the
proxy yet — and with a slow summary call the commit can land after request N+1 has
already gone out on the old front. Only the first request that carries the front
knows which turns predate it.

The same stability is what lets the provider cache the prompt: a cached prefix is
billed at a fraction of the input price and answered faster, but only if it is
byte-identical. Between collections, every request's conversation is the previous
one plus new turns at the end.

## Why it exists

A prompt rewritten anywhere but its start breaks two things the user depends on.
The provider rejects a history changed in the middle, because every later thinking
block no longer matches its prompt; and a prompt whose first bytes change on every
request is never cached, so a conversation sending far fewer tokens can still cost
more than sending all of it. A collection that changes only the front, and only
occasionally, is the one rewrite both accept.

## Why this shape

- **Change the prompt only at the front, only at a collection.** The one rewrite the
  provider's checks accept and caching survives.
- **Cut only where a span may begin.** Before a user message that opens an exchange,
  or, inside an exchange the assistant has already answered, between its tool
  rounds — never between a question and its first answer.

  ```
   you  "Which database?"      ┐
   AI   "Postgres, because…"   ┘ one exchange: the cut falls before it or after it
   you  "And the cache?"       ◄─ a cut may fall here
   AI   [tool call]            ┐
        [tool result]          │ one long agentic exchange
   AI   [tool call]            ◄─ or here, between its tool rounds
        [tool result]          │
   AI   "Done."                ┘
  ```

  Measured before it: with `last_n_turns_to_keep` at 5 and replies far larger than
  `min_cut_tokens`, nearly every cut fell between a question and its answer. Each
  question became a span of its own that the summariser could not shorten — one
  refused, paid call each — and each answer a span whose opening line was its first
  heading rather than what was asked. *Accepted cost:* the tail can be one turn
  longer than `last_n_turns_to_keep`. *Rejected: cutting only before a user message.*
  One long agentic exchange has no such point, so it would never be collected and
  its prompt would grow without bound.
- **Batch by the tokens the plan would condense, not by window fill.** LLM-GC
  condenses what has gone quiet, which is its generational premise; batching only
  stops it collecting for a handful of tokens.
- **Plan before paying.** Drafting and planning cost no model call, so a pass that
  will not collect, or whose cut cannot pay for itself, stops before buying
  summaries it would throw away.
- **The guard measures the cut alone.** Re-planned older spans can cost more on
  purpose — a span the conversation returned to is restored — and the recap budget
  bounds that. Measured with them included, a collection that restored a span (+112
  tokens) while archiving a turn (−104) was abandoned whole, restore included.
- **A failed summary keeps the span whole.** Losing detail is never the result of a
  failure.
- **A turn leaves the young generation only into a span.** Committing checks that
  every departing turn lands in exactly one new span and refuses the collection
  otherwise: a plan that would lose a turn fails loudly instead of losing it.
- **Spans hold their turns, as data.** A span's coverage is its turns, so "where is
  turn 17 now?" can never be answered wrongly, and changing a span's level moves
  nothing.
- **Derive the front from the generations rather than storing it.** Rendering is
  deterministic and the generations change only when a collection commits, so the
  front is frozen between collections by construction.

### Alternatives rejected

- **Rebuilding the whole conversation on every request** — summaries slotted among
  kept turns, condensed turns removed from the middle, facts chosen for each
  question at the top. This was the first design. The provider rejected it (a
  history rewritten from the middle invalidates every later thinking block), it
  defeated caching (the first bytes changed every request), and it could open with
  an assistant turn once the turns before a kept reply had been summarised away.
- **Stripping every thinking block on every request and rewriting freely.**
  Accepted by the provider, but the model loses its reasoning every turn and nothing
  ever caches.
- **The provider's own server-side compaction.** Sanctioned and simple, but it
  summarises everything alike; LLM-GC adds detail levels chosen by relevance,
  reversibility, facts with supersession, and transparency about what the model has
  lost. It is the baseline LLM-GC is compared against.
- **Collecting whenever any turn ages out.** In a growing conversation that is
  nearly every request — a cache reset each time.
- **A fill-percentage trigger.** A capacity rule, not a generational one.
- **Retreating the cut to just before a failed summary.** Earlier, a failed summary
  left its turns in the tail for a later, longer run. With fixed span boundaries a
  refused summary would be refused again; keeping the span whole loses nothing and
  stays bounded by the budget.

### Accepted costs

- **A collection resets the cache once.** That is what `min_cut_tokens` amortises.
- **One summary call per new span**, including spans the plan will archive, so every
  level's material exists before it is needed, **and one call to read the facts** of
  all of them.
- **A no-op compactor never summarises.** Its "summary" is the turns themselves, never
  cheaper than keeping them.

## Known gaps

- **A span that lost its summary is never summarised later.** Only a provider
  *failure* is worth retrying; a refusal would be refused again.
- **Defaults are drafts.** `min_cut_tokens`, `last_n_turns_to_keep`, the level lines,
  the margin and the recap budget are to be tuned on the decision-survival check and
  on real conversations.
- **Non-text content inside the front.** A condensed turn's images and tool calls
  survive only as the text that described them.
- **A collection can become visible in the middle of a tool round.** When the first
  request carrying a new front is a tool result, the reply that made the call loses
  its thinking for that step. The fix is to hold a new front back until a request
  that is not a tool result.
