# A Collection: How the Prompt Changes

## How it works

A **collection** (a *cut*, in the code) is the one moment LLM-GC changes what the
model is sent. It takes the turns that have aged out of the tail, makes spans of
them, and re-plans the level of every span the session holds. Everything it
decides reaches the model through one message at the start of the prompt, the
**front**; every later turn is sent exactly as the client wrote it.

It happens in two halves. A background pass **prepares** a cut and stages it; the request
path **switches** to it when the cache and the model's health call for it. This document
is about what each half does. [when-to-collect.md](when-to-collect.md) is about when.

```
before any collection:   t0 t1 t2 … t39 t40 … t57                (the client's history, untouched)

a collection:            [FRONT] t40 … t57
                          └ spans for t0–t39, each at its level, in one user-role message

between collections:     [FRONT] t40 … t57 t58
                         [FRONT] t40 … t58 t59    ← same front bytes; the prompt only grows at the end
                         [FRONT] t40 … t59 t60

the next collection:     [FRONT′] t52 … t60       ← new spans added, every span's level re-planned
```

### Preparing a cut, step by step

```
 after a response, in the background, one pass per session at a time

 1  CUT POINT   the latest position that keeps at least `last_n_turns_to_keep` turns
                verbatim, where a span may begin, and splits no tool round;
                everything before it is the batch
                      │
 2  DRAFT       group the batch's turns that no staged cut holds into spans (one exchange
                each) and take each span's opening line — no model call
                      │
 3  PLAN        score every span, old, staged and new, against what the user is discussing
                now, and plan every level, with each new span's summary estimated
                      │
                      ├── the new turns the plan would not keep word for word
                      │   are under `min_cut_tokens`?  ──► stop: not enough to prepare
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
 7  STAGE       keep the planned result as the session's staged cut, in place of any
                earlier one. The prompt has not changed.
```

**Steps 2 and 3 cost nothing**, which is why they come first: a pass decides
whether it is worth making before it pays for any model call. Its estimate leaves out
the new spans' facts, which are not read yet; the measure in step 6 includes them. The same draft and
plan is the dashboard's preview of the next collection, so the preview cannot drift
from what the pass does.

A later pass **extends** a staged cut rather than starting again: the spans already
condensed are kept, and only the turns that have gone quiet since are drafted and
paid for. A summary is bought once per span, however many passes there are.

### Switching to it

```
 on the request path, as a request is built

 the switch policy says switch ──► COMMIT   one synchronous step, onto the session's
                                            CURRENT snapshot; this request is the first to
                                            carry the new front
```

The commit applies the staged cut to whatever the snapshot is now, in one step with no
`await` inside it. Turns recorded since the pass survive, because the cut only removes
what it names. A staged cut belongs to the turns it was made from: if the client's history
stops agreeing with them, a rewind or an edit, it is discarded.

The assembler that renders the prompt still changes nothing; the switch is a step of its
own before it.

### What a cut must satisfy

1. **Enough has gone quiet to prepare one.** The new turns the plan would not keep word
   for word add up to at least `min_cut_tokens`. A batch the plan would keep entirely is
   never prepared: moving kept turns into the front saves nothing.
2. **It would save something, at the estimated costs.** A session's first collection also
   pays for the front's header, so a short batch can be worth condensing and still not
   worth a collection.
3. **No tool round is split.** No tool result kept after the cut may answer a tool
   call before it. The provider's rule is stricter than "no orphans": a tool result
   must immediately follow the message holding its call. The front is always the
   first message, so it can never land between the two.
4. **The most recent turns stay verbatim** — at least `last_n_turns_to_keep`.
5. **The cut pays for itself.** Its new spans, rendered, must cost less than the
   turns they replace. Re-planned older spans are not part of this measure:
   restoring a span the conversation returned to costs tokens on purpose, and is
   bounded by the recap budget instead.

Whether a prepared cut becomes the prompt now is the switch policy's question, and not one
of these.

```
 a pass's outcome            why
 ──────────────────────────  ──────────────────────────────────────────────────────
 staged                      rules 1–5 held; the cut is ready to switch to
 not enough to prepare       rule 1: the new turns are below min_cut_tokens
 no saving                   rule 2 (decided before paying) or rule 5 (after) — the
                             cut it would have made is reported, so a reader can see
                             by how much it fell short
 already running             another pass is in flight for this session
 discarded: history changed  the client rewound past the turns it prepared while it ran
```

The dashboard's **Run GC now** prepares a cut with `min_cut_tokens` taken as zero and
switches to it at once, waiving the switch policy; every other rule still holds.

### What a collection reports

A switch reports its **outcome**, by turn: which of the cut's turns were
kept, compacted, archived or folded out of view, and which turns condensed earlier
it showed with less detail or with more. Its token totals describe the cut's turns
by the level the plan gave them. A pass that stopped for
"no saving" reports the outcome it *would* have had, and the context it would have
produced. A cut that waits reports why, and what it is waiting for.

### When a summary or the facts fail

The span is **still made**:

```
 summary refused or failed  ──► the span cannot be summarised: the plan keeps it whole
                                 (or archives it, if its score says so)
 facts could not be read    ──► the collection's spans have no facts: archived,
                                 each shows its opening line alone
```

Both are reported as degradations, and nothing is lost: the span holds its turns. The
staged cut keeps them, so the collection that later switches it in reports them too, not
only the pass that met them.
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

**Never in the middle of a tool round.** A request that answers a reply's tool calls
must send that reply back with its thinking intact; stripping it is refused, and
keeping it after the front changed is refused too, since the history before it is no
longer the one it was made against. So no cut is switched in on a request whose newest
message carries tool results, not even one Run GC now asked for: the first request
after the round carries it. A round can run for many requests, and the cut waits that
long, with the model sent everything word for word meanwhile. *Rejected: asking the
provider to drop invalidated thinking instead of refusing (its `drop_block` setting)* —
it needs a beta header and a request field added to the client's request, which a
gateway may refuse, and the model would lose the reasoning of the call it is in the
middle of.

The switch happens as that request is built, so the request that switches is by
definition the first to carry the front, and the boundary is the turns it sends:

```
request N    answered against the OLD front              reply N produced against the OLD front
request N+1  the switch happens; first to carry it       sends reply N back  -> its thinking is stale
             boundary = the turns it sends                reply N+1 produced against the NEW front
request N+2  same front                                   reply N+1 keeps its thinking
```

**A rewind moves the boundary back, never forward.** When the person edits a turn the
front does not stand in for, every turn after the edit is discarded, and the replies that
follow are produced against the front still in place, so they keep their thinking. Left
where it was, the boundary stripped valid thinking from each new reply up to it, and from
one mid tool round the provider could refuse the request. Before the front's first request
the boundary is not set yet, and that request sets it from the history it carries.

The thinking a switch drops is part of what it saves, which the switch policy weighs
([when to collect](when-to-collect.md)).

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

- **Change the prompt only at the front, only at a switch.** The one rewrite the
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
- **Prepare on what has gone quiet; switch on the cache and on health.** Preparing costs
  LLM-GC's tokens and changes nothing the client sees, so a flat floor is the right gate
  for it. Switching costs the provider's cache, so it waits for a cold cache, for a saving
  that repays the rewrite, or for the model's stale weight to reach its bound. See
  [when-to-collect.md](when-to-collect.md).
- **Hold a prepared cut rather than commit it.** Committing as soon as a pass finishes
  fixes the cache rewrite at an arbitrary moment. A staged cut can be switched in at the
  request that would miss the cache anyway.
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
  deterministic and the generations change only when a switch commits, so the
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

- **A switch rewrites the cache once.** The switch policy decides when that is worth it.
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
