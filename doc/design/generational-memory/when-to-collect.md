# When to Collect

> **Being built.** The staged cut and the switch on the request path are not in the code
> yet: until they are, a pass commits its cut as soon as it finishes. The switch policy is
> the first piece to land.

## How it works

A collection has two halves, and they cost different things, so they have different
triggers.

- **Preparing a cut** is the background work: draft the spans, plan their levels, read
  the facts, write the summaries, plan again. It costs LLM-GC's own model tokens and
  changes nothing the model is sent. Its result is a **staged cut**, held ready.
- **Switching to it** is the moment the prompt changes: the staged cut becomes the
  front, and the request being built is the first to carry it. It makes no model call. It
  costs the provider's prompt cache, because from the front onwards the prompt no longer
  matches what was cached.

```
 BACKGROUND, after each answer                 REQUEST PATH, building the next request
 spends LLM-GC's own tokens                    spends the provider's cache; no model call
 ──────────────────────────────────────        ─────────────────────────────────────────────────
 enough gone quiet?          no ─► nothing     is a cut staged?             no  ─► send as is
   │ yes                                         │ yes
 would the cut shrink the prompt,              would it shrink the prompt?  no  ─► wait
 at estimated summary sizes?  no ─► stop       did the developer ask?       yes ─► switch
   │ yes                                       is the cache cold?           yes ─► switch  (free)
 PREPARE: read facts, write summaries,         saving ≥ share × rewritten?  yes ─► switch  (repays)
 plan again, measure the real saving           stale weight ≥ the bound?    yes ─► switch  (health)
   │                                           otherwise                        ─► wait, says why
   ▼
 STAGED CUT ──────────────────────────────►    switch: the staged cut becomes the front, and
 prompt unchanged; extended by each pass       this request is the first to carry it
```

The left half decides whether a cut is worth *paying for*; the right half decides whether
a paid-for cut is worth *switching to* now.

What decides *what* is condensed and *how* is unchanged: the plan scores every span against
what is being discussed now and gives it a level (see [the-cut.md](the-cut.md)). This
document is about *when* the prompt changes.

### What a switch costs, and what it buys

Provider caches match a prompt from its first byte. A switch changes the message after
the system prompt and tools, so those still match and everything behind the front does
not:

```
                      tools + system     the messages
                      ──────────────     ──────────────────────────────────────────────
 cached before        ▓▓▓ read ▓▓▓       t0 t1 … t39 │ t40 … tN
 a switch                                 └ to become the front

 first request        ▓▓▓ read ▓▓▓       [FRONT] t40 … tN [new]
 after a switch                          ░░░░░░░░░░ all written again ░░░░░░░░░░░

 later requests       ▓▓▓ read ▓▓▓       [FRONT] t40 … tN … tN+k
                                         ▓▓▓▓▓▓▓▓▓▓ read again ▓▓▓▓▓▓▓▓▓▓ │ new
```

The rewrite is the front plus the tail, **whatever the size of the cut**. Cutting more does
not make it dearer; it makes what is rewritten smaller. That is why a staged cut is
switched in whole or not at all.

What a switch buys: every later request carries fewer tokens, and the model reasons over
less that has gone quiet. A cached read costs a small fraction of normal input, while
tokens the provider must write again cost at least normal input (on one provider's
published rates at the time of writing, a read is a tenth or less of the base price and a
five-minute write is 1.25 times it). So **tokens saved are not money saved**: a switch that
removes 20,000 tokens from a warm prompt and makes the provider write 26,000 again buys
back its rewrite only after many more requests.

### Warm and cold

A conversation's cache is **warm** when the next request would read what the last one
wrote, and **cold** when it would miss anyway. A switch on a cold request costs nothing: the
whole prompt is written regardless, and a smaller prompt is written for less.

```
 time ──────────────────────────────────────────────────────────────────────────────►

 requests   ●  ●  ●  ●  ●                                                     ●
            └ a burst: each request reads what the last wrote ┘               │
 passes     ▲  ▲  ▲  ▲  ▲  → a cut is staged                                  │
                           └──── idle for longer than the cache lifetime ─────┤
                                 the provider's entry expires                 ▼
                                          this request misses the cache whatever happens:
                                          switching costs nothing and writes less
```

Three observations decide it, all read from the conversation, none stored:

- whether the conversation's replies report cached tokens at all, read or written;
- when its previous request began, because the cache's lifetime counts from the start of
  the request that last read or wrote the entry;
- the lifetime, taken from the request when it states one, and otherwise the wire
  format's documented default.

The cache is cold when the conversation does not use it, or when the gap since the
previous request exceeds the lifetime. A conversation that does not use the cache is
always cold: every token is billed in full, so a smaller prompt is strictly cheaper.
**When the observations cannot settle it, the cache counts as warm**: wrongly warm costs
one missed free switch, wrongly cold costs a rewrite.

### The switch policy

Everything turns on one measure, **stale weight**: the tokens of the staged cut's turns
that the plan condenses, that is, what the model is still carrying that has gone quiet.
Three zones on one axis:

```
 stale weight ──►

 0 ─────────── floor ───────────────────────────────────────────── bound ───────────────►
   nothing         │ a cut is prepared and staged                   │ overdue
   to prepare      │ it is switched in only when the cache is cold, │ switched in at the next
                   │ or when the saving repays the rewrite          │ request, whatever it costs
                   │                                                │
   min_cut_tokens ─┘                    max_stale_ratio × the model's window ─┘
```

While a request is being built with a cut staged, the policy takes these in order and
stops at the first that holds:

```
 1  would the switch shrink the prompt?       no ──► WAIT    no saving
 2  did the developer press Run GC now?       yes ─► SWITCH  requested
 3  is the cache cold?                        yes ─► SWITCH  cache cold            (free)
 4  saving ≥ share × tokens written again?    yes ─► SWITCH  repays its rewrite    (expected to pay back)
 5  stale weight ≥ the bound?                 yes ─► SWITCH  stale weight at bound (health; may cost more)
 6  otherwise                                     ─► WAIT    waiting for payback
```

- **Saving** is how many tokens the switch removes from the prompt, measured on the real
  summaries. **Tokens written again** is the size of the messages after the switch.
  The system prompt and tools are read either way and drop out of the comparison.
- **The order only affects the label.** Each switch carries the cheapest reason that
  holds, so a switch labelled *stale weight at bound* is one that was warm, did not repay
  its rewrite and was forced: the only automatic kind that can cost more than not
  collecting.
- **Setting `min_saving_share` or `max_stale_ratio` to zero** collects as soon as there is
  a saving, as a pass did when it committed on finishing.
- **Setting the share high and the bound high** approaches collecting only when the cache
  is cold.

### Two promises

**The model is never worse off than without LLM-GC.**

- While a cut waits, the model is sent the original turns, word for word, exactly as
  without LLM-GC. Waiting cannot lose a decision; a lossy summary can.
- A switch never makes the prompt larger.
- What the model carries that has gone quiet is held under the bound: a cut that reaches
  it is switched in at the next request. The bound holds as long as passes complete; a
  paused or failed pass is reported, and the conversation grows until it can run.
- A waiting cut is shown, with its reason. Deferral is never silent.

**The bill is never worse than without LLM-GC, unless health forces it, and then it says so.**

- Cold: free. Warm: only when the saving is at least the share times what is rewritten,
  which is when the rewrite is expected to be repaid within a plausible run of requests.
- Only the stale-weight bound, and the developer's own Run GC now, can switch in a cut that
  costs more than it saves.
- Two honest limits: "repays" is an expectation, because payback depends on how many
  requests follow inside the cache lifetime; and LLM-GC's own summary and fact calls are a
  separate bill, metered and shown beside the conversation's, not part of the test.

### Other providers

The engine never sees a vendor's field name or a price. The wire-format translator reads
the cached tokens in each reply and the lifetime a request states, and hands the engine a
cache state. What differs between vendors, whether a write carries a premium, how much a
read saves, how long entries live, whether caching is automatic or marked by the client,
also differs between models of one vendor and between releases, so none of it is stored.

- Every prompt cache in use matches from the first byte, so "a changed front misses
  everything behind it" holds everywhere.
- A provider that reports no cached tokens is treated as cold, so a cut is switched in as
  soon as there is a saving: the right behaviour for a conversation with no cache to lose.
- The default share suits a steep write premium. Where writes are free it only makes the
  policy more patient than it needs to be, and the bound still caps the wait.
- Only the Anthropic wire format is read today.

### Settings

| Setting | The question it answers | Default |
|---|---|---|
| `min_cut_tokens` | How much must have gone quiet before a pass is worth its model calls? | 1000 |
| `min_saving_share` | When the cache is warm, how many times what a switch makes the provider write again must it remove? | 1.0 |
| `max_stale_ratio` | At most how much stale weight, as a share of the model's window, may the model carry before a switch is forced? | 0.10 |

All three are drafts, to be tuned by watching the dashboard and against the benchmarks.
The window is read per model, as the recap's budget is.

### What the dashboard shows

The next-collection card names, for the staged cut: its stale weight against the floor and
the bound; its saving against the share times what it would rewrite; the cache state and
the gap behind it; and the zone it is in, with the reason it switched or is waiting. A cut
that waits says what it waits for.

## Why it exists

A collection that is made whenever enough has aged out fixes the moment of its cache
rewrite at an arbitrary point, whether or not the next request would have missed the cache
anyway. Measured on a live Claude Code session, that was 11 collections in 26 requests:
each rewrote the messages behind the front, and the session cost more with LLM-GC than
without, because tokens saved were read at a fraction of the price of the tokens written
again. A flat token floor cannot see this: the right size depends on what sits behind the
front.

Cost is not the only force. The project exists so that earlier decisions stay usable and
the developer can see what the model carries; fewer tokens are a byproduct. So the design
separates the two: **health sets the deadline, and the cache picks the moment before it.**
LSM-tree databases schedule compaction the same way, during quiet time but forced at a hard
limit.

## Why this shape

- **Two halves with two triggers.** Preparing costs LLM-GC's tokens and changes nothing the
  client sees; switching costs the client's cache. One trigger for both cannot be right for
  either.
- **The switch is decided as the request is built.** That is the only moment the cache
  state is known, and the commit is one synchronous step with no model call, so the
  request path still never waits on the background.
- **The bound counts stale weight, not window fill.** It counts what the plan says has gone
  quiet, which is the generational premise, and it scales with the model through the
  window.
- **The share is a ratio of tokens, not a price.** A ratio and an observed cache state need
  no rates, and rates differ between models, vendors and releases.
- **A cut is switched in whole.** The rewrite does not depend on how much is cut, so a
  partial switch costs the same for a smaller saving.
- **Unsure means warm.** The wrong guess in that direction costs an opportunity, not money.
- **Prior art.** A provider's own context editing has the same two controls: a size trigger
  for when to look, and a minimum to clear, there to decide whether clearing is worth
  breaking the cache. This adds what such an API cannot know: a cold cache, and a deadline
  stated in what has gone quiet.

### Alternatives rejected

- **A larger flat floor.** The right size depends on what sits behind the front, and a flat
  figure cannot see a cold cache. It stays as the floor for *preparing*, where a flat figure
  is right: a pass has a fixed overhead whatever the conversation's size.
- **A price model** (write premium, read discount, requests still to come). The textbook
  answer, but the rates differ between models and vendors, and the horizon is unknowable.
- **Collecting only when cold.** The simplest rule that suits the cache, but a long burst
  never shrinks and the model carries unbounded noise. The bound removes that while keeping
  every cold switch.
- **A window-fill trigger.** A capacity rule: it neither sees the cache nor whether what
  fills the window has gone quiet.
- **A timer that switches when the cache expires.** A request arriving just after expiry goes
  out on the old prompt and rewrites the cache with it; the next request, now warm, pays the
  rewrite. Only the request being built knows it is cold.
- **A cache breakpoint on the front, or a front that only grows.** It could save at most the
  front's share of a rewrite, since the tail is rewritten regardless, and it would stop the
  older spans being re-planned, which is how a span the conversation returns to is shown in
  detail again. Providers also cap the breakpoints in a request, and the client uses some.
- **Committing as soon as a pass finishes.** It fixes the rewrite at an arbitrary moment.

### Accepted costs

- **In a warm burst the model carries stale material**, up to the bound, for the sake of the
  bill. The bound is the setting that trades them.
- **Fewer tokens saved in short warm sessions.** A saving that costs more than it saves is
  not counted as one.
- **A staged cut is as of the last pass.** If the conversation returns to an old topic
  during a long idle, a span staged as condensed stays so until the next collection plans
  it again.
- **Summaries are bought for a cut that may never be switched in**, if the session ends
  first: at most one staged cut's worth.
- **More state.** One staged cut per session, and a switch on the request path.
- **The default share is patient where writes are free.**

## Known gaps

- **The lifetime is assumed when a request does not state it.** Learning it from the hits
  and misses a conversation has seen would replace the assumption, and would serve a
  provider that publishes none.
- **A client that caches only its system prompt looks as if it caches everything**, so its
  cuts wait longer than they need to: the safe direction.
- **Other reasons a request misses the cache anyway** (a changed system prompt, tool set or
  model) are not yet read as cold.
- **The bound's default is a draft.** The right value comes from replaying the
  decision-survival check at several bounds and seeing whether recall drops.
- **A switch can land on a tool-result request**, where the reply that made the call loses
  its thinking for that step. A policy that is not cold and not overdue could hold back
  for a request that is not one.
