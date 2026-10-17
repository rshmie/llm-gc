# When to Collect

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

### The provider's prompt cache, in plain words

Every request sends the whole conversation again. To avoid reading it all from scratch
each time, the provider **remembers the prompt it has just read**, for a few minutes.

Think of a reviewer who has just read the first 50 pages of your document. You send it back
with one new page at the end:

- **Back within a few minutes, first 50 pages unchanged:** they skip what they have already
  read and read only the new page. Fast and cheap. This is a **warm** cache.
- **Back after they have forgotten, or they never kept it:** they read all 51 pages from
  the start, at full price. This is a **cold** cache.
- **Page 1 changed:** they cannot trust anything after it, so they read everything again,
  even if they remembered it. That is what a collection does: it replaces the start of the
  conversation with the recap.

That last point is why *when* a collection takes effect matters:

```
                         don't collect now                   collect now (swap in the recap)
                         ─────────────────────────────────   ─────────────────────────────────────────
 WARM (they remember)    reads only the new turn: cheap      must read everything again: EXPENSIVE
 COLD (they forgot)      reads everything: full price        reads everything, but less of it: CHEAPER
```

When warm, collecting throws away something that was nearly free; it is worth it only if
the saving is big. When cold, the full read happens anyway, so collecting makes it smaller
and costs nothing extra. The rest of this section says the same thing precisely.

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

A switch also stops sending the model's earlier thinking. Each reply's thinking is bound to
the prompt it was written against, so the first request after a switch sends every earlier
turn without it ([the cut](the-cut.md)). That thinking is billed as input on every request
until then, so it is part of what a switch saves, and none of it is written again:

```
                      the messages, with the thinking each reply carries
 before a switch      t0 t1 … t39 │ t40 … tN          + thinking of every reply since the last switch
 after a switch       [FRONT] t40 … tN [new]          + no thinking: all of it predates the front
                      └──── written again ────┘
 saving   = what the recap saves against the turns it replaces  +  all the thinking still sent
 rewrite  = the front and the tail, text only
```

In one live conversation, LLM-GC's text alone predicted a drop of about 8,000 of the
provider's tokens at a switch; the provider's own count fell by about 24,000.

What a switch buys: every later request carries fewer tokens, and the model reasons over
less that has gone quiet. A cached read costs a small fraction of normal input, while
tokens the provider must write again cost at least normal input (on one provider's
published rates at the time of writing, a read is a tenth or less of the base price and a
five-minute write is 1.25 times it). So **tokens saved are not money saved**: a switch that
removes 20,000 tokens from a warm prompt and makes the provider write 26,000 again buys
back its rewrite only after many more requests.

### Warm and cold

- **Warm:** the provider still remembers this conversation's prompt, so the next request
  is cheap as long as its start is unchanged.
- **Cold:** the provider has forgotten it (the conversation sat idle too long), or never
  kept it (the client does not use the cache). The next request is read in full whatever
  LLM-GC does, so a switch on it costs nothing and makes that full read smaller.

```
 time ──────────────────────────────────────────────────────────────────────────────►

 requests   ●  ●  ●  ●  ●                                                     ●
            └ a burst: the provider remembers (WARM) ┘                        │
 passes     ▲  ▲  ▲  ▲  ▲  → a cut is staged, waiting                         │
                           └──── idle for longer than the cache lifetime ─────┤
                                 the provider forgets (COLD)                  ▼
                                          this request is read in full whatever happens:
                                          switching here costs nothing and makes it smaller
```

How the proxy tells which, precisely. Three observations decide it, all read from the
conversation's own traffic; no vendor figure is kept:

- whether any of its replies has reported cached tokens, read or written. Once one has,
  the conversation counts as using the cache: a later reply too short to be cached does
  not mean it stopped;
- when its newest answered request began, because the cache's lifetime counts from the
  start of the request that last read or wrote the entry, not from the end of its reply.
  A refused request never reached the cache and says nothing about it;
- the lifetime its last cache marker asks for, in prompt order, because that is the entry
  covering the conversation; with none named, the wire format's documented default.

The cache is cold when the conversation does not use it, or when it has been idle longer
than the lifetime plus a margin of a few seconds, which absorbs the gap between when the
proxy forwards a request and when the provider receives it. A conversation that does not
use the cache is always cold: every token is billed in full, so a smaller prompt is
strictly cheaper. **When the observations cannot settle it, the cache counts as warm**,
including before any request has been answered: wrongly warm costs one missed free
switch, wrongly cold costs a rewrite.

The proxy keeps these observations per conversation and hands the engine only the
answer, so the engine has no clock and no knowledge of any provider.

### The switch policy

Everything turns on one measure, **stale weight**: the tokens of the staged cut's turns
that the plan condenses, with the thinking those turns still send, that is, what the model
is still carrying that has gone quiet.
Three zones on one axis:

```
 stale weight ──►

 0 ─────────── floor ───────────────────────────────────────────── bound ───────────────►
   nothing         │ a cut is prepared and staged                   │ overdue
   to prepare      │ it is switched in only when the cache is cold, │ switched in at the next
                   │ or when the saving repays the rewrite          │ request, whatever it costs
                   │                                                │
   min_cut_tokens ─┘          max_stale_ratio × the model's window, in LLM-GC's count ─┘
```

While a request is being built with a cut staged, the policy takes these in order and
stops at the first that holds:

```
 0  does the request answer tool calls?       yes ─► WAIT    tool round open       (even if requested)
 1  would the switch shrink the prompt?       no ──► WAIT    no saving
 2  did the developer press Run GC now?       yes ─► SWITCH  requested
 3  is the cache cold?                        yes ─► SWITCH  cache cold            (free)
 4  saving ≥ share × tokens written again?    yes ─► SWITCH  repays its rewrite    (expected to pay back)
 5  stale weight ≥ the bound?                 yes ─► SWITCH  stale weight at bound (health; may cost more)
 6  otherwise                                     ─► WAIT    waiting for payback
```

- **A tool round holds every switch back** ([the cut](the-cut.md)): the reply whose tool
  calls the request answers must keep its thinking, which a new front invalidates. Run
  GC now pressed mid round is remembered and made by the first request after it; it is
  forgotten if the cut it asked for is discarded. The dashboard learns whether a round
  is open from the proxy's request log: open while the newest request is unanswered, or
  once its reply ended in tool calls.
- **Saving** is how many tokens the switch removes from the prompt: the recap's saving,
  measured on the real summaries when the cut is staged, plus all the model's earlier
  thinking still sent, measured as the request is built. **Tokens written again** is the
  size of the messages after the switch, text only, since no turn sends its thinking then.
  The system prompt and tools are read either way and drop out of the comparison.
- **Thinking is an estimate**, made by the proxy from each reply's usage
  ([the proxy](../proxy/overview.md)); a conversation without it measures as text alone.
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
- A switch's cut never makes the prompt larger. A span restored because the
  conversation returned to it costs tokens on purpose, bounded by the recap budget.
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
The window is read per model and converted to LLM-GC's count at the rate the
conversation's replies measure, as the recap's budget is
([detail levels](detail-levels.md)).

### What the dashboard shows

Before a cut is prepared, the next-collection card shows the two conditions for preparing
one. Once one is prepared, it shows the three ways it is switched in, each with its state:
whether the cache is cold, with the idle time and lifetime behind that; its saving against
the share times what it would make the provider cache again; and its stale weight against
the bound. Its first line says whether the next request switches it in and why, or that it
waits and what switching now would cost. A cut that waits always says so.

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
- **The bound and the stale weight are counted alike.** Stale weight is LLM-GC's count and
  the window the model's, so the bound is taken of the window converted at the measured
  rate; otherwise 10% would mean about 15% of a Claude window, and something else on each
  model. *Rejected: the bound as a share of the prompt instead of the window* — it needs no
  conversion, but it fires hardest when the saving is smallest: a 10,000-token
  conversation with 2,000 gone quiet is at 20%, and a warm cache would be rewritten to
  save little.
- **The share is a ratio of tokens, not a price.** A ratio and an observed cache state need
  no rates, and rates differ between models, vendors and releases.
- **A cut is switched in whole.** The rewrite does not depend on how much is cut, so a
  partial switch costs the same for a smaller saving.
- **The thinking a switch drops is measured at the switch, not when the cut is staged.**
  Every reply after staging adds thinking the switch would also drop, so a figure taken at
  staging would grow stale and keep the cut waiting. The recap's own saving is fixed once
  its summaries exist, so it is taken at staging. *Rejected: counting thinking in what is
  written again* — after a switch no turn sends its thinking, and counting it there would
  cancel the saving it is part of. *Rejected: leaving thinking out* — the saving then reads
  as a fraction of the real one, and cuts wait far longer than the policy intends.
- **The cut's own check stays in text.** A pass stages a cut only if its recap is smaller
  than the turns it replaces. Any switch drops the thinking, so the thinking says nothing
  about whether this recap is a good one.
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
- **The model loses its recent reasoning at every switch**, the tail's included, and counting
  the thinking makes switches come sooner. The provider recommends this shape of compaction,
  and the summary keeps what was decided, but a reply right after a switch may think again
  to rebuild what it lost. Its output, compared with the replies before the switch, shows
  how much.

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
