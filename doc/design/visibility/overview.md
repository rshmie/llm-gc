# Visibility Layer — Overview

A model answers in the same confident tone whether its context is fresh or
full of noise. There is no built-in signal that says *"the foundation under
this answer is unreliable"* — and the first signal the developer usually gets
is a wrong answer. The visibility layer exists to be the **earlier** signal:
to surface what state the context is in *before* the wrong answer, so the
human can decide when to trust the output.

That is not a feature bolted onto the engine. It is the project's reason for
being — *epistemic transparency*. The retention and control work exists in
part to give this layer something coherent to be transparent about.

### Transparency, not observability

These are two different jobs, and conflating them is the fastest way for this
layer to drift away from its purpose.

|  | Transparency | Observability |
|---|---|---|
| Asks | What does the model know, at what fidelity, and what has it lost? | What did LLM-GC do, how long did it take, did it fail? |
| Subject | The conversation | The tool |
| Reader | A developer judging whether to trust an answer | Whoever operates or debugs LLM-GC |
| Status here | The reason the project exists | Necessary, and not the point |

Observability is a well-understood discipline and LLM-GC needs it: a cleanup
engine whose own behaviour is unexplainable is not production-grade. But a
surface reporting level counts, generation sizes and pass durations
is describing *the collector*. Nothing in it answers the question the
developer actually has, which is whether the answer in front of them rests on
solid ground.

The consequence is an ordering, not an exclusion. **The knowledge-state view
leads; the engine-activity view sits one level down.** Both read the same
monitor snapshot. What separates them is which question the display is
composed to answer — and the developer's attention is the scarce resource
being allocated.

### Two pieces

The layer is two pieces, with a deliberately clean split:

- **The data sources.** The Health Monitor listens to the event bus
  continuously and keeps, per session, a live answer to *"what has LLM-GC
  done to this conversation?"* — its last pass, its history of passes,
  transitions and degradations. The knowledge-state projection answers *"what
  does the model know?"* from a read-only inspection of the session. Neither
  renders anything.
- **The Visualizer (dashboard) — the view.** It renders what those report. It
  computes nothing of its own, and its one action is the developer's Run GC now.

This document covers the layer as a whole and the data sources in depth.

---

## Why it exists

The motivating asymmetry: an AI model has no reliable internal sense of what
it knows and does not know, so a human cannot lean on the model to flag when
its grounding is weak. The human *can* exercise that judgment — but only with
data. Principle 4 of the project (*the human exercises judgment; the system
provides data*) is the operating form of this. The visibility layer is where
that principle becomes real.

What would break without it: context management would happen **invisibly**.
The engine would score, summarise, archive and fold — quietly changing the
material the model reasons over. The developer would have no way to see that
any of it happened, let alone judge whether it was safe. A system that
manages context invisibly has replaced one opaque actor (the model) with two.
That is the opposite of the project's intent.

The Health Monitor is also where the *continuous* view of context state
becomes real. Context health is a property of the conversation **as it
evolves**, not a property of garbage-collection operations. A view that only
existed at collection time would leave the developer staring at stale state
during the very reasoning the cleanup was meant to protect.

---

## How it works

### A passive subscriber with a hybrid data model

The Health Monitor subscribes to five events: a pass finished, a span made at
the summarised level, a span archived (made archived, or lowered to archived or
out of view), a fact superseded, and part of a collection falling short. It is the same publish/subscribe channel every
other consumer (logger, benchmarks, the dashboard) reads from; the monitor is
just one more subscriber, and the bus neither knows nor cares that it is
listening.

**One monitor serves every session, and keeps them apart.** Every event names
the session it describes, and the monitor keeps a separate record per session:
its last pass, its transitions, and its recent degradations. Events with no
session share one record. The number of records
is bounded, dropping the session heard from least recently, so a monitor never
grows with the number of conversations a proxy has ever served.

**It keeps no conversation content.** A finished pass's signals are built the
moment the event arrives and only those are kept — counts, tokens, turn
indices — never the result itself, which carries the whole context. Holding it
would keep every conversation's text alive in the monitor after its session
closed.

Its data sourcing is deliberately hybrid — *aggregator, not computer*:

```
   PUSHED: what happened                        PULLED: what is true now
 GcService                                    Session snapshot
     │ events, after each commit                  │ read-only, no model call
     ▼                                            ▼
 ┌ HEALTH MONITOR, per session ──┐          ┌ KNOWLEDGE STATE ──────────────┐
 │ the last pass's signals       │          │ every span, and exactly what  │
 │ transitions (append-only)     │          │   the model reads for it      │
 │ recent degradations           │          │ every turn, through its span  │
 │                               │          │ every fact, and if it is sent │
 │                               │          │ the next collection, planned  │
 └───────────────┬───────────────┘          └───────────────┬───────────────┘
                 └──────────────► DASHBOARD ◄───────────────┘
                     polls once a second · computes nothing
```

- **Current-state signals** (token budget, the latest pass's breakdown, each
  generation's contents) are **read** from the most recent pass's result and from
  the session's real storage, never re-derived from accumulated events. The
  components that computed those numbers are the authority on them.
- **What a collection moved is part of its own record**, not inferred. Its
  outcome names every turn it took, at the level it gave each, and every turn
  condensed earlier that it showed with less or more detail; the dashboard's
  timeline is drawn from these records, one entry per collection, and a fact is
  listed as replaced by the collection that took the turn replacing it. *Rejected:
  a log of move events, grouped by time.* It recorded only spans made or lowered,
  so a raised span never appeared; its groups merged collections that ran close
  together; and a bounded log could hold part of a collection's moves as if they
  were all of it. *Rejected: inferring moves by diffing each pass's decisions.* A
  decision is not a move: diffing missed moves made in the same pass a turn
  arrived, and invented moves for passes that committed nothing. The rule: *only
  the component that performs a state change is the authority that it happened.*
- **The monitor still logs each move and supersession as it is announced**, for a
  subscriber that wants moves as they happen rather than per collection.
- **Generation contents are read from storage**, for the same reason: the young
  generation is the turns a pass forwarded word for word, and the old generation
  is every span, with what the front spends on them at their levels.

The snapshot it exposes is **built whole on every read** — never cached,
never mutated field-by-field — so a reader can never observe a torn,
half-updated state.

It is **passive**. It observes and reports; it never triggers a collection
pass itself. Deciding *when* to clean context is a decision with consequences
for what the model produces next, and that decision stays with the developer
or the application. The dashboard's one control, **Run GC now**, does not
contradict this: it is the *developer* pulling the lever explicitly — the human
judgment the passivity rule exists to protect. It runs the same aging pass with
only the batch minimum waived, so every other rule of a cut still holds and the
result is reported like any automatic pass.

### What it exposes — named signals, no aggregate

The monitor's public output is a **snapshot**: an immutable read-out of its
current state at a point in time (the `ContextHealth` model). The snapshot
carries a set of named, individually-explainable signals, grouped by which
of the two questions they answer.

**Transparency signals — about the conversation.** These lead, because they
are what the developer acts on.

| Signal | What it measures | The question it answers |
|---|---|---|
| **Knowledge state** | every span with its level, its score, and the exact text the model reads for it; every recorded turn with its words and how the model receives it — word for word, or through its span kept, compacted, archived or out of view; every fact with how it reaches the model and what replaced it | "What does the model still know, and how well?" |
| **Generation lifecycle state and recent transitions** | current young / old / permanent contents (counts and token shares) plus the last *N* promote / archive / supersede transitions | "Where does my decision live now — and did anything supersede it?" |
| **Token budget pressure** | current context tokens as a fraction of the configured budget | "How crowded is the context the model is reasoning over?" |

**Observability signals — about LLM-GC.** These sit one level down.

| Signal | What it measures | The question it answers |
|---|---|---|
| **Latest GC breakdown** | the most recent collection's turns by the level it gave them — kept, summarised, archived — with their tokens, plus the run's status, timing, and failure detail | "How much did the last cleanup change — and did it succeed at all?" A bypassed or failed pass is visible, never hidden behind a normal-looking breakdown; signals honestly report `None` ("not measured yet") rather than a fabricated zero before any pass has run |

**Knowledge state is the signal that most directly serves the project's
purpose.** The other signals describe *collector activity*: counts, sizes,
durations, transitions. A developer reading "old generation holds 21
spans, pressure 0.64" cannot get from there to "the model no longer has the
database decision word for word" — and that inference is the entire product.

Knowledge state is that inference made by the system. It pivots one session's
data from *"what did the collector do"* to *"what does the model know"*, keyed
by turn and by fact:

```
 turn   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
       ░░ ░░ ░░ ░░ ▓▓ ▓▓ ▓▓ ▓▓ ▒▒ ▒▒ ▒▒ ▒▒ ██ ██ ▒▒ ▒▒ ┄┄ ┄┄ ┄┄ ┄┄ ┄┄ ┄┄
                                                       ▔▔ ▔▔ taken by the next collection

       ░░ out of view   ▓▓ archived   ▒▒ compacted   ██ kept   ┄┄ verbatim, in the tail
```

Here every span is one exchange of two turns.

- **Per span**, its level, its score now, and the exact text the front carries for
  it, read from the rendered front rather than rendered again, so it is byte for
  byte what the model reads. A folded span carries its run's shared line.
- **Per turn**, how the model receives it, and its words. A turn in the tail is
  verbatim. Every other turn is in exactly one span and is received the way its
  span is shown: quoted, under its summary, by its opening line and facts, or only
  as a range on a folded run's line. A reply that yielded no fact of its own is no
  less visible than the question it answered — the model reads their span's line
  for both. The words come from the span, which keeps them.
- **Per fact**, how it reaches the model: on its span's line when the span is
  compacted or archived; inside the quoted words when it is kept; not at all when
  it is folded out of view. *Stored, not sent* is the signal that matters most: the
  conversation has not forgotten it, and the model cannot see it. Each fact says
  who stated it — you, or the assistant — and a fact later replaced names the turn,
  and the speaker, that replaced it.
- **The next collection**: which verbatim turns it would take, as which spans and
  at what level; which spans already collected it would show at another level; and
  the conditions it must meet, each with its state — a batch worth
  collecting, against `min_cut_tokens`, and a cut that pays for itself. The first
  is planned exactly. The second is estimated at each summary's target size —
  the very estimate a collection checks before paying for any summary, so a cut
  the panel shows as not paying is one the collection skips — and measured again
  on the real summaries; when that refuses a cut, the panel shows the attempt's
  figures too. *Rejected: a progress meter alone.* It reads as a promise, and on a conversation of short messages the
  meter filled, Run GC now was pressed, and nothing happened, with nothing on the
  panel saying why.

It is built from a single read-only **inspection** of the session that the
engine produces with the same rendering that builds the prompt and the same
draft and plan a real collection runs, so a dashboard showing it cannot disagree
with what the model is sent or with what the next collection will decide.

The projection belongs to the monitoring package, not to the view. The view
computes nothing of its own (see *Two packages*), and deciding that a turn is
archived rather than quoted is a derivation over spans and facts, not a
rendering choice. Putting it in the visualizer would duplicate that
logic in every future consumer — the post-session report and the benchmark
harness both need the same answer.

The generation-lifecycle signal is the richest of the built three: its
temporal dimension — *where* a fact lives and *how long* it has lived there —
is information the developer's epistemic vigilance acts on directly. The
temporal view is first-class, not metadata. Knowledge state is, in effect,
that signal re-indexed by the thing the developer actually cares about.

The tracked transitions are not the same kind of move. **Promote** (a new span
made at the summarised level) and **archive** (a span made archived, or lowered to
archived or out of view) are changes in how much of a turn the model sees;
neither destroys anything, because the span keeps its turns. **Supersede** is
contradiction resolution within the permanent generation — a newer fact
overriding an older one. A span raised again, because the conversation returned to
it, is restored from its own words within the old generation; no turn ever returns
to the young generation.

The **transformation ratio** — the proportion of context that has been
compacted or archived relative to the total — is computed and kept as an
**internal** snapshot field. It is accessible for benchmarking and serves as
the baseline any future aggregate must beat. It is *not* surfaced as a
headline number (see *Why this shape*).

**Distinct from the span score.** These are conversation-level signals, read by
the developer. They are not the *span score* the scoring layer produces — that
score is computed per span and feeds the planner's choice of level. Different
scope (conversation vs. span), different consumer (developer vs. planner),
different purpose; the two share none of their internals.

### Two packages

The layer's two pieces live in two packages, because they have two jobs:

- **`monitoring/`** — the Health Monitor (`ContextHealthMonitor`), its
  snapshot (`ContextHealth`), and the knowledge-state projection. Pure Python;
  **no web dependencies**.
- **`visualizer/`** — the dashboard view, and later the post-session report.
  Web dependencies live here.

The dependency runs one way: the view reads the monitor. Keeping them
separate keeps the engine's in-process import graph — which is what wires the
monitor into the bus — free of any web-framework code. The doc area
("visibility") names the *layer*; the two packages are its
implementation.

The dashboard runs inside the proxy process, beside it rather than within it:
the same engine and the same event loop, on its own localhost port. It has two
views, one per question, and the transparency view opens first:

| View | Answers | Shows |
|---|---|---|
| **The conversation** | What does the model know, at what fidelity, and what has it lost? | Every exchange with exactly what the model reads for it — quoted word for word, its summary, its opening line and facts, or one shared line when folded out of view — every message as it was said, and every fact with how it reaches the model |
| **LLM-GC at work** | What did LLM-GC do, and did it work? | What the next collection would take; why every exchange is at its level, in the planner's own steps (the band its score earns, the margin, a level no cheaper than a richer one, the budget); every pass and its outcome, degradations, transitions, the proxy's counters, and the settings in effect |

It polls once a second, and fetches a session's content again only when it has
changed. Server-push (WebSocket) delivery is a planned upgrade of the transport,
not of the split.

**One vocabulary, explained where it appears.** The interface uses the code's
words for how much of a span the model sees, its *level* — *kept*, *compacted*,
*archived*, *out of view* — and for where a message lives, the *young*, *old* and
*permanent* generations. Four words are plainer on screen than in the code, and
each one's explanation names the code's word: a *collection* is a cut (a check that
moves messages out of the young generation), the *recap* is the front (the first
message of every request), a *check* is a pass, and a *recent* message is a verbatim
one, still in the young generation. Every term carries a short explanation where it
appears — hovering any category says what it means — and the guide maps each
on-screen word to the code's.

```
code      cut          front    pass    KEPT   COMPACTED   ARCHIVED   OUT_OF_VIEW   VERBATIM (young)
screen    collection   recap    check   kept   compacted   archived   out of view   recent
model     —            —        —       [Quoted exactly]  [Summarised]  [Archived]  [Out of view]
```

The last row is what the model reads in the front. Its markers say what each line
*is* — a quote, a summary — because a model reading `[Compacted]` has to guess what
was done; the dashboard explains *compacted* as summarised.

*Rejected: "word for word" for the young generation.* A kept exchange is quoted word
for word too, so the label could not tell a collected message from one no collection
has taken — which is the difference that matters: a recent message is sent as its
own message, and a kept one is quoted inside the recap and can lose detail later.

*Rejected: a separate plain vocabulary* (clean-up, condensed block, quoted whole,
kept as facts): read beside the code, two vocabularies for one system confuse more
than either, and "clean-up" reads as something that triggers GC rather than what GC
does after a reply. *Rejected: the code's words alone*: "cut" reads as truncation
and "front" means nothing on its own. *Rejected: the design vocabulary with a
glossary page*: a reader who must leave the screen to decode it has been handed
engine telemetry again. *Cost accepted:* cut and front differ between screen and
code, and the guide maps them. Token
figures compare like with like: the conversation as written is the session's
recorded messages at their own sizes, counted as the engine counts them. The newest
message reaches the session only with the next request, and counting it as written
showed a saving equal to that message which never happened.

**Run GC now explains its outcome, including when nothing changed.** When a
collection would not have shrunk the prompt, the engine reports what it would have
done — which messages, how many quoted, summarised or archived, and the
prompt size it would have produced — and the dashboard says why in those terms.
"It would not have shrunk the context" alone tells a reader nothing they can act
on.

**What was said is kept.** The engine keeps every turn it records, whole, for
the life of its session: a span in the old generation holds the turns it
condenses, so it can be shown at any level again. The dashboard shows a condensed
turn's words beside what the model receives instead, read from its span. A
transcript recorder the proxy feeds while the dashboard runs supplies only what
the session has not recorded yet — the newest request's own message, which joins
the conversation when a later request carries it with its reply — and each
conversation's opening words for the picker: text only, bounded by session count,
dropped with its session. That message is shown as the *newest request*, never as
something the user said: clients send side requests too (after each reply, Claude
Code asks the model to suggest the user's next prompt), and one arrives with the
same history, system prompt and tools as a real next message, so nothing in it
tells the two apart. *Rejected: recognising known side requests by their wording*
— it is tied to one client's current text, and anything it misses is still
attributed to the user. *Rejected: showing the message only while it is being
answered* — a chat client's latest message would vanish once its reply arrived.
For the same reason a turn carrying tool results is shown as *tool results*, not
as the user's words: the client sends what its tools returned under the user's
role. Instructions the client wrote into a message — Claude Code's
`<system-reminder>` blocks — show as a tag with their size, not among its words:
they are not conversation, are never condensed, and are not counted in the
request's size, which covers what LLM-GC can change. Their size is read from the
client's latest request, not from the record: a client can rewrite them after the
message was recorded, and they are sent as it sends them now. *Accepted cost:* the engine holds
every original in memory for every session, dashboard or not — the price of being
able to restore one. *Rejected: no originals* — an archived turn could
never be shown whole again, to the model or to a person.

**It shows conversation content, so it is opt-in and localhost-only.** The proxy
and the dashboard have different exposure: the proxy is an API meant, one day,
to be reached behind authentication; the dashboard shows every conversation the
process holds. Separate listeners keep the two from being exposed together. Two
further guards close the browser-side routes to localhost: requests whose `Host`
is not a localhost name are refused, so a hostile page cannot rebind its own
domain to 127.0.0.1 and read the dashboard as same-origin; and Run GC now
requires a custom header, which another site's page cannot send without a
preflight the server never grants.

---

## Why this shape

### Continuous, not per-collection

**Rejected: monitoring only during garbage-collection operations.** A health
view that updated only on a collection pass would blink in and out — a
developer inspecting the dashboard *between* operations would see stale
state, which is exactly when transparency is most useful (during the
reasoning, not after the cleanup). Because context health is a property of
the evolving conversation, the monitor tracks the conversation, not the
collection events. The cost of this choice is recorded under *Consequences*.

### Passive, not auto-triggering

**Rejected: a continuous monitor that also auto-triggers collection under
"pressure"** (the way a runtime garbage collector fires under heap pressure).
Auto-triggering is more proactive and has an appealing precedent, but pulling
the cleanup lever automatically — without an audit trail, and in a way that
changes what the model produces next — is contrary to *the human exercises
judgment*. Automatic triggering may make sense once a grounded predictor
exists to justify it, and even then it should be opt-in.

### Signals, not an aggregated score

This is the central decision. The view must serve epistemic transparency,
which has a hard implication: it must be **legible** at a glance *and*
**honest** about the limits of what it shows. Anything that looks more
authoritative than its underlying math supports is a worse failure than
showing nothing — because it puts a *new* black box (an inscrutable number)
in front of the developer, in place of the original one (the model).

One caution carries the whole decision: **a number can be perfectly accurate
and still be the wrong thing to show.** Accuracy and legibility are different
bars, and a transparency surface has to clear both. The case where that bites
hardest is the second one below.

Three candidate shapes were considered:

1. **A single likelihood-of-hallucination score** — *"given this context,
   how likely is the next answer to be wrong?"* This is what the developer
   most wants. **Rejected on honesty.** Absent ground-truth hallucination
   labels, a number labelled "hallucination probability" claims an accuracy
   a heuristic cannot back. It violates *the system is honest about its own
   outputs*.

2. **A single transformation-ratio score** — the fraction of context that
   has been transformed. **Rejected for the surface — but for a different
   reason than (1).** It is honest at the *math* layer: the ratio is a real,
   exactly-measurable property of the context. It fails at the *UX* layer. A
   single top-level number, however carefully labelled, becomes the
   glance-target and is read as authoritative; the qualifier ("transformation
   ratio, not hallucination probability") becomes decoration and does not
   survive contact with real UI behaviour. The math's honesty does not rescue
   the glance.

3. **A set of named, individually-explainable signals** — **chosen.** There
   is no single number to glance at, so there is no single number to misread.
   Each signal is named for what it literally measures. The *aggregation* —
   the question "given these signals, how worried should I be?" — is left to
   the developer's own epistemic vigilance, which is precisely the human
   capability the project exists to engage.

The transformation ratio (family 2) is **demoted, not deleted**. The monitor
still computes it and exposes it via the snapshot for benchmarks and research,
and as the baseline a future learned aggregate must beat. It is simply not a
surfaced headline. This preserves the defensible math without the UX hazard
of making it a glance-target.

Two further alternatives were rejected:

- **A weighted composite over the signals.** Gameable — any change to the
  weights yields a different headline number — and harder to defend than the
  signals it summarizes. The signals are more honest as independent displays.
- **No signals at all until a learned predictor exists.** The signals are
  genuinely informative on their own terms — they answer questions the
  developer can act on ("how full is my budget?", "how much did the last
  cleanup change?", "where do my permanent facts live?"). Shipping nothing until
  an ideal aggregate exists would mean shipping no visibility at all in the
  meantime, in a project whose whole point is visibility.

A learned hallucination predictor, done honestly against real labels, is a
**planned extension**. If it can be defensibly grounded — correlating with
real hallucination signals strongly enough that a developer should trust it —
an aggregator ships *behind this same signal interface*, alongside (not
replacing) the individual signals. If grounding turns out to be weak,
signals-only is the **permanent design**, and that is an honest answer to the
research question, not a failure. The project would rather ship several honest
signals than one composite whose construction the developer cannot inspect or
trust. **Aggregation is earned, not assumed.**

### Context-side signals today; response-side signals as an open direction

Every signal above is derived from **context state** — what was kept,
compacted, archived, extracted. The layer never reads the model's *response*.
That boundary is worth naming because the proxy is the one component in the
system that sees both sides: the context that was sent and the answer that
came back. Nothing else in the ecosystem is positioned to join them, because
nothing else knows what was removed.

Two response-side signals are credible, and they are not equally good.

**Grounding failure is the strong one.** When the model asks for information
the permanent generation already holds — *"which database are you using?"*
when a `database` fact is stored and was not injected — that is near-airtight
evidence that cleanup removed something load-bearing. It is specific, it has
high precision, and it is the only signal available from live traffic that can
**falsify** the *never make things worse* principle rather than merely
illustrate it. A close variant is the model restating something as unknown
that the session established earlier.

**Hedging is the weak one**, and weak in an instructive way. Hedged language
("I think", "it seems", "based on what I recall") is cheap to detect and
needs no labels, which makes it tempting as an early stand-in for the
hallucination signal the project cannot yet ground. But it is confounded by
prompt phrasing, question difficulty, system-prompt style, and model version,
so a rise in hedging cannot be attributed to cleanup. Worse, its failure mode
runs the wrong way: a model missing its grounding frequently answers
*confidently and wrongly* rather than hedging. Absence of hedging is therefore
not evidence of good grounding, which is precisely the inference a developer
would be tempted to draw. Hedging can be recorded as context alongside a
genuine signal; it cannot carry a claim on its own.

Both are correlational. The instrument that gets closer to causation is
**counterfactual replay**: re-issue the same query against the full,
uncompacted context and compare the two answers. That is expensive, so it
belongs to benchmarking rather than to the live path — but it is what turns
*never make things worse* from a principle into a measurement.

This whole direction expands the layer's input surface from "context state"
to "context state plus model output", which is a real architectural change,
not a new display. It is recorded here as a considered direction, not a
commitment.

### A turn is shown by its span's level

How the model receives a turn is named after the detail levels — verbatim, kept,
compacted, archived, out of view — and decided by the span the turn is in, never by
the turn's own facts. *Rejected: one value for archived and out of view, told apart
by whether the turn's own facts are sent.* A reply that yielded no fact of its own
then read as "the model sees nothing of this message" while the model read its
span's archived line: the screen overstated the loss, and a warning that is usually
wrong teaches the reader to skip the one that is right. *Rejected: a "contradicted"
level for a turn whose facts were all replaced.* Its span is still shown at its
level, with a note naming the replacement, so the turn is no less visible than
before; the replacement is a property of the fact, and the fact carries it.

### The facts panel leads with what needs a look

Facts are listed under what they are — short *topic: value* notes — rather than as
"what the model remembers": the model reads far more than its facts, and a list of
notes under that title passes a small annotation off as the whole memory. The
panel opens with the facts the model cannot see and with each replacement, shown
as the old value and the new one, so a false replacement is plain at a glance.
Every fact sits folded below them; most are on the page already, on their
exchange's line. *Rejected: every fact first, grouped by how it is sent.* The
groups that needed a look sat under forty notes that did not. The list is headed
with how far it reaches — on record up to the newest collected message — because
facts are read only at a collection: a change of plan in a recent message is in
its words, which the model receives as written, and is filed once that message is
collected. Headed "current", a value the conversation had already changed read as
still in force.

### A banner only for what needs a look

The top of the conversation view carries a banner only for something the reader
should act on or know about: messages the model can no longer see, and a collection
that fell short — plus a passing line while a collection runs. Everything else is said
in the panel it is about: that nothing is out of view, in the message counts; that a
collection runs after the next reply, beside them; replaced facts, at the top of the
facts panel. A warning can be dismissed, and returns only when there is more of what
it warns about: the reader has seen this loss, not every future one. *Rejected:
standing banners for good news and routine state.* Three of them sat above the
page on every visit, and a warning among them was one more line to skim past.

### The next collection is previewed, not remembered

The preview runs a collection's own first steps — find the cut point, draft the
spans, score and plan every span, the ones already collected included — on the
session as it stands, with no model call
and nothing announced on the bus; a preview reported as a collection would have
every subscriber record work that never happened. *Rejected: show the last real
pass's plan.* It describes the conversation before that pass's own cut, so
straight after every collection it would show the batch just taken. *Accepted
cost:* the preview drafts and scores on every change to the session, on the event
loop; cheap with word-overlap relevance, and to move to a worker thread if an
embedding implementation is configured.

### Intent and outcome are different signals

A collection can complete and still fall short of its own plan. The plan wants a
span summarised, the compactor refuses the summary it gets back, and the span is
kept word for word instead. Nothing is lost and the context is correct — it is
just larger than the collection set out to make it.

A dashboard that drew only the plan would draw transformations that never
occurred, which is the same class of failure as the aggregated risk score this
layer exists without: a confident-looking number with nothing behind it. The
difference is that here the subject is the system's own behaviour, so getting it
wrong means the transparency layer is the thing lying.

So a completed pass carries a structured record of what it intended and did not
manage, with the reason separated by what an operator would do next — a refused
summary means the run or the cap was wrong; a failed provider call means check the
provider; a failed extraction means check the extractor; a front that cannot
fit means the budget is too small for the conversation. Each one is
announced on the bus as it happens as well as collected onto the result, so a
subscriber does not have to wait for the pass to end, and degradation is never
something a consumer has to infer from counts that fail to add up.

Failure is not the only gap between intent and outcome. The collection that takes
new turns also re-plans every older span, lowering some and raising others (see
`generational-memory/the-cut.md`). So a committed collection carries its
**outcome**: which of its new turns went to which level, and which turns condensed
earlier it showed with less or more detail in the same pass, each named by turn. The
dashboard draws what a collection did from the outcome, and what it could not do
from the degradations.

Two decisions inside this that are easy to get wrong:

- **The run status stays "completed".** A separate status value for a degraded
  pass was rejected: status answers *did the pass run*, and degradation answers
  *did everything it intended happen*. Fusing them means no consumer can ask the
  first without handling the second, and every equality check already written
  against the status silently stops matching — which is precisely what a new enum
  value is worst at. What would change this: a degraded pass needing different
  *handling* at the HTTP boundary rather than different reporting.
- **An arbitrary exception's message is not recorded.** These records reach the
  dashboard over HTTP, and payloads carry no raw conversation content. The
  project's own exception messages are content-free by convention, so theirs are
  kept; anything else — including an exception from a compactor written by someone
  else — contributes only its type name. A failure cause is not a trust boundary
  we control.

---

## Consequences

- **The signals are correlative with hallucination risk, not causal.** High
  token-budget pressure does not *guarantee* a bad answer; a quiet
  generation lifecycle does not *guarantee* a good one. The signals
  are observations about context state, not predictions about answer quality.
  That distinction must be stated prominently anywhere the signals are
  displayed — otherwise they quietly become the same authoritative black box
  the aggregated score was rejected for.
- **A clean pass must stay clean.** The degradation record is only useful while
  it is rare: a field that is populated on most passes becomes noise consumers
  learn to skip, and then it is worse than absent, because its presence implies
  someone is reading it. Anything that routinely degrades is a design fault to
  fix rather than a condition to report.

- **The monitor's view drifts if the engine grows without it.** Every new
  event type that affects context state has to be considered for inclusion in
  the monitor, or its view silently falls out of sync with reality. This is a
  recurring tax — the price of the continuous design — and it is paid every
  time the engine gains a new state-changing behaviour.

- **Until a grounded predictor exists, the hallucination-transparency claim
  rests on the developer's own vigilance** acting on observable signals, not
  on a system-produced number. This is a deliberate choice, and a defensible
  product: the signals do the part that can be done honestly, and the human
  does the part that cannot yet be automated honestly.
