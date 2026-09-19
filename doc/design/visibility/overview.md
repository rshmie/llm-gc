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
surface reporting classification counts, generation sizes and pass durations
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

- **The Health Monitor — the data source.** It listens to the event bus
  continuously and keeps a live answer to one question: *"what state is the
  context in right now?"* It renders nothing; it maintains a snapshot of
  named signals.
- **The Visualizer (dashboard) — the view.** It renders what the monitor
  knows. It reads the monitor's snapshot and the event stream; it computes
  nothing of its own and emits no events.

This document covers the layer as a whole and the Health Monitor in depth.

---

## Why it exists

The motivating asymmetry: an AI model has no reliable internal sense of what
it knows and does not know, so a human cannot lean on the model to flag when
its grounding is weak. The human *can* exercise that judgment — but only with
data. Principle 4 of the project (*the human exercises judgment; the system
provides data*) is the operating form of this. The visibility layer is where
that principle becomes real.

What would break without it: context management would happen **invisibly**.
The engine would score, sweep, compact, and archive — quietly changing the
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

The Health Monitor subscribes to exactly three events: garbage collection
finished, message archived, and knowledge entry superseded. It is the same
publish/subscribe channel every other consumer (logger, benchmarks, the
dashboard) reads from; the monitor is just one more subscriber, and the bus
neither knows nor cares that it is listening.

Its data sourcing is deliberately hybrid — *aggregator, not computer*:

- **Current-state signals** (token budget, GC breakdown, generation-tier
  contents) are **read** from the most recent GC result and from generational
  memory's real storage, never re-derived from accumulated events. The
  components that computed those numbers are the authority on them.
- **The transitions timeline** is the one genuinely event-accumulated piece,
  because history has no other source: current state cannot show that a
  supersession ever happened — superseded entries are excluded from every
  active view by design — so an append-only log is the only place that
  moment stays visible.
- **Promotions are recorded from the producer's own event**, not inferred.
  Generational memory announces which turns it moved into the old generation,
  and the monitor writes one transition per turn.

  *Rejected: inferring promotion by diffing classifications.* The monitor
  originally detected promotion itself, by comparing each pass's per-turn
  classifications against the last pass and treating a KEEP→COMPACT change as
  a promotion. It required no producer changes, which is why it was chosen —
  but it confused a **decision** with a **move**, and failed in both
  directions once turns genuinely started moving. It **missed** real
  promotions, because a turn appended and classified COMPACT in the same pass
  has no earlier KEEP to diff against. It **invented** promotions on the
  stateless collect path, which classifies COMPACT but never promotes, so the
  timeline claimed transitions into an old generation that was in fact empty.
  The rule this leaves behind: *only the component that performs a state
  change is the authority that it happened.*
- **Old-generation contents are read from storage**, for the same reason.
  Deriving them from the last pass's COMPACT entries reported only the most
  recent pass's classifications — a session that had promoted four turns over
  four passes still showed one — and reported a populated old generation on
  the stateless path, where nothing had ever been promoted. Young-generation
  contents are still derived from the last sweep's KEEP entries, which is
  exactly the set of turns that stayed.

The snapshot it exposes is **built whole on every read** — never cached,
never mutated field-by-field — so a reader can never observe a torn,
half-updated state.

It is **passive**. It observes and reports; it never triggers a collection
pass itself. Deciding *when* to clean context is a decision with consequences
for what the model produces next, and that decision stays with the developer
or the application. The dashboard's manual "Run GC now" control does not
contradict this: that is the *developer* pulling the lever explicitly — the
human judgment the passivity rule exists to protect.

### What it exposes — named signals, no aggregate

The monitor's public output is a **snapshot**: an immutable read-out of its
current state at a point in time (the `ContextHealth` model). The snapshot
carries a set of named, individually-explainable signals, grouped by which
of the two questions they answer.

**Transparency signals — about the conversation.** These lead, because they
are what the developer acts on.

| Signal | What it measures | The question it answers |
|---|---|---|
| **Knowledge state** *(not yet built)* | every fact and decision the session has produced, with its current fidelity — verbatim, summarized, fact-only, or superseded — and its origin turn | "What does the model still know, and how well?" |
| **Generation lifecycle state and recent transitions** | current young / old / permanent contents (counts and token shares) plus the last *N* promote / archive / supersede transitions | "Where does my decision live now — and did anything supersede it?" |
| **Token budget pressure** | current context tokens as a fraction of the configured budget | "How crowded is the context the model is reasoning over?" |

**Observability signals — about LLM-GC.** These sit one level down.

| Signal | What it measures | The question it answers |
|---|---|---|
| **Latest GC breakdown** | KEEP / COMPACT / ARCHIVE counts and token totals from the most recent collection, plus the run's status, timing, and failure detail | "How much did the last cleanup change — and did it succeed at all?" A bypassed or failed pass is visible, never hidden behind a normal-looking breakdown; signals honestly report `None` ("not measured yet") rather than a fabricated zero before any pass has run |

**Knowledge state is the signal that most directly serves the project's
purpose, and it is the one that does not exist yet.** The three built signals
all describe *collector activity*: counts, sizes, durations, transitions. A
developer reading "old generation holds 21 summaries, pressure 0.64" cannot
get from there to "the model no longer has the database decision verbatim" —
yet that inference is the entire product, and today the human has to make it
unaided from engine telemetry.

Almost all of the underlying data already exists: permanent-generation
entries carry their origin turn, old-generation summaries record the turns
they cover, and the transitions log records every move. What is missing is
the **projection** — pivoting that data from *"what did the collector do"* to
*"what does the model know"*, keyed by fact rather than by pass.

The projection belongs to the monitor, not to the view. The view computes
nothing of its own (see *Two packages*), and deciding that a fact is
"fact-only" rather than "verbatim" is a derivation over three storage tiers,
not a rendering choice. Putting it in the visualizer would duplicate that
logic in every future consumer — the post-session report and the benchmark
harness both need the same answer.

The generation-lifecycle signal is the richest of the built three: its
temporal dimension — *where* a fact lives and *how long* it has lived there —
is information the developer's epistemic vigilance acts on directly. The
temporal view is first-class, not metadata. Knowledge state is, in effect,
that signal re-indexed by the thing the developer actually cares about.

The three tracked transitions are not the same kind of move, and the set is
deliberately exactly these three. **Promote** (young → old) and **archive**
(old → permanent) are generational flow — a turn moving between tiers;
archive is extract-first, so the turn's knowledge is preserved before the
verbatim turn is removed, making it lossless by construction. **Supersede**
is contradiction resolution *within* the permanent tier — a newer fact
overriding an older contradicting one. There is deliberately no *demote*
(old → young): memory flows one direction only. Resurrecting a turn because
its relevance rose again is an unbounded action the system chooses not to
take — and it is unnecessary, because archive has already preserved that
turn's knowledge, so nothing that matters is lost by keeping the flow
one-way.

The **transformation ratio** — the proportion of context that has been
compacted or archived relative to the total — is computed and kept as an
**internal** snapshot field. It is accessible for benchmarking and serves as
the baseline any future aggregate must beat. It is *not* surfaced as a
headline number (see *Why this shape*).

**Distinct from the relevance score.** These are conversation-level signals,
read by the developer. They are not the per-turn *relevance score* the scoring
layer produces — that score is computed per message and feeds the sweeper's
KEEP / COMPACT / ARCHIVE classification. Different scope (conversation vs.
turn), different consumer (developer vs. sweeper), different purpose; the two
share none of their internals.

### Two packages

The layer's two pieces live in two packages, because they have two jobs:

- **`monitoring/`** — the Health Monitor (`ContextHealthMonitor`) and its
  snapshot (`ContextHealth`). Pure Python; **no web dependencies**.
- **`visualizer/`** — the dashboard view, and later the post-session report.
  Web dependencies live here.

The dependency runs one way: the view reads the monitor. Keeping them
separate keeps the engine's in-process import graph — which is what wires the
monitor into the bus — free of any web-framework code. The doc area
("visibility") names the *layer*; the two packages are its
implementation.

The current dashboard reads the snapshot over a periodic HTTP poll and
renders each signal as a standalone display (no aggregate); server-push
(WebSocket) delivery is a planned upgrade of the transport, not of the
split — the monitor stays the single data source either way.

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

---

## Consequences

- **The signals are correlative with hallucination risk, not causal.** High
  token-budget pressure does not *guarantee* a bad answer; a quiet
  generation lifecycle does not *guarantee* a good one. The signals
  are observations about context state, not predictions about answer quality.
  That distinction must be stated prominently anywhere the signals are
  displayed — otherwise they quietly become the same authoritative black box
  the aggregated score was rejected for.

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
