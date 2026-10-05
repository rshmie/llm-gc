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

- **The data sources.** The Health Monitor listens to the event bus
  continuously and keeps, per session, a live answer to *"what has LLM-GC
  done to this conversation?"* — its last pass, its history of passes,
  transitions and degradations. The knowledge-state projection answers *"what
  does the model know?"* from a read-only inspection of the session. Neither
  renders anything.
- **The Visualizer (dashboard) — the view.** It renders what those report,
  beside what the client said (a transcript the proxy hands it). It computes
  nothing of its own, and its one action is the developer's Run GC now.

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

The Health Monitor subscribes to five events: garbage collection finished,
turns promoted into the old generation, message archived, knowledge entry
superseded, and aging degraded. It is the same publish/subscribe channel every
other consumer (logger, benchmarks, the dashboard) reads from; the monitor is
just one more subscriber, and the bus neither knows nor cares that it is
listening.

**One monitor serves every session, and keeps them apart.** Every event names
the session it describes, and the monitor keeps a separate record per session:
its last pass, its transitions, and its recent degradations. Events with no
session — the stateless library path — share one record. The number of records
is bounded, dropping the session heard from least recently, so a monitor never
grows with the number of conversations a proxy has ever served.

**It keeps no conversation content.** A finished pass's signals are built the
moment the event arrives and only those are kept — counts, tokens, turn
indices — never the result itself, which carries the whole context. Holding it
would keep every conversation's text alive in the monitor after its session
closed.

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
| **Knowledge state** | every recorded turn with its current fidelity — verbatim, quoted, summarised, facts only, or superseded — and every fact with whether the model is sent it, what replaced it, and the turns it came from | "What does the model still know, and how well?" |
| **Generation lifecycle state and recent transitions** | current young / old / permanent contents (counts and token shares) plus the last *N* promote / archive / supersede transitions | "Where does my decision live now — and did anything supersede it?" |
| **Token budget pressure** | current context tokens as a fraction of the configured budget | "How crowded is the context the model is reasoning over?" |

**Observability signals — about LLM-GC.** These sit one level down.

| Signal | What it measures | The question it answers |
|---|---|---|
| **Latest GC breakdown** | KEEP / COMPACT / ARCHIVE counts and token totals from the most recent collection, plus the run's status, timing, and failure detail | "How much did the last cleanup change — and did it succeed at all?" A bypassed or failed pass is visible, never hidden behind a normal-looking breakdown; signals honestly report `None` ("not measured yet") rather than a fabricated zero before any pass has run |

**Knowledge state is the signal that most directly serves the project's
purpose.** The other signals describe *collector activity*: counts, sizes,
durations, transitions. A developer reading "old generation holds 21
summaries, pressure 0.64" cannot get from there to "the model no longer has the
database decision verbatim" — and that inference is the entire product.

Knowledge state is that inference made by the system. It pivots one session's
data from *"what did the collector do"* to *"what does the model know"*, keyed
by turn and by fact:

```
 turn  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
       ░░ ░░ ▓▓ ░░ ▓▓ ╳╳ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▓▓ ▒▒ ▒▒ ▒▒ ▒▒ ██ ██ ██ ██ ██
       └────── facts only ──────┘          └ quoted / ┘ └── verbatim ──┘
       ░░ facts stored, not sent           summarised     ▔▔ ▔▔ ▔▔ taken by
       ╳╳ superseded                                           the next cut
```

- **Per turn**, where it is now. A turn in the tail is verbatim; one covered by
  a front entry is quoted or summarised; anything else has been reduced to facts
  — and if every one of its facts was later contradicted, it is superseded. The
  last case needs no record of its own: every departure from the young
  generation leaves a front entry or at least one fact, and an archived front
  entry passes the turns it covered on to its facts, so a turn found in neither
  place can only have been reduced to facts.
- **Per fact**, whether it is **sent**. Facts are chosen for the front by
  importance and then cut to a budget, so a fact can be stored and still never
  reach the model. *Stored, not sent* is the signal that matters most: the
  conversation has not forgotten it, and the model cannot see it.
- **The next cut**: which verbatim turns it would take, what each would become,
  and the two conditions it must meet, each with its state — a batch worth
  cutting, against `min_cut_tokens`, and a prompt that comes out smaller. Only the
  first can be known in advance: whether the prompt shrinks depends on summaries
  not yet written, so the dashboard says it is checked as the cut runs, and shows
  the last attempt's figures when one was refused. *Rejected: a progress meter
  alone.* It reads as a promise, and on a conversation of short messages the
  meter filled, Run GC now was pressed, and nothing happened, with nothing on the
  panel saying why.

It is built from a single read-only **inspection** of the session that the
engine produces with the same rendering that builds the prompt and the same
marking step a real aging pass runs, so a dashboard showing it cannot disagree
with what the model is sent or with what the next pass will decide.

The projection belongs to the monitoring package, not to the view. The view
computes nothing of its own (see *Two packages*), and deciding that a turn is
"facts only" rather than "quoted" is a derivation over three storage tiers,
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
| **The conversation** | What does the model know, at what fidelity, and what has it lost? | Every turn as it was said, with what the model still has of it — word for word, quoted, inside a summary, as facts, out of view, or contradicted — and the facts the model is and is not sent |
| **LLM-GC at work** | What did LLM-GC do, and did it work? | How the collector scores the verbatim turns against its thresholds, every pass and its outcome, degradations, transitions, the proxy's counters, and the settings in effect |

It polls once a second, and fetches a session's content again only when it has
changed. Server-push (WebSocket) delivery is a planned upgrade of the transport,
not of the split.

**One vocabulary, explained where it appears.** The interface uses the design's
words for what the collector decides, *keep*, *compact* and *archive*, and for
where a message lives, the *young*, *old* and *permanent* generations. Three words
are plainer on screen than in the code, and each one's explanation names the
code's word: a *collection* is a cut (a check that moves messages out of the young
generation), the *recap* is the front (the first message of every request), and a
*check* is a pass. Every term carries a short explanation where it appears, and the
guide maps each on-screen word to the code's.

```
code      cut          front    pass    keep   compact     archive    young generation
screen    collection   recap    check   kept   compacted   archived   word for word
```

*Rejected: a separate plain vocabulary*, which the dashboard first used (clean-up,
condensed block, quoted whole, kept as facts): read beside the code, two
vocabularies for one system confused more than either, and "clean-up" read as
something that triggers GC rather than what GC does after a reply. *Rejected: the
code's words alone*: "cut" reads as truncation and "front" means nothing on its
own. *Rejected, earlier: the design vocabulary with a glossary page*: a reader who
must leave the screen to decode it has been handed engine telemetry again. *Cost
accepted:* cut and front differ between screen and code; renaming them in the code
touches public configuration (`min_cut_tokens`), so it is a decision for before
1.0, not a cosmetic one. Token
figures compare like with like: the conversation as written counts only the
messages the session has recorded, because the newest message reaches it with
the next request, and comparing the whole transcript showed a saving equal to
that message which never happened.

**Run GC now explains its outcome, including when nothing changed.** When a
clean-up would not have shrunk the prompt, the engine reports what it would have
done — which messages, how many quoted, summarised or reduced to facts, and the
prompt size it would have produced — and the dashboard says why in those terms.
"It would not have shrunk the context" alone tells a reader nothing they can act
on.

**What was said comes from the client, not the engine.** The engine keeps only
what the model is sent; once a turn is summarised or reduced to facts its words
are gone, by design. To show what was lost beside what remains, the proxy hands
each request's conversation to a transcript recorder that exists only when the
dashboard runs: text only, bounded by session count, dropped with its session.
*Rejected: the engine keeps the originals* — a turn that was "removed" would
still be held for the life of the session, in every deployment, dashboard or
not. *Rejected: no originals* — the view could say turn 6 is facts only, but not
what was said in it, which is the comparison a developer needs.

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

### The next cut is previewed, not remembered

The preview runs the aging pass's own marking step — score, sweep, find the cut
point — on the session as it stands, with no model calls and nothing announced on
the bus; a preview reported as a sweep would have every subscriber record work
that never happened. *Rejected: show the last real pass's sweep.* It describes
the conversation before the pass's own cut, so straight after every cut it would
show the batch just taken; and surfacing a bypassed pass's sweep would put
compaction counts on a pass that compacted nothing. *Accepted cost:* the preview
scores on every change to the session, on the event loop; cheap with the default
scorers, and to move to a worker thread if an embedding scorer is configured.

### Intent and outcome are different signals

A GC pass can complete and still fall short of its own plan. The sweeper decides
a run should be summarised, the compactor refuses the summary it gets back, and
the turns stay verbatim. Nothing is lost and the context is correct — it is just
larger than the pass set out to make it.

The classification counts on a GC result describe the **sweeper's decision**, not
what happened. Read alone, a pass whose compactions were all refused reports
several compactions and files none. A dashboard rendering those counts would draw
transformations that never occurred, which is the same class of failure as the
aggregated risk score this layer exists without: a confident-looking number with
nothing behind it. The difference is that here the subject is the system's own
behaviour, so getting it wrong means the transparency layer is the thing lying.

So a completed pass carries a structured record of what it intended and did not
manage, with the reason separated by what an operator would do next — a refused
summary means the run or the cap was wrong; a failed provider call means check the
provider; a failed archive means check extraction and storage. Each one is
announced on the bus as it happens as well as collected onto the result, so a
subscriber does not have to wait for the pass to end, and degradation is never
something a consumer has to infer from counts that fail to add up.

Failure is not the only gap between intent and outcome. The pass that makes a cut
also sweeps the front, and can archive an entry the cut has just made (see
`generational-memory/the-cut.md`). Read as what happened, the decision counts then
described a clean-up as "1 summarised, 6 reduced to facts" beside a panel listing
all 12 messages it had reduced to facts — the transparency layer contradicting
itself. So a committed cut also carries its **outcome**: where its turns ended up,
and how many earlier-condensed turns went to facts in the same pass. The dashboard
draws what a clean-up did only from the outcome, and what it could not do only
from the degradations; the decision counts are for tuning the sweeper.

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
