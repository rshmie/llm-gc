# LLM-GC — Project Overview

In a long conversation with an LLM, every request resends the whole history. By turn
50, the decision made at turn 5 ("we'll use Postgres") is one short message under tens
of thousands of tokens of old tool output and settled debate. The model may no longer
use it reliably, nothing tells you so, and you pay to resend all of it on every call.

LLM-GC sits between an application and the model provider and manages that history
the way a garbage collector manages memory. Recent turns are sent exactly as written.
Older exchanges are condensed — quoted whole, summarised, or archived to a one-line
record with their facts — by how much each matters to what is being discussed now.
Nothing is deleted: a condensed exchange is shown in more detail again when the
conversation returns to it, and every decision is filed with whatever later replaced
it. **And none of it is silent:** the developer can see, for every turn, what the
model still receives.

Fewer tokens are a byproduct. The design optimizes for what the model keeps, not for
how much is removed.

This document is the conceptual entry point: the mental model the rest of the docs
assume. For installation, usage and measured results, see the project
[README](../README.md).

---

## The problem this project solves

In a long conversation the context window fills with material that no longer serves
the current question: old tool output, resolved debugging, file contents read hours
ago, decisions made and moved past. The model cannot clean its own context; every
call resends all of it. Three things follow.

1. **Earlier decisions become hard for the model to use.** Asked at turn 50 which
   database was chosen at turn 5, the model has the answer in its context, but as
   one short message among tens of thousands of tokens of less relevant material.
   Liu et al. (2023, *"Lost in the Middle: How Language Models Use Long Contexts"*)
   showed that information in the middle of a long context is recalled less
   reliably than information at its ends. Later models may narrow that gap; today,
   a decision can be in the window and still not be used with confidence.
2. **Nobody can see it happening.** The model answers in the same confident tone
   whether its context is clean or mostly noise. Nothing says "the ground under this
   answer has gone"; the first signal is usually a wrong answer.
3. **Every request pays for all of it.** The stale material is resent, and billed,
   on every call.

Existing approaches each address part of this. Larger context windows push the
problem further out. A provider's own compaction makes what is sent smaller. Model
routing makes each request cheaper. All are real and useful; none puts the cleanup
policy in the developer's hands, and none shows the developer what has happened to
the context.

LLM-GC is built for that missing part: a layer of context management between the
application and the provider that the developer can control and inspect.

---

## Why this project exists

The motivation is **epistemic transparency**. A model has no reliable sense of what
it does and does not know, so it cannot be relied on to say when its grounding has
gone weak. A person can make that judgement — whether to trust an answer, restate a
decision, or start afresh — but only with the data to make it from.

So visibility is not a feature added to a context manager; it is the reason the
project exists. The condensing serves it as much as it saves tokens: a context that
is managed deliberately is one that can be described honestly. The savings come for
free.

That gives the developer three commitments:

- **Visibility.** What the model is sent, what has been condensed and how far, what
  was extracted, and what the model no longer sees can all be inspected, turn by
  turn.
- **Control.** What counts as relevant, what counts as a decision, how much detail
  each part keeps, and when to collect are settings the developer owns, not the
  provider.
- **Retention.** Decisions made earlier stay where the model can act on them, rather
  than fading under later material, and a decision that was later replaced is marked
  as replaced.

---

## What LLM-GC does to a conversation

The same conversation at turn 50, without and with LLM-GC:

```
 WITHOUT: the whole history, on every request
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ t0 t1 … t5 "we'll use Postgres" t6 … t28 [30k tokens of logs] … t49 t50  │
 └──────────────────────────────────────────────────────────────────────────┘

 WITH: one recap of the older exchanges, then the recent turns as written
 ┌ RECAP ───────────────────────────────────────────────────────────────────┐
 │ [Archived] (turns 0–1) Set up the repo for the ingest service. · …        │
 │ [Quoted exactly] (turn 5, user) We'll use Postgres for the event store.   │ ← asked
 │ [Quoted exactly] (turn 6, assistant) …every word of the reply…            │   about now
 │ [Summarised] (turns 28–31) Why is the nightly job failing? · …            │ ← gone
 │ A few lines on the log investigation and its fix.                         │   quiet
 │ …                                                                         │
 └───────────────────────────────────────────────────────────────────────────┘
   t45 t46 t47 t48 t49 t50   ← exactly as the client sent them
```

Each entry in the recap is one exchange — a **span** — shown at the detail level its
score earns. Recent, or under discussion now: quoted exactly. Gone quiet: summarised,
or archived to its opening line and facts. Levels are chosen again at every
collection, so if the user returns to the log investigation, it is shown in full
again, from its own words. The dashboard shows the developer the same picture, turn
by turn.

---

## Why cleanup demands transparency

The project borrows its structure from garbage collection, and the borrowing
has one limit that shapes everything else.

A runtime garbage collector is **sound**. Reachability is decidable and
exact: trace from the root set, and an object is either reachable or it is
not. The collector never frees a live object. That guarantee is precisely
why a runtime collector can run silently — it has nothing to report, because
it cannot be wrong.

Relevance is not decidable. The root set for a conversation is the *next
question*, and it has not been asked yet. A turn that looks irrelevant at
turn 10 can be the most important thing in the conversation at turn 12, and
no amount of analysis at turn 10 can determine that. Every relevance decision
is a bet placed against a future that does not exist yet.

Three consequences follow, and they run through the whole design:

**Cleanup cannot be silent.** A system that makes unsound, unverifiable
decisions about what the model is allowed to see — and hides those decisions
from the person who will be handed the resulting answer — has replaced one
opaque actor with two. Transparency here is not a feature chosen for its
appeal. It is the only honest way to ship an operation that cannot be proven
correct.

**Condensing has to be recoverable, not final.** Because a bet can be wrong,
nothing is deleted. A turn leaving the young generation lands in a span that
keeps it whole, shown with less detail while the conversation is elsewhere and
with more when it returns, and the decisions inside it survive as extracted
facts. A wrong bet then costs fidelity for a while rather than knowledge.

**Conservatism is calibration, not timidity.** Showing a turn in more detail
than needed costs a bounded, known number of tokens. Condensing away a detail
that *was* needed costs an unpredictable amount, and stays invisible until it
surfaces as a wrong answer. Asymmetric costs under an undecidable rule demand an
asymmetric policy.

Principles 1 to 3 below are these consequences stated as commitments.
Principles 4 and 5 are what they demand of the surface the developer sees.

---

## Core principles

These are the invariants the project commits to. They guide every
architectural decision and are what we will not compromise on, even when
it would be convenient.

1. **Quality preservation is non-negotiable.** The system must not cause
   the model to produce worse answers than it would without the system.
   Better to show more context than necessary than to condense away
   something that mattered.

2. **Conservative by default.** The system condenses what has gone quiet and
   keeps detail for what is being discussed, and nothing it condenses is
   deleted. The asymmetry is intentional: a span shown in more detail than
   needed costs a few extra tokens; one condensed too far risks a wrong
   answer. The first cost is small and bounded; the second is unpredictable.

3. **Never make things worse.** If the system fails, times out, or
   encounters an unexpected condition, the application must still work as
   if the system were not present. The user experience is never degraded
   below the baseline of "no LLM-GC at all."

4. **The human exercises judgment; the system provides data.** This is
   the *epistemic transparency* commitment from the previous section,
   expressed as an operating rule. Given that AI models cannot reliably
   tell the user when their grounding is weak, LLM-GC's job is not to
   make that judgment for the user. It is to surface what is happening
   to the context — span scores, detail levels, what the model still sees
   and what it no longer does — so the human can decide when to trust the
   output.

5. **The system is honest about its own outputs.** Whatever LLM-GC reports
   — signals, summaries, detail levels — is labeled
   for what it is: a heuristic, an estimate, a snapshot. The system does
   not claim certainty it does not have. Replacing the model's opacity
   with a different opacity would defeat the purpose of principle 4.

---

## How the system is structured

The system has five major parts. This section gives the conceptual map; each
part's design doc under `design/` covers it in detail.

```
          HOT PATH: every request; no model call, never waits
 ┌────────┐  request  ┌──────────────────────────────────────┐  request  ┌──────────┐
 │ client │ ────────► │ PROXY                                │ ────────► │ provider │
 │        │ ◄──────── │ 1 find the conversation's session    │ ◄──────── │ (model)  │
 └────────┘   reply   │ 2 record the turns it has not seen   │   reply   └──────────┘
                      │ 3 send [recap] + recent turns        │
                      └───────┬──────────────────▲───────────┘
                              │ after the reply  │ reads the current snapshot
 ═════════════════════════════╪══════════════════╪═══════════════════════════════════
          COLD PATH: in the background, one pass per session at a time
                              ▼                  │
 ┌ ENGINE (GcService) ─────────────────────┐   ┌─┴─ SESSION (one per conversation) ─┐
 │ is a collection worth making? if so:    │   │ young      recent turns, verbatim  │
 │  draft spans                (grouping)  │   │ old        spans, each at a level  │
 │  score, plan levels  (scorer, planner)  │   │ permanent  facts, and what         │
 │  read facts, summarise   (model calls)  │   │            replaced them           │
 │  commit ────────────────────────────────┼──►│                                    │
 └───────────────┬─────────────────────────┘   └────────────────────────────────────┘
                 │ events, after the commit
                 ▼
   EVENT BUS ──► HEALTH MONITOR ──► DASHBOARD ◄── reads the session too, never writes
```

The double line is the rule the design is built around: **nothing above it waits for
anything below it.** A request reads whatever snapshot is current and is forwarded at
once; a collection, with its model calls, runs after the reply and commits a new
snapshot when it is done. The next request simply finds it there.

**The Engine** is where context-management decisions happen. When older
turns have aged out of the recent tail, the engine groups them into spans, one
per exchange, scores every span by how much it matters now, plans each span's
detail level, and builds the recap the model receives. The measure of relevance,
the importance scorers and the compactor are each pluggable — an embedding
relevance, a different summariser — and can be swapped without changing how the
engine is called.

**Generational Memory** is where context lives across the lifecycle of a
conversation. Recent turns sit in the *young generation* in full detail.
As they age past their useful window, they move to the *old generation* as
**spans** — one exchange each: something the user said, and the replies to it.
A span keeps its turns and is shown at a detail level re-decided at every
collection: quoted whole, summarised, or archived to an opening line and
its facts. The facts read from every span, and what later replaced them, form
the *permanent generation*, which is not discarded as the conversation grows.
This separation lets the system apply different rules to context in different
stages of its life.

**The Events Bus** is the cross-cutting channel every component publishes
to. When a collection makes a span, lowers one to less detail, or files a
fact that replaces an older one — each emits a structured event.
The events bus is what makes the system observable. The visibility layer below
consumes it today; a logger, the benchmarks and post-session analysis are planned
consumers of the same stream.

**The Visibility Layer** is where the human gets what they need to judge. It is
the *Health Monitor*, which follows each conversation continuously, not only when
a collection runs, and the *dashboard* (the Visualizer), which shows that state
live. Post-session reports are planned. Without this layer the system would manage
context invisibly, which would miss the point entirely.

The layer answers two different questions, for two different readers, and
keeping them apart is what keeps the project pointed at its purpose.

- **Transparency** asks: *what does the model currently know, at what
  fidelity, and what has it lost?* Its subject is the **conversation**. Its
  reader is the developer deciding whether to trust an answer and what to do
  next. This is the project's reason for existing.
- **Observability** asks: *what did LLM-GC do, how long did it take, did it
  fail?* Its subject is the **tool**. Its reader is whoever operates or
  debugs the tool. It is necessary, and it is not the point.

Both are served from the same data; the difference is which question a
surface is built to answer. A view reporting level counts and generation
sizes is describing the collector. A view reporting *"the decision you made at
turn 5 is no longer in verbatim context — it survives as its exchange's opening
line and an extracted fact"* is describing the conversation. The second is what
the developer acts on, so it leads. The first stays one level down, for
when something needs explaining. The dashboard is built the same way: one view
for the conversation, which opens first, and one for LLM-GC at work.

**The Proxy** is how the system is most often used in practice. It sits as
a small HTTP server between an application (such as Claude Code) and an
LLM provider. It intercepts each request, sends the provider the session's
current recap in place of the turns it condenses, and lets the engine collect
after the reply. The proxy is intentionally thin: it contains no GC logic of its
own, only the glue that connects the engine to a real conversation flow.

Session state has its own design doc. Persistent storage and embedding relevance
are planned, and their docs describe where they will plug in.

---

## Where to look next

Every doc under `design/` is a living description of one area, and each carries
a **Why this shape** section: the rationale, and the credible alternatives that
were rejected. Read those to understand *why* the system is built the way it is,
not just what it does.

**The engine**

- **[design/scoring/overview.md](design/scoring/overview.md)** — how much a span
  matters now: whether it is live, and the decisions and specifics it states.
  Each importance scorer has its own doc alongside it.
- **[design/generational-memory/detail-levels.md](design/generational-memory/detail-levels.md)**
  — the four levels a span is shown at, and how the planner picks them under one
  budget.
- **[design/generational-memory/compaction.md](design/generational-memory/compaction.md)**
  — how a span gets its summary, and when a summary is refused.

**Memory and state**

- **[design/generational-memory/overview.md](design/generational-memory/overview.md)**
  — the young / old / permanent generations, spans, and supersession.
- **[design/generational-memory/the-cut.md](design/generational-memory/the-cut.md)**
  — a collection: when one happens, and why it is the one moment the conversation
  the model receives is changed.
- **[design/session/overview.md](design/session/overview.md)** — how one
  conversation's state persists across turns, and the concurrency contract that
  protects it.
- **[design/storage.md](design/storage.md)** — how persisted records survive an
  upgrade of the system that wrote them.

**Meeting real traffic**

- **[design/proxy/overview.md](design/proxy/overview.md)** — how the proxy finds
  the conversation a request belongs to, keeps its record true to the client's
  history, and never makes a conversation worse than not using LLM-GC.
- **[design/llm-provider.md](design/llm-provider.md)** — the model calls LLM-GC
  makes for itself, and why they never share the user's key.
- **[design/configuration.md](design/configuration.md)** — where settings come from,
  which one wins, and why a bad one stops startup instead of degrading a request.

**Transparency and failure**

- **[design/events/overview.md](design/events/overview.md)** — the event bus
  every component reports through.
- **[design/visibility/overview.md](design/visibility/overview.md)** — what the
  model knows of each conversation, what LLM-GC did to it, the dashboard that
  shows both, and why none of it is collapsed into a single score.
- **[design/failure-handling.md](design/failure-handling.md)** — what happens
  when the engine itself fails, and how that stays visible.

For installation, usage examples, and measured results, see the project
[README](../README.md).
