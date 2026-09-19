# LLM-GC — Project Overview

LLM-GC applies garbage collection ideas — mark, sweep, compact, generational
memory — to the context window of LLM conversations. The system identifies
what is no longer useful, removes or compresses it, and extracts decisions
worth preserving into a long-lived store. Reducing the number of tokens sent
to the model is a byproduct of removing waste; the design optimizes for what
gets kept, not for how much gets removed.

The project is built around three commitments to the developer:

- **Visibility.** What is in the context, what is fresh, what has been
  compressed, what was extracted, and what was discarded is observable —
  not hidden behind a black box.
- **Control.** The cleanup policy — what counts as relevant, what counts as
  a decision worth preserving, when to compact, when to archive — is
  configurable and inspectable, not fixed by the provider.
- **Retention.** Decisions and references made earlier in the session are
  explicitly preserved rather than left to fade under later material, so
  the model can act on them later in the session.

### Why this project exists

Motivation behind LLM-GC is **epistemic transparency**: AI models do
not have a reliable internal sense of what they know and don't know, so
humans cannot lean on the model to flag when its grounding is weak. The
human, however, can exercise that judgment — *if they have the data*. The
visibility commitment above is not a feature; it is the project's reason for
being. The retention and control work exists in part to make that visibility
useful (there has to be something coherent to be transparent *about*).
Token efficiency is the part that comes for free.

This document is the conceptual entry point to the LLM-GC documentation. It
gives you the mental model that the rest of the docs assume. For the
user-facing pitch, install steps, and measured results, see the project
README.

---

## The problem this project solves

In a long LLM conversation, the model's context window accumulates material
that is no longer useful for the current question — old tool outputs,
resolved debugging exchanges, file contents from earlier in the session,
decisions that were made and moved past. The model has no mechanism to clean
its own context: every API call resends the full conversation history.

Two consequences follow:

1. **Earlier decisions become hard for the model to retrieve.** When a user
   asks at turn 50 what database was chosen at turn 5, the relevant turn is
   still technically present in the context — but it is one short message
   inside tens of thousands of tokens of less-relevant material. Liu et al.
   (2023, *"Lost in the Middle: How Language Models Use Long Contexts"*)
   showed that, at the time of writing, information placed in the middle of
   a long context is recalled less reliably than information at the ends.
   Future models may close this gap; today, the decision can be present in
   the window but not in a form the model can use confidently.

2. **The developer has no visibility into degradation.** The model answers
   with the same tone whether its context is fresh or full of noise. There
   is no signal that says "the foundation under this answer is unreliable."
   A wrong answer is usually the first signal.

Existing approaches address parts of this. Larger context windows raise the
threshold at which problems start. Provider-side compaction reduces the size
of what is sent. Model-routing systems reduce per-request cost. Each of these
is real and useful. None of them put the cleanup policy in the developer's
hands, and none of them surface what is happening to the context in a form
the developer can inspect.

LLM-GC is built for the part those approaches don't address: giving the
developer a controllable, inspectable layer of context management that sits
between their application and the LLM provider.

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

**Removal has to be recoverable, not final.** Because a bet can be wrong,
what is swept must stay reachable in *some* form. This is why the system
compresses rather than deletes: a turn leaving the young generation survives
as a summary, and the decisions inside it survive as extracted facts. A
wrong bet then costs fidelity rather than knowledge.

**Conservatism is calibration, not timidity.** Keeping a turn that was not
needed costs a bounded, known number of tokens. Removing a turn that *was*
needed costs an unpredictable amount, and stays invisible until it surfaces
as a wrong answer. Asymmetric costs under an undecidable rule demand an
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
   Better to keep more context than necessary than to remove something
   that mattered.

2. **Conservative by default.** When uncertain whether a piece of context
   is still useful, the system keeps it. The asymmetry is intentional: a
   message kept unnecessarily costs a few extra tokens; a message removed
   incorrectly risks a wrong answer. The first cost is small and bounded;
   the second is unpredictable.

3. **Never make things worse.** If the system fails, times out, or
   encounters an unexpected condition, the application must still work as
   if the system were not present. The user experience is never degraded
   below the baseline of "no LLM-GC at all."

4. **The human exercises judgment; the system provides data.** This is
   the *epistemic transparency* commitment from the previous section,
   expressed as an operating rule. Given that AI models cannot reliably
   tell the user when their grounding is weak, LLM-GC's job is not to
   make that judgment for the user. It is to surface what is happening
   to the context — relevance scores, generation state, what was kept,
   what was removed — so the human can decide when to trust the output.

5. **The system is honest about its own outputs.** Whatever LLM-GC reports
   — risk scores, health metrics, summaries, classifications — is labeled
   for what it is: a heuristic, an estimate, a snapshot. The system does
   not claim certainty it does not have. Replacing the model's opacity
   with a different opacity would defeat the purpose of principle 4.

---

## How the system is structured

The system has five major parts. This section gives the conceptual map; each
part's design doc under `design/` covers it in detail.

**The Engine** is where context-management decisions happen. When the system
is asked to clean up a conversation, the engine scores each turn for
relevance, classifies turns into keep / compress / archive buckets, and
assembles the resulting context. Scoring, classification, and assembly are
each pluggable — different scoring strategies, different classifiers,
different compaction approaches can be swapped without changing how the
engine is called.

**Generational Memory** is where context lives across the lifecycle of a
conversation. Recent turns sit in the *young generation* in full detail.
As they age past their useful window, they move to the *old generation*
where they are compressed into summaries. Decisions, references, and
extracted facts move into the *permanent generation*, which is not
discarded as the conversation grows. This separation lets the system apply
different rules to context in different stages of its life.

**The Events Bus** is the cross-cutting channel every component publishes
to. When the engine scores a turn, when memory promotes a turn between
generations, when a decision is extracted — each emits a structured event.
The events bus is what makes the system observable; the visibility layer
below is one of its consumers, but logging, benchmarking, and post-session
analysis are also consumers of the same stream.

**The Visibility Layer** is where the human gets what they need to judge. It
consists of the *Health Monitor*, which observes the conversation
continuously — not only when GC operations run — and the *Visualizer*, which
renders that state in real time and generates post-session reports. Without
this layer the system would manage context invisibly, which would miss the
point entirely.

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
surface is built to answer. A view reporting classification counts and
generation sizes is describing the collector. A view reporting *"the
decision you made at turn 5 is no longer in verbatim context — it survives
as an extracted fact"* is describing the conversation. The second is what
the developer acts on, so it leads. The first stays one level down, for
when something needs explaining.

**The Proxy** is how the system is most often used in practice. It sits as
a small HTTP server between an application (such as Claude Code) and an
LLM provider, intercepts each request, runs the engine on the conversation
context, and forwards the cleaned version to the provider. The proxy is
intentionally thin: it contains no GC logic of its own, only the glue that
connects the engine to a real conversation flow.

Storage backends, session management, and embedding models are real parts
of the system but live below this map; see their design docs under `design/`.

---

## Where to look next

Every doc under `design/` is a living description of one area, and each carries
a **Why this shape** section: the rationale, and the credible alternatives that
were rejected. Read those to understand *why* the system is built the way it is,
not just what it does.

**The engine**

- **[design/scoring/overview.md](design/scoring/overview.md)** — how a turn's
  relevance is measured across five dimensions, and how the scores combine.
  Each scorer has its own doc alongside it.
- **[design/sweeper-and-composition/sweeping-mechanism.md](design/sweeper-and-composition/sweeping-mechanism.md)**
  — how scores become KEEP / COMPACT / ARCHIVE classifications.
- **[design/sweeper-and-composition/composition-mechanism.md](design/sweeper-and-composition/composition-mechanism.md)**
  — how those classifications are assembled back into a message list.
- **[design/sweeper-and-composition/compaction-mechanism.md](design/sweeper-and-composition/compaction-mechanism.md)**
  — how compacted runs are summarised, and why the default does nothing.

**Memory and state**

- **[design/generational-memory/overview.md](design/generational-memory/overview.md)**
  — the young / old / permanent generations and how context moves between them.
- **[design/session/overview.md](design/session/overview.md)** — how one
  conversation's state persists across turns, and the concurrency contract that
  protects it.
- **[design/storage.md](design/storage.md)** — how persisted records survive an
  upgrade of the system that wrote them.

**Transparency and failure**

- **[design/events/overview.md](design/events/overview.md)** — the event bus
  every component reports through.
- **[design/visibility/overview.md](design/visibility/overview.md)** — the
  signals the system exposes about its own confidence, and why they are not
  collapsed into a single score.
- **[design/failure-handling.md](design/failure-handling.md)** — what happens
  when the engine itself fails, and how that stays visible.

For installation, usage examples, and measured results, see the project
[README](../README.md).
