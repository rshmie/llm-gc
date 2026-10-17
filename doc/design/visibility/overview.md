# Visibility Layer — Overview

A model answers in the same confident tone whether its context is fresh or full of noise. There is no built-in signal that says *"the foundation under this answer is unreliable"* — and the first signal the developer usually gets is a wrong answer. The visibility layer exists to be the **earlier** signal: to surface what state the context is in *before* the wrong answer, so the human can decide when to trust the output.

That is not a feature bolted onto the engine. It is the project's reason for being — *epistemic transparency*. The retention and control work exists in part to give this layer something coherent to be transparent about.

### Transparency, not observability

These are two different jobs, and conflating them is the fastest way for this layer to drift away from its purpose.

|  | Transparency | Observability |
|---|---|---|
| Asks | What does the model know, at what fidelity, and what has it lost? | What did LLM-GC do, how long did it take, did it fail? |
| Subject | The conversation | The tool |
| Reader | A developer judging whether to trust an answer | Whoever operates or debugs LLM-GC |
| Status here | The reason the project exists | Necessary, and not the point |

Observability is a well-understood discipline and LLM-GC needs it: a cleanup engine whose own behaviour is unexplainable is not production-grade. But a surface reporting level counts, generation sizes and pass durations is describing *the collector*. Nothing in it answers the question the developer actually has, which is whether the answer in front of them rests on solid ground.

The consequence is an ordering, not an exclusion. **The knowledge-state view leads; the engine-activity view sits one level down.** Both read the same monitor snapshot. What separates them is which question the display is composed to answer — and the developer's attention is the scarce resource being allocated.

### Two pieces

The layer is two pieces, with a deliberately clean split:

- **The data sources.** The Health Monitor listens to the event bus continuously and keeps, per session, a live answer to *"what has LLM-GC done to this conversation?"* — its last pass, its history of passes, transitions and degradations. The knowledge-state projection answers *"what does the model know?"* from a read-only inspection of the session. Neither renders anything.
- **The Visualizer (dashboard) — the view.** It renders what those report. It computes nothing of its own, and its one action is the developer's Run GC now.

This document covers the layer as a whole and the data sources in depth.

---

## Why it exists

The motivating asymmetry: an AI model has no reliable internal sense of what it knows and does not know, so a human cannot lean on the model to flag when its grounding is weak. The human *can* exercise that judgment — but only with data. Principle 4 of the project (*the human exercises judgment; the system provides data*) is the operating form of this. The visibility layer is where that principle becomes real.

What would break without it: context management would happen **invisibly**. The engine would score, summarise, archive and fold — quietly changing the material the model reasons over. The developer would have no way to see that any of it happened, let alone judge whether it was safe. A system that manages context invisibly has replaced one opaque actor (the model) with two. That is the opposite of the project's intent.

The Health Monitor is also where the *continuous* view of context state becomes real. Context health is a property of the conversation **as it evolves**, not a property of garbage-collection operations. A view that only existed at collection time would leave the developer staring at stale state during the very reasoning the cleanup was meant to protect.

---

## How it works

### A passive subscriber with a hybrid data model

The Health Monitor subscribes to five events: a pass finished, a span made at the summarised level, a span archived (made archived, or lowered to archived or out of view), a fact superseded, and part of a collection falling short. It is the same publish/subscribe channel every other consumer (logger, benchmarks, the dashboard) reads from; the monitor is just one more subscriber, and the bus neither knows nor cares that it is listening.

**One monitor serves every session, and keeps them apart.** Every event names the session it describes, and the monitor keeps a separate record per session: its last pass, its transitions, and its recent degradations. Events with no session share one record. The number of records is bounded, dropping the session heard from least recently, so a monitor never grows with the number of conversations a proxy has ever served.

**It keeps no conversation content.** A finished pass's signals are built the moment the event arrives and only those are kept — counts, tokens, turn indices — never the result itself, which carries the whole context. Holding it would keep every conversation's text alive in the monitor after its session closed.

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

- **Current-state signals** (token budget, the latest pass's breakdown, each generation's contents) are **read** from the most recent pass's result and from the session's real storage, never re-derived from accumulated events. The components that computed those numbers are the authority on them.
- **What a collection moved is part of its own record**, not inferred. Its outcome names every turn it took, at the level it gave each, and every turn condensed earlier that it showed with less or more detail; the dashboard's timeline is drawn from these records, one entry per collection, and a fact is listed as replaced by the collection that took the turn replacing it. *Rejected: a log of move events, grouped by time.* It recorded only spans made or lowered, so a raised span never appeared; its groups merged collections that ran close together; and a bounded log could hold part of a collection's moves as if they were all of it. *Rejected: inferring moves by diffing each pass's decisions.* A decision is not a move: diffing missed moves made in the same pass a turn arrived, and invented moves for passes that committed nothing. The rule: *only the component that performs a state change is the authority that it happened.*
- **The monitor still logs each move and supersession as it is announced**, for a subscriber that wants moves as they happen rather than per collection.
- **Generation contents are read from storage**, for the same reason: the young generation is the turns a pass forwarded word for word, and the old generation is every span, with what the front spends on them at their levels.

The snapshot it exposes is **built whole on every read** — never cached, never mutated field-by-field — so a reader can never observe a torn, half-updated state.

It is **passive**. It observes and reports; it never triggers a collection pass itself. Deciding *when* to clean context is a decision with consequences for what the model produces next, and that decision stays with the developer or the application. The dashboard's one control, **Run GC now**, does not contradict this: it is the *developer* pulling the lever explicitly — the human judgment the passivity rule exists to protect. It prepares a cut with the batch minimum waived and switches to it at once, waiving the switch policy, so every other rule of a cut still holds and the result is reported like any automatic collection.

### What it exposes — named signals, no aggregate

The monitor's public output is a **snapshot**: an immutable read-out of its current state at a point in time (the `ContextHealth` model). The snapshot carries a set of named, individually-explainable signals, grouped by which of the two questions they answer.

**Transparency signals — about the conversation.** These lead, because they are what the developer acts on.

| Signal | What it measures | The question it answers |
|---|---|---|
| **Knowledge state** | every span with its level, its score, and the exact text the model reads for it; every recorded turn with its words and how the model receives it — word for word, or through its span kept, compacted, archived or out of view; every fact with how it reaches the model and what replaced it | "What does the model still know, and how well?" |
| **Generation lifecycle state and recent transitions** | current young / old / permanent contents (counts and token shares) plus the last *N* promote / archive / supersede transitions | "Where does my decision live now — and did anything supersede it?" |
| **Token budget pressure** | current context tokens as a fraction of the configured budget | "How crowded is the context the model is reasoning over?" |

**Observability signals — about LLM-GC.** These sit one level down.

| Signal | What it measures | The question it answers |
|---|---|---|
| **Latest GC breakdown** | the most recent collection's turns by the level it gave them — kept, summarised, archived — with their tokens, plus the run's status, timing, and failure detail | "How much did the last cleanup change — and did it succeed at all?" A bypassed or failed pass is visible, never hidden behind a normal-looking breakdown; signals honestly report `None` ("not measured yet") rather than a fabricated zero before any pass has run |

**Knowledge state is the signal that most directly serves the project's purpose.** The other signals describe *collector activity*: counts, sizes, durations, transitions. A developer reading "old generation holds 21 spans, pressure 0.64" cannot get from there to "the model no longer has the database decision word for word" — and that inference is the entire product.

Knowledge state is that inference made by the system. It pivots one session's data from *"what did the collector do"* to *"what does the model know"*, keyed by turn and by fact:

```
 turn   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
       ░░ ░░ ░░ ░░ ▓▓ ▓▓ ▓▓ ▓▓ ▒▒ ▒▒ ▒▒ ▒▒ ██ ██ ▒▒ ▒▒ ┄┄ ┄┄ ┄┄ ┄┄ ┄┄ ┄┄
                                                       ▔▔ ▔▔ taken by the next collection

       ░░ out of view   ▓▓ archived   ▒▒ compacted   ██ kept   ┄┄ verbatim, in the tail
```

Here every span is one exchange of two turns.

- **Per span**, its level, its score now, and the exact text the front carries for it, read from the rendered front rather than rendered again, so it is byte for byte what the model reads. A folded span carries its run's shared line.
- **Per turn**, how the model receives it, and its words. A turn in the tail is verbatim. Every other turn is in exactly one span and is received the way its span is shown: quoted, under its summary, by its opening line and facts, or only as a range on a folded run's line. A reply that yielded no fact of its own is no less visible than the question it answered — the model reads their span's line for both. The words come from the span, which keeps them.
- **Per fact**, how it reaches the model: on its span's line when the span is compacted or archived; inside the quoted words when it is kept; not at all when it is folded out of view. *Stored, not sent* is the signal that matters most: the conversation has not forgotten it, and the model cannot see it. Each fact says who stated it — you, or the assistant — and a fact later replaced names the turn, and the speaker, that replaced it.
- **The next collection**: which verbatim turns it would take, as which spans and at what level; which spans already collected it would show at another level; and the conditions it must meet, each with its state — a batch worth preparing, against `min_cut_tokens`, and a cut that pays for itself. The first is planned exactly. The second is estimated at each summary's target size — the very estimate a pass checks before paying for any summary, so a cut the panel shows as not paying is one the pass skips — and measured again on the real summaries; when that refuses a cut, the panel shows the attempt's figures too. Once a cut is staged the card also says what decides its switch: its stale weight against the bound, both in LLM-GC's count, with the bound's share of the model's window and that window converted at the measured rate; its saving against the share times what it would rewrite, the cache state and the gap behind it, and the reason it switched or is waiting ([when-to-collect.md](../generational-memory/when-to-collect.md)). *Rejected: a progress meter alone.* It reads as a promise, and on a conversation of short messages the meter filled, Run GC now was pressed, and nothing happened, with nothing on the panel saying why.

It is built from a single read-only **inspection** of the session that the engine produces with the same rendering that builds the prompt and the same draft and plan a real collection runs, so a dashboard showing it cannot disagree with what the model is sent or with what the next collection will decide.

The projection belongs to the monitoring package, not to the view. The view computes nothing of its own (see *Two packages*), and deciding that a turn is archived rather than quoted is a derivation over spans and facts, not a rendering choice. Putting it in the visualizer would duplicate that logic in every future consumer — the post-session report and the benchmark harness both need the same answer.

The generation-lifecycle signal is the richest of the built three: its temporal dimension — *where* a fact lives and *how long* it has lived there — is information the developer's epistemic vigilance acts on directly. The temporal view is first-class, not metadata. Knowledge state is, in effect, that signal re-indexed by the thing the developer actually cares about.

The tracked transitions are not the same kind of move. **Promote** (a new span made at the summarised level) and **archive** (a span made archived, or lowered to archived or out of view) are changes in how much of a turn the model sees; neither destroys anything, because the span keeps its turns. **Supersede** is contradiction resolution within the permanent generation — a newer fact overriding an older one. A span raised again, because the conversation returned to it, is restored from its own words within the old generation; no turn ever returns to the young generation.

The **transformation ratio** — the proportion of context that has been compacted or archived relative to the total — is computed and kept as an **internal** snapshot field. It is accessible for benchmarking and serves as the baseline any future aggregate must beat. It is *not* surfaced as a headline number (see *Why this shape*).

**Distinct from the span score.** These are conversation-level signals, read by the developer. They are not the *span score* the scoring layer produces — that score is computed per span and feeds the planner's choice of level. Different scope (conversation vs. span), different consumer (developer vs. planner), different purpose; the two share none of their internals.

### Two packages

The layer's two pieces live in two packages, because they have two jobs:

- **`monitoring/`** — the Health Monitor (`ContextHealthMonitor`), its snapshot (`ContextHealth`), and the knowledge-state projection. Pure Python; **no web dependencies**.
- **`visualizer/`** — the dashboard view, and later the post-session report. Web dependencies live here.

The dependency runs one way: the view reads the monitor. Keeping them separate keeps the engine's in-process import graph — which is what wires the monitor into the bus — free of any web-framework code. The doc area ("visibility") names the *layer*; the two packages are its implementation.

The dashboard runs inside the proxy process, beside it rather than within it: the same engine and the same event loop, on its own localhost port. It has three views, one per question. One picker at the top chooses a conversation, grouped by the Claude Code session it ran in; the Conversation view opens first:

| View | Answers | Shows |
|---|---|---|
| **Conversation** | What does the model remember, what has it forgotten, and should I worry? | Those three questions answered in words at the top, with the token counts beneath, each named for whose count it is; then every fact, by who stated it, with how it reaches the model; a map of the conversation; and every exchange, newest first unless switched, with exactly what the model reads for it and what was said |
| **Requests** | What did this piece of work do, and what did it cost? | The Claude Code session turn by turn, oldest first: each turn's requests on a waterfall in time, each subagent under the request that started it, side requests folded away; the session's totals by LLM-GC's count and by the provider's; one row per conversation |
| **Engine** | What did LLM-GC do, and did it work? | What the next collection would take, and what decides when it is switched in; why every exchange is at its level, in the planner's own steps; every pass and its outcome, degradations, transitions, the proxy's counters, and the settings in effect |

Above every view, beside the picker, two chips say the conversation's state in a few words: the provider's cache, warm or cold and for how long, and the next collection, from "not enough yet" to "switches in next request".

It polls once a second, and fetches a conversation's content again only when it has changed, or when its cache turns from warm to cold, which can change the verdict on a prepared collection. Server-push (WebSocket) delivery is a planned upgrade of the transport, not of the split.

**The screen says what the model reads.** For how much of an exchange the model sees, its *level*, the screen uses the words of the marker the model itself reads in the recap: *quoted* for an exchange marked `[Quoted exactly]`, *summarised* for `[Summarised]`, *archived* for `[Archived]`, and *out of view*. A message not yet collected is *recent*. The code keeps its own names, and the guide maps them once:

```
model     [Quoted exactly]  [Summarised]  [Archived]  (one shared line)   (its own message)
screen    quoted            summarised    archived    out of view         recent
code      KEPT              COMPACTED     ARCHIVED    OUT_OF_VIEW         VERBATIM (young)
```

Four more words are plainer on screen than in the code, and each one's explanation names the code's word: a *collection* is a cut, the *recap* is the front, a *check* is a pass, and a *prepared* collection is a staged cut, *switched in* by a request. The provider's cache is *warm* or *cold*, as the design docs say.

*Rejected: the code's level names on screen.* A badge saying *compacted* beside a recap line marked `[Summarised]` gave one level two names, and a reader translated twice. *Rejected: "word for word" for the young generation.* A quoted exchange is word for word too, so the label could not tell a collected message from one no collection has taken, which is the difference that matters. *Rejected: a separate plain vocabulary* (clean-up, condensed block, kept as facts): read beside the docs, two vocabularies for one system confuse more than either. *Rejected: the design vocabulary with a glossary page*: a reader who must leave the screen to decode it has been handed engine telemetry again. *Cost accepted:* five words differ between screen and code, and the guide maps them. Token figures compare like with like: the conversation as written is the session's recorded messages at their own sizes, counted as the engine counts them. The newest message reaches the session only with the next request, and counting it as written showed a saving equal to that message which never happened.

**Run GC now explains its outcome, including when nothing changed.** When a collection would not have shrunk the prompt, the engine reports what it would have done — which messages, how many quoted, summarised or archived, and the prompt size it would have produced — and the dashboard says why in those terms. "It would not have shrunk the context" alone tells a reader nothing they can act on.

**What was said is kept.** The engine keeps every turn it records, whole, for the life of its session: a span in the old generation holds the turns it condenses, so it can be shown at any level again. The dashboard shows a condensed turn's words beside what the model receives instead, read from its span. A transcript recorder the proxy feeds while the dashboard runs supplies only what the session has not recorded yet — the newest request's own message, which joins the conversation when a later request carries it with its reply — and each conversation's opening words for the picker, with the client session and agent its requests named: text only, bounded by session count, dropped with its session. That message is shown as the *newest request*, never as something the user said: clients send side requests too (after each reply, Claude Code asks the model to suggest the user's next prompt), and one arrives with the same history, system prompt and tools as a real next message, so nothing in it tells the two apart. *Rejected: recognising known side requests by their wording* — it is tied to one client's current text, and anything it misses is still attributed to the user. *Rejected: showing the message only while it is being answered* — a chat client's latest message would vanish once its reply arrived. Its reply is shown above it as the *newest reply*, read by the proxy as it reached the client, and replaced by the recorded turn when the next request carries it: someone watching looks for the answer they just read. It is never recorded from there — a regenerated or broken reply must not enter the record — and it is kept only while it answers the newest message, so a slower reply to an earlier request is not shown under a question it did not answer. *Accepted cost:* a reply the client regenerates is shown until the regenerated one replaces it. For the same reason a turn carrying tool results is shown as *tool results*, not as the user's words: the client sends what its tools returned under the user's role. A subagent's relayed report is shown as a *subagent report*, for the same reason. Instructions the client wrote into a message — Claude Code's `<system-reminder>` blocks — show as a tag with their size, not among its words: they are not conversation, are never condensed, and are not counted in the request's size, which covers what LLM-GC can change. Their size is read from the client's latest request, not from the record: a client can rewrite them after the message was recorded, and they are sent as it sends them now. *Accepted cost:* the engine holds every original in memory for every session, dashboard or not — the price of being able to restore one. *Rejected: no originals* — an archived turn could never be shown whole again, to the model or to a person.

**It shows conversation content, so it is opt-in and localhost-only.** The proxy and the dashboard have different exposure: the proxy is an API meant, one day, to be reached behind authentication; the dashboard shows every conversation the process holds. Separate listeners keep the two from being exposed together. Two further guards close the browser-side routes to localhost: requests whose `Host` is not a localhost name are refused, so a hostile page cannot rebind its own domain to 127.0.0.1 and read the dashboard as same-origin; and Run GC now requires a custom header, which another site's page cannot send without a preflight the server never grants.

---

## Why this shape

### Continuous, not per-collection

**Rejected: monitoring only during garbage-collection operations.** A health view that updated only on a collection pass would blink in and out — a developer inspecting the dashboard *between* operations would see stale state, which is exactly when transparency is most useful (during the reasoning, not after the cleanup). Because context health is a property of the evolving conversation, the monitor tracks the conversation, not the collection events. The cost of this choice is recorded under *Consequences*.

### Passive, not auto-triggering

**Rejected: a continuous monitor that also auto-triggers collection under "pressure"** (the way a runtime garbage collector fires under heap pressure). Auto-triggering is more proactive and has an appealing precedent, but pulling the cleanup lever automatically — without an audit trail, and in a way that changes what the model produces next — is contrary to *the human exercises judgment*. Automatic triggering may make sense once a grounded predictor exists to justify it, and even then it should be opt-in.

### Signals, not an aggregated score

This is the central decision. The view must serve epistemic transparency, which has a hard implication: it must be **legible** at a glance *and* **honest** about the limits of what it shows. Anything that looks more authoritative than its underlying math supports is a worse failure than showing nothing — because it puts a *new* black box (an inscrutable number) in front of the developer, in place of the original one (the model).

One caution carries the whole decision: **a number can be perfectly accurate and still be the wrong thing to show.** Accuracy and legibility are different bars, and a transparency surface has to clear both. The case where that bites hardest is the second one below.

Three candidate shapes were considered:

1. **A single likelihood-of-hallucination score** — *"given this context, how likely is the next answer to be wrong?"* This is what the developer most wants. **Rejected on honesty.** Absent ground-truth hallucination labels, a number labelled "hallucination probability" claims an accuracy a heuristic cannot back. It violates *the system is honest about its own outputs*.

2. **A single transformation-ratio score** — the fraction of context that has been transformed. **Rejected for the surface — but for a different reason than (1).** It is honest at the *math* layer: the ratio is a real, exactly-measurable property of the context. It fails at the *UX* layer. A single top-level number, however carefully labelled, becomes the glance-target and is read as authoritative; the qualifier ("transformation ratio, not hallucination probability") becomes decoration and does not survive contact with real UI behaviour. The math's honesty does not rescue the glance.

3. **A set of named, individually-explainable signals** — **chosen.** There is no single number to glance at, so there is no single number to misread. Each signal is named for what it literally measures. The *aggregation* — the question "given these signals, how worried should I be?" — is left to the developer's own epistemic vigilance, which is precisely the human capability the project exists to engage.

The transformation ratio (family 2) is **demoted, not deleted**. The monitor still computes it and exposes it via the snapshot for benchmarks and research, and as the baseline a future learned aggregate must beat. It is simply not a surfaced headline. This preserves the defensible math without the UX hazard of making it a glance-target.

Two further alternatives were rejected:

- **A weighted composite over the signals.** Gameable — any change to the weights yields a different headline number — and harder to defend than the signals it summarizes. The signals are more honest as independent displays.
- **No signals at all until a learned predictor exists.** The signals are genuinely informative on their own terms — they answer questions the developer can act on ("how full is my budget?", "how much did the last cleanup change?", "where do my permanent facts live?"). Shipping nothing until an ideal aggregate exists would mean shipping no visibility at all in the meantime, in a project whose whole point is visibility.

A learned hallucination predictor, done honestly against real labels, is a **planned extension**. If it can be defensibly grounded — correlating with real hallucination signals strongly enough that a developer should trust it — an aggregator ships *behind this same signal interface*, alongside (not replacing) the individual signals. If grounding turns out to be weak, signals-only is the **permanent design**, and that is an honest answer to the research question, not a failure. The project would rather ship several honest signals than one composite whose construction the developer cannot inspect or trust. **Aggregation is earned, not assumed.**

### Context-side signals today; response-side signals as an open direction

Every signal above is derived from **context state** — what was kept, compacted, archived, extracted. The layer never reads the model's *response*. That boundary is worth naming because the proxy is the one component in the system that sees both sides: the context that was sent and the answer that came back. Nothing else in the ecosystem is positioned to join them, because nothing else knows what was removed.

Two response-side signals are credible, and they are not equally good.

**Grounding failure is the strong one.** When the model asks for information the permanent generation already holds — *"which database are you using?"* when a `database` fact is stored and was not injected — that is near-airtight evidence that cleanup removed something load-bearing. It is specific, it has high precision, and it is the only signal available from live traffic that can **falsify** the *never make things worse* principle rather than merely illustrate it. A close variant is the model restating something as unknown that the session established earlier.

**Hedging is the weak one**, and weak in an instructive way. Hedged language ("I think", "it seems", "based on what I recall") is cheap to detect and needs no labels, which makes it tempting as an early stand-in for the hallucination signal the project cannot yet ground. But it is confounded by prompt phrasing, question difficulty, system-prompt style, and model version, so a rise in hedging cannot be attributed to cleanup. Worse, its failure mode runs the wrong way: a model missing its grounding frequently answers *confidently and wrongly* rather than hedging. Absence of hedging is therefore not evidence of good grounding, which is precisely the inference a developer would be tempted to draw. Hedging can be recorded as context alongside a genuine signal; it cannot carry a claim on its own.

Both are correlational. The instrument that gets closer to causation is **counterfactual replay**: re-issue the same query against the full, uncompacted context and compare the two answers. That is expensive, so it belongs to benchmarking rather than to the live path — but it is what turns *never make things worse* from a principle into a measurement.

This whole direction expands the layer's input surface from "context state" to "context state plus model output", which is a real architectural change, not a new display. It is recorded here as a considered direction, not a commitment.

### A turn is shown by its span's level

How the model receives a turn is named after the detail levels — verbatim, kept, compacted, archived, out of view — and decided by the span the turn is in, never by the turn's own facts. *Rejected: one value for archived and out of view, told apart by whether the turn's own facts are sent.* A reply that yielded no fact of its own then read as "the model sees nothing of this message" while the model read its span's archived line: the screen overstated the loss, and a warning that is usually wrong teaches the reader to skip the one that is right. *Rejected: a "contradicted" level for a turn whose facts were all replaced.* Its span is still shown at its level, with a note naming the replacement, so the turn is no less visible than before; the replacement is a property of the fact, and the fact carries it.

### Facts are an audit of the prompt, not a digest of the conversation

Under the conversation's totals, every fact read so far is shown by who stated it, with how it reaches the model, and every value replaced beside them:

```
What the model is told, fact by fact
As of message 21 · 13 facts · 12 reach the model · ⚠ 1 doesn't · nothing replaced
Facts are read from messages 0–21. Messages 22–29 are sent word for word and read
only once collected, so a fact here may be out of date: a value changed there
reaches the model now, and shows here then.

┌ YOU SAID ──────────────┐ ┌ THE ASSISTANT SAID ────┐ ┌╌ ↺ ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐
│ index shards           │ │ master nodes           │ ╎ No replacements on     ╎
│ 9 primaries, 2 replicas│ │ recommended 3 dedicated│ ╎ record. When a later   ╎
│ message 6 ✓ sent as a  │ │ message 1 ⚠ not sent → │ ╎ message changes a      ╎
│ fact                 → │ │                        │ ╎ value, both appear here╎
└────────────────────────┘ └────────────────────────┘ └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
› How fact collection works
```

The section is meant to answer four questions in seconds:

- **What survived?** Each fact is either ✓ sent as a fact, listed in the recap beside its exchange's summary or opening line; ✓ sent word for word, inside its exchange when that exchange is kept; or ⚠ not sent, when that exchange is out of view. "Sent" is read from the facts the rendered front carries, so every status can be checked against the bytes forwarded.
- **What changed?** Each replacement shows the old and new values, links to both messages, and says whether the recap notes it. The recap does, on the old value's line, unless that exchange is out of view, and then the old value isn't sent at all.
- **Whose value is it?** Who said a fact is its column. That decides what a later message may change: only the user replaces a decision the user made. When facts are read from the user's turns only, as they are without a model, the assistant's column says it was not read: empty, it would claim the assistant settled nothing.
- **Where did it come from?** Every fact names its message and jumps to it.

Every label names what the model gets, so the section reads without its guide. The headline is worded so its parts visibly add up: "13 facts · 12 reach the model · ⚠ 1 doesn't", with replaced values counted apart as earlier values, never as facts. A status is an outcome, "sent as a fact" or "sent word for word", not where the fact sits in the recap, and its meaning is one hover away, its dotted underline saying so. *Rejected: "shown verbatim" and "summarised only".* A fact sent as a fact is the one a summary can't drop, since it is listed on its own; "summarised only" would warn about the safest case.

What needs a look still comes first. The headline counts the facts not sent, they sort above the rest of their column, and replacements have a column of their own. None of them sits under forty notes that need nothing. The section folds, as the Engine tab's cards do, but its headline, the note on how far the facts reach and the warning for facts that couldn't be read stay in view. Folding hides detail, never a fact the model cannot see.

Each column scrolls inside a box about four and a half cards tall, so the card keeps its height however many facts a long conversation piles up, and the conversation below it never moves further down the page. The cut card, and a fade on whichever edge has more, show the column goes on; its heading counts every fact in it. A column keeps its place when the section is redrawn, as it is at every update and keystroke. *Rejected: listing the first few with "Show all".* Opened, it grows the page by every fact, which is the length the section exists to contain.

The value is what each fact card shows largest, with its topic as a label above it: a value without its topic ("9 primaries") says nothing. An empty column is a dashed card saying why it is empty, not a blank, so it never reads as content that failed to load; when the assistant's turns are not read, its card says so rather than promising facts that will never come. What each status means is said once, in one place, and shown twice: on hover over the status, and under "How fact collection works", folded below the columns.

*Rejected: a model-written digest (goal, key decisions, implementation, open questions).* It would be one model's account of what another was told, paid for at every collection and stale by the next message. It would also be the only thing on the page that couldn't be checked against what was sent. Open questions appear only inside summaries, and nothing records them as items or knows when one is answered.

*Rejected: "what the model knows", or "what the model is told" alone.* The model is told far more than its facts, since summaries and recent messages carry the rest, so either title would pass a small annotation off as the whole prompt. The dashboard shows what was sent, not what the model made of it, and "fact by fact" limits the claim to what is shown.

*Rejected: "current facts".* Facts are read only at a collection, so the headline opens with the message they are as of, the newest collected, and the note under it says what that costs: newer messages go word for word and are read only once collected, so a change of plan in one of them reaches the model in its words while a fact here still shows the old value. Headed "current", a value the conversation had already changed would read as still in force.

### The page answers three questions before it shows numbers

The project's promise is to show what the model remembers, what it has forgotten, and when to worry, so the Conversation view opens with exactly those three, in words:

```
 WHAT THE MODEL REMEMBERS          WHAT IT HAS FORGOTTEN               SHOULD YOU WORRY?
 9 recent messages, as written      20 messages no longer word for word  ✓ No. Every fact reaches
 12 messages as 3 summaries         ✓ nothing out of view                  the model, nothing is out
 8 by their opening line and facts  ! 21 messages' facts couldn't        of view, and no collection
 6 of 7 facts told to it              be read                              fell short.
 ─────────────────────────────────────────────────────────────────────────────────────────────
 Next request ≈ 3,207 tokens of conversation + ≈ 1,840 of the model's earlier thinking · 71% less than as written
 Last request 55,712 tokens by the provider's count · 28% of the model's window · 1.48× LLM-GC's count
 Last collection 3 min ago · What moved →  · So far 16 requests, 29% saved
```

- **Each count names what it counts**, in the level words above, and the numbers come beneath the words, each named for whose count it is: LLM-GC's own tokenizer is an estimate, the provider's is what it bills.
- **The model's earlier thinking is shown apart from the text.** It has no text, so its size is LLM-GC's estimate from each reply's usage ([the proxy](../proxy/overview.md#the-models-hidden-thinking)), a rougher kind of figure than a tokenizer's count. Wherever it is part of a number, it is said beside it: the next request's thinking, the share saved against the conversation as written, a prepared collection's saving and stale weight, and the thinking each collection stopped sending. *Rejected: folding it into the text figures* — one number would then mix a count with an estimate, and a reader could not tell how much of it to trust.
- **"Should you worry?" holds every warning**, in one place, each linking to its evidence: messages out of view, facts the model can't see, a collection that fell short, a window more than 80% full, and a rate-limit pause. Stale weight near its bound is mentioned without alarm. A warning can be dismissed and returns only when there is more of what it warns about.
- **The all-clear is said only when it is true**, never because the warnings were dismissed: then it says that nothing is new since.

One warning cannot be dismissed: the upstream is refusing LLM-GC's own credential. A single summary that falls short is harmless and passes, but a refused credential makes every later collection fall short, so the prompt silently stops getting smaller. Each shortfall alone says nothing about the cause; only together do they say the credential is bad. It is shown first, in the critical tone, says what still holds and what to do, is found from the engine's error code rather than by matching text, and clears as soon as a later collection finishes with nothing falling short.

*Rejected: opening on the prompt's size.* A token count answers "how big", which is not the question a reader brings, and the promise was hidden three cards down. *Rejected: standing banners for good news and routine state.* Above the page on every visit, a warning among them was one more line to skim past. *Rejected: an aggregated health score* (see *Signals, not an aggregated score*): the three answers are each a signal stated plainly.

### What lifts is what opens something

A block that opens something when clicked rises a little on hover and on keyboard focus, and settles when pressed: a message's bar on the map, a fact's card. The movement is the promise of a click, so it is kept for blocks that keep it. Everything else that is clickable already says so in its own way: a row that unfolds in place carries a chevron, a link underlines, a button shades. A static card does not move at all. *Rejected: lifting every card.* The dashboard's cards are panels to read, not things to open, and a panel that rose under the pointer would promise a click that does nothing, which teaches the reader to stop trusting the lift where it does mean something.

### Explanations come in two lengths

Every term, card and status carries an explanation where it appears, in two lengths. A tooltip gives the short one: what this is, in a sentence or two, about thirty words at most. The guide holds the full one: every term and every card, in a reference at its foot, and clicking any ⓘ opens the guide at that entry. On the page itself, a card says what it needs to in one line, and a phrase underlined with dots holds the rest of it, on hover or focus.

```
 hover an ⓘ     →  title · one or two sentences · "Click for the full explanation →"
 click it       →  the guide opens at that entry, marked briefly
 dotted phrase  →  the detail a one-line note left out
```

*Rejected: the full explanation in every tooltip.* Some ran past a hundred words, read while holding the pointer still, and one gluing two explanations together ran longer still; a reader stops reading long before the end. *Rejected: dropping the long explanations.* The project exists to explain what happened to the model's context, so the detail stays, one click away.

### The map draws what the model reads

The map draws one bar for each exchange in the recap and one for each recent message: the units the model reads, an entry of the recap or a message of its own. A bar's width is its size as written, in tokens, and its height how much of it the model still has, so a summarised exchange is a short bar as wide as what it replaced. A collected exchange's bar names its messages, and clicking it opens the exchange in the list below.

- **It fits a long conversation.** A collection turns many messages into a few exchanges, so the map grows with what the model reads rather than with every message ever sent. Each bar still keeps a minimum width, never under 10 pixels, solved by repeating the step until it settles, so a one-line message stays clickable.
- **Past the card's width it scrolls sideways as one.** The group labels, the bars, the Sent row under them and the axis share a single scroll area. It opens at the newest end, where the live action is, and the edges fade so bars out of view are not read as the start or end of the conversation.
- **The map and every card below it fold**, and each remembers how it was left.

*Rejected: one bar per message, collected or not.* A long conversation became a strip of slivers, and for collected messages it drew a distinction the model doesn't have: it reads one recap entry for the whole exchange. *Rejected: giving every bar the same width,* which loses the scale that shows a one-line reply beside a pasted file. *Rejected: a zoom control,* another thing to learn while the default stays unreadable.

### The exchange list opens on what just happened

Under the map, the exchanges are listed newest first, and a control beside them turns the list over; the choice is kept in the browser. A reader opens the dashboard to check what the model has just received. Oldest first, that is at the foot of a list whose first screen is the recap, the part read least. The map already opens at its newest end, and the list now agrees with it about which end matters.

- **The order is only how the rows are laid out.** What is folded, and where each part of the prompt begins, are worked out oldest first, as the model reads it. The rows are then laid out in the chosen order, so each part's label leads it either way, and the button that loads earlier messages sits at the end farthest from the newest. An opened run of archived exchanges follows the list's order; what was said inside one exchange still reads as it was spoken.
- **The page says how its order relates to the model's.** The subtitle names the order showing and, newest first, says the model reads it the other way round. A phrase that meant "first" says "oldest": the next collection takes the oldest recent messages.
- **The exact-prompt view turns over with the same control.** Both views of the card follow the one choice, so changing view never changes the order under the reader, and the control is always there. Each view's subtitle says how the order showing relates to the model's. The claim that a view is what the model reads is carried by that subtitle and by the copy button, which always takes the prompt in the model's order, as its tooltip says; it does not depend on how the rows are laid out.
- **A redraw keeps the reader's place, in both views.** The list is rebuilt whenever the conversation changes, and newest first grows at its head, which would push down whatever a reader had scrolled to. The first row in view is put back where it was. A reader still at the head of the list sees new rows arrive. A row a collection has folded into the recap is gone, and the reader is left where the page puts them.

*Cost accepted:* the map still runs oldest to newest from left to right, so the list and the map run opposite ways; hovering a row or a bar still marks its counterpart. *Rejected: a "jump to newest" button on an oldest-first list.* The page still opened on the part read least, and the button only worked around that. *Rejected: reversing the rows with the stylesheet.* It turns the picture over but not the order that keyboard focus and screen readers follow. *Rejected: leaving the exact-prompt view oldest first, with the control hidden there.* One card then changed order whenever a reader changed view, with nothing to change it back, and its subtitle still said "newest first" above rows that were not. The worry behind it, that a turned-over view would misstate the model's order, is answered by saying the order, as the other view does.

### One picker, and the page opens on the conversation

One Claude Code session reaches the proxy as several conversations: its main one, one per subagent, and a new main one after `/compact` or an edited message, each collected on its own (see [the proxy](../proxy/overview.md)). One picker lists them all, grouped by the session they ran in, and the page opens on what the model remembers:

```
top bar     [ Conversation | Requests | Engine ]
bar         Conversation ▾  Claude Code session 6d84277c · since 10:56
                              Main · Find where requests are validated · 28 messages · 2 collections
                              Subagent 1 · Survey every place a request body is read
                              Earlier main · …
            Cache warm · 3 min left   Collection: not enough yet · 986 of 1,000 tokens   [Run GC now]
```

- **One choice drives every tab.** The picked conversation is shown on the Conversation and Engine tabs, and the session it ran in on the Requests tab. A session whose conversations have all closed is still listed, for its requests alone; conversations whose client names no session are grouped on their own.
- **Each conversation is named by its part.** The newest main conversation is *Main*, the one in use; subagents by the order they started; a replaced main conversation is *Earlier main*.
- **The main conversation is followed.** Until the person picks another, the one shown is the session's main conversation, and when `/compact` replaces it the page moves to the new one.
- **The trace opens a conversation where it matters.** *Open this message* shows the turn's message in its conversation, with a way back to the Requests tab.

*Rejected: a session picker with a conversation switcher under it.* Two pickers with two scopes, and "session" meant both a Claude Code session and, in the code, a conversation. *Rejected: opening on the request trace.* It answers what a piece of work cost, not what the model knows, which is the question the project exists for. *Rejected: hiding subagent conversations.* They are prompts LLM-GC forwards and records, and hiding them would hide what it does.

### Measured and forecast are never one number

The page holds three scopes, each named where it is read, and two kinds of number, never mixed:

| Scope | Where | Measured: what requests carried | Forecast: what the next request will |
|---|---|---|---|
| One conversation | Conversation tab, the summary's foot | *Last request*, by the provider's count; *so far*, its requests and saving | *Next request*: the recap and recent messages, against the conversation as written |
| One client session | Requests tab: its totals, its turns, its conversations | Every request it made, by turn and by conversation; the provider's usage; LLM-GC's own calls | — |
| The proxy | Engine tab, *across all conversations* | Every request since the proxy started, the provider's usage, LLM-GC's own calls | — |

Where each sits on the page:

```
┌ Requests tab ────────────────────────────────────── MEASURED · the session ───────┐
│ SENT 146,108 · FORWARDED 113,453 · READ FROM CACHE 1M · WRITTEN 32k · …  (tiles)   │
│ ▶ Turn 1 "The checkout tests fail…" · 18 requests · 2 subagents                   │
│ ▶ Turn 2 "Good. Add a regression test…" · 5 requests · −27,556 · ◆ 2 collections  │
│ ▼ Turn 3 "This session is being continued…" · 3 requests     ← newest, open       │
│     Your message ▕█▏  Tool results  ▕▏  Tool results     ▕▏                       │
│ Conversations in this session: Main · Subagent 1 · Subagent 2 · Earlier main      │
└───────────────────────────────────────────────────────────────────────────────────┘
┌ Conversation tab ─────────────────────────────────────────────────────────────────┐
│ REMEMBERS · FORGOTTEN · SHOULD YOU WORRY?                         (in words)      │
│ Next request ≈ 3,647 tokens of conversation          FORECAST, LLM-GC's count     │
│ Last request 5,410 tokens · 0.5% of the window       MEASURED, the provider's     │
│ So far 6 requests, 44% saved                         MEASURED, LLM-GC's count     │
└───────────────────────────────────────────────────────────────────────────────────┘
┌ Engine tab ───────────────────────── MEASURED · every conversation, since start ──┐
│ requests · total sent · what the provider reported · LLM-GC's own model calls     │
└───────────────────────────────────────────────────────────────────────────────────┘
```

The session's table is the answer to "which conversation cost and saved what": the conversations a reader thinks of as one session, counted request by request, each a row that opens it. A collection made after a conversation's newest request is marked *not yet sent*: a finished subagent's last collection saved nothing, and a forecast of the next request's size, shown as "saved", would have claimed it did. *Rejected: summing the conversations' next-request sizes* — each is one request's forecast, and a session's spending is every request it made. *Rejected: drawing subagents inside the main conversation's map* — their messages were never in the main conversation's prompt, so the map would show tokens it never carried.

The model's window is the one the provider reports for the conversation's model, or the configured one, said to be assumed.

A subagent's conversation, recorded but not collected, says so wherever a collection would be forecast: its Collections cell reads *not collected*, the summary and the *Next collection* card name the reason instead of a forecast, and *Run GC now* explains rather than collects. Its map and messages are shown as for any conversation, every message sent as written.

### A session is traced turn by turn

A Claude Code session is one piece of work to the person running it, and several conversations to LLM-GC. The trace shows it the first way: the turns the person typed, each with every request it led to and every subagent it started, while each conversation keeps its own detail view.

```
Turn 1 · "Use two subagents in parallel: …" · 11 requests
    your message → Agent, Agent
      ├ Subagent 1 · "Count the test files…" · started by Agent toolu_A · not collected
      │   task → Bash · tool results → Bash · tool results → SubagentHandback
      └ Subagent 2 · "Summarise README.md…" · started by Agent toolu_B · not collected
    tool results · side request (not kept) · subagent report · subagent report
Turn 2 · "Thanks. Which test runner do we use?"
```

- **Oldest turn first, the newest open.** The trace reads in the order things happened, as the conversation does; earlier turns fold behind *Show N earlier turns* above them, as a chat loads older messages.
- **A turn is a span of the main conversation's time.** It begins at a message the person typed that the conversation kept, and runs to the next. A retry of the same message does not begin another, and a typed message whose place another request took is *pending* until the conversation carries one of them on. The main conversation's tool rounds, its side requests and requests no conversation has claimed belong to the turn their time falls in; requests before the first turn — a quota check — are listed before it.
- **A subagent belongs to the turn that started it.** Through its link to the call that started it, a subagent stays under that turn while it runs on after the person types again, and a subagent another subagent started stays with the first. One no call is known to have started falls in the turn of its first request.
- **Cost adds up the tree; a window does not.** A turn's totals count its own requests and its subagents'. Each conversation's window and recap stay its own: a subagent's requests were never in the main conversation's prompt.
- **Tokens by kind, as the provider counts them, not dollars.** A token count adds kinds of token whose prices differ a hundredfold: the tokens LLM-GC removes are mostly cache reads, the cheapest, and a collection changes the recap, so the prompt after the system prompt is written to the cache again, among the dearest. A session can save tokens and cost more. So the provider's own count is shown by kind — cache read, input, cache write, output — with the cache writes on requests that carried a collection totalled for the session and each turn, and shown on each such request, beside what LLM-GC saved. LLM-GC's own model calls are shown in tokens too. *Rejected: dollars at list prices* — no provider serves its prices, so a price list would be a table kept by hand for every model of every provider, wrong for a negotiated price or a gateway's own, and silently stale when a price changes. The provider's counts are read from every reply, exact, for any model, with nothing to maintain; turning them into one figure is left to whoever knows what their tokens cost.
- **Built on read, from the request log.** A record settled since the last read changes the trace with it. The words a turn or a subagent is recognised by come from the transcripts the dashboard holds, so a closed conversation's turns show without them.

*Rejected: nesting by time alone* — a background subagent that outlives its turn would appear under a turn that never asked for it. *Rejected: one swimlane per conversation across the whole session* — a long session becomes a wide chart no one can read; within an open turn, lanes are worth it.

### The conversation view keeps no collection log

Collections are what LLM-GC did, and the Engine view is where that is shown: what moved at each one, and the prompt's size check by check. The conversation view carries only what a reader of the conversation needs, beside the message counts a collection changes:
- a collection running now;
- when the last one ran, with a link to what it moved on the Engine view;
- what the next one will change, which also shows on the map.

The exchanges therefore run the full width of the page, with their words held to a readable line length. *Rejected: a column of recent collections beside the exchanges.* It repeated those lines in a narrow column that sat empty below them for the length of a long conversation.

### The next collection is previewed, not remembered

The preview runs a collection's own first steps — find the cut point, draft the spans, score and plan every span, the ones already collected included — on the session as it stands, with no model call and nothing announced on the bus; a preview reported as a collection would have every subscriber record work that never happened. *Rejected: show the last real pass's plan.* It describes the conversation before that pass's own cut, so straight after every collection it would show the batch just taken. *Accepted cost:* the preview drafts and scores on every change to the session, on the event loop; cheap with word-overlap relevance, and to move to a worker thread if an embedding implementation is configured.

### Intent and outcome are different signals

A collection can complete and still fall short of its own plan. The plan wants a span summarised, the compactor refuses the summary it gets back, and the span is kept word for word instead. Nothing is lost and the context is correct — it is just larger than the collection set out to make it.

A dashboard that drew only the plan would draw transformations that never occurred, which is the same class of failure as the aggregated risk score this layer exists without: a confident-looking number with nothing behind it. The difference is that here the subject is the system's own behaviour, so getting it wrong means the transparency layer is the thing lying.

So a completed pass carries a structured record of what it intended and did not manage, with the reason separated by what an operator would do next — a refused summary means the run or the cap was wrong; a failed provider call means check the provider; a failed extraction means check the extractor; a front that cannot fit means the budget is too small for the conversation. Each one is announced on the bus as it happens as well as collected onto the result, so a subscriber does not have to wait for the pass to end, and degradation is never something a consumer has to infer from counts that fail to add up.

Failure is not the only gap between intent and outcome. The collection that takes new turns also re-plans every older span, lowering some and raising others (see `generational-memory/the-cut.md`). So a committed collection carries its **outcome**: which of its new turns went to which level, and which turns condensed earlier it showed with less or more detail in the same pass, each named by turn. The dashboard draws what a collection did from the outcome, and what it could not do from the degradations.

Two decisions inside this that are easy to get wrong:

- **The run status stays "completed".** A separate status value for a degraded pass was rejected: status answers *did the pass run*, and degradation answers *did everything it intended happen*. Fusing them means no consumer can ask the first without handling the second, and every equality check already written against the status silently stops matching — which is precisely what a new enum value is worst at. What would change this: a degraded pass needing different *handling* at the HTTP boundary rather than different reporting.
- **An arbitrary exception's message is not recorded.** These records reach the dashboard over HTTP, and payloads carry no raw conversation content. The project's own exception messages are content-free by convention, so theirs are kept; anything else — including an exception from a compactor written by someone else — contributes only its type name. A failure cause is not a trust boundary we control.

---

## Consequences

- **The signals are correlative with hallucination risk, not causal.** High token-budget pressure does not *guarantee* a bad answer; a quiet generation lifecycle does not *guarantee* a good one. The signals are observations about context state, not predictions about answer quality. That distinction must be stated prominently anywhere the signals are displayed — otherwise they quietly become the same authoritative black box the aggregated score was rejected for.
- **A clean pass must stay clean.** The degradation record is only useful while it is rare: a field that is populated on most passes becomes noise consumers learn to skip, and then it is worse than absent, because its presence implies someone is reading it. Anything that routinely degrades is a design fault to fix rather than a condition to report.

- **The monitor's view drifts if the engine grows without it.** Every new event type that affects context state has to be considered for inclusion in the monitor, or its view silently falls out of sync with reality. This is a recurring tax — the price of the continuous design — and it is paid every time the engine gains a new state-changing behaviour.

- **Until a grounded predictor exists, the hallucination-transparency claim rests on the developer's own vigilance** acting on observable signals, not on a system-produced number. This is a deliberate choice, and a defensible product: the signals do the part that can be done honestly, and the human does the part that cannot yet be automated honestly.
