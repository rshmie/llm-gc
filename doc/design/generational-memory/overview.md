# Generational Memory Model

## The Problem

LLM conversations accumulate context linearly. A 100-turn conversation at 300 tokens/turn is
30,000 tokens. Context windows have hard limits. When they fill up, existing approaches either:

- **Truncate from the top** — knowledge from early turns is permanently lost
- **Sliding window (keep last N)** — same loss, just predictable timing
- **Compress everything into summaries** — smaller, but summaries grow linearly with
  conversation length and become lossy-on-lossy as they compound

All three approaches share the same failure mode: **information loss is inevitable and
uncontrolled.** The system cannot guarantee that a critical decision from turn 5 survives
to turn 500.

Generational memory solves this: turns age through tiers of decreasing token cost, and
before any turn is fully removed, its **information** is preserved permanently — extracted into
a durable knowledge entry where the extractor recognises one, or kept **verbatim** where it does
not. The *knowledge* survives, and nothing leaves active context without a trace.

---

## How Turns Get Classified (Score-Driven, Not Position-Driven)

Classification into generations is driven by the **weighted relevance score** from the
RelevanceScorer (recency + similarity + density + decision + reference signals). The sweeper
uses these scores to classify each turn:

```
  score >= keep_threshold (default 0.7)     → KEEP      → Young Generation
  archive_threshold <= score < keep_threshold → COMPACT   → Old Generation (after summarization)
  score < archive_threshold (default 0.3)   → ARCHIVE   → Permanent Generation (after extraction)
```

**Override rules (applied regardless of score):**
- Last N turns → always KEEP (configurable via `last_n_turns_to_keep`)
- System prompt → always KEEP

**Score-boosting signals (not overrides — contribute to higher score):**
- Turns referenced by later messages → higher reference score component
- Turns semantically similar to current query → higher similarity score component

This means: a turn from 30 turns ago with a high relevance score (e.g., a key decision that's
still referenced) stays in young generation. A recent turn with low information density could
theoretically score below keep_threshold. **Position is ONE input (recency scorer), not the
sole determinant.**

---

## The Three Generations

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUNG GENERATION                              │
│                                                                     │
│  Contains:   Turns classified KEEP — either by score (>= keep_     │
│              threshold) or by override (last N, system prompt)      │
│  Kept as:    Full verbatim text, no modification                   │
│  Storage:    In the messages array itself (no external storage)     │
│  Token cost: HIGH (full verbatim text)                              │
│  Analogy:    Working memory - crystal clear, immediate recall       │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐         │
│  │ Turn 18: "Let's add rate limiting to API" [score 0.85] │         │
│  │ Turn 19: "Use Redis for token bucket" [score 0.79]     │         │
│  │ Turn 20: "Here's the middleware code" [override: lastN]│         │
│  │ Turn 3:  "Project uses microservices" [score 0.72]     │  ← old  │
│  │          turn, but still high score (referenced often)  │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        OLD GENERATION                                │
│                                                                     │
│  Contains:   Summaries of COMPACT runs (score between archive_      │
│              threshold and keep_threshold), and KEEP turns quoted   │
│              word for word when a cut takes them. Each entry records│
│              the turns it stands in for.                            │
│  Size:       Variable (bounded — entries eventually ARCHIVE)        │
│  Storage:    Session store                                          │
│  Token cost: MEDIUM (~80 tokens per 5-turn summary block)           │
│  Analogy:    Short-term memory — "we talked about auth, something   │
│              about JWT..." You remember the gist, not every word.   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐         │
│  │ [Summary of turns 1-5: Set up project, chose           │         │
│  │  PostgreSQL for DB, discussed schema options.] [80 tok] │         │
│  │                                                        │         │
│  │ [Summary of turns 6-12: Implemented JWT auth with      │         │
│  │  7-day refresh. Redis for session storage.] [100 tok]   │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     PERMANENT GENERATION                             │
│                                                                     │
│  Contains:   Extracted knowledge entries (facts, decisions,         │
│              preferences) from turns classified ARCHIVE              │
│  Size:       Grows sublinearly (deduplicated - one entry per topic) │
│  Storage:    In-memory, session-scoped (persistence not yet built)  │
│  Token cost: LOW (~10 tokens per entry)                             │
│  Analogy:    Long-term memory — "the project uses PostgreSQL."      │
│              A knowledge entry, detached from the conversation.     │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐         │
│  │ fact: "Database is PostgreSQL"         [source: turn 5] │         │
│  │   -> SUPERSEDED by: "Database is MySQL" [source: turn 15]│         │
│  │ fact: "Auth method is JWT"             [source: turn 8] │         │
│  │ fact: "Token expiry: 7 days"           [source: turn 8] │         │
│  │ pref: "Prefers explicit error handling" [source: turn 12]│        │
│  └────────────────────────────────────────────────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Growth Characteristics

| Generation | Placement Criteria | Token Cost | Growth Pattern | Bounded? |
|-----------|-------------------|-----------|---------------|----------|
| Young | score >= keep_threshold OR override | High | Variable (depends on score distribution) | Soft — more high-scoring turns = larger young gen |
| Old | archive_threshold <= score < keep_threshold | Medium | Linear with conversation length | Partially — entries eventually archive |
| Permanent | score < archive_threshold (after extraction) | Low | Sublinear (deduplicated facts) | Effectively yes — unique decisions are finite |

The key insight: **summaries grow linearly, but knowledge entries grow sublinearly.** Whether you
discussed PostgreSQL across turns 5, 15, and 45 — the permanent gen holds ONE entry. This is
what prevents unbounded context growth.

One caveat to that bound: what the extractor cannot read is preserved **verbatim** as a `RAW`
entry — the whole turn when it reads nothing, or the sentences no fact came from when it reads
part of one. That is the no-information-loss guarantee. Those are full-size and not deduplicated, so the
sublinear bound holds for *recognised* knowledge, plus a verbatim term that scales with how often
extraction finds nothing. The project accepts that storage cost in exchange for never dropping a
turn without a trace.

Note: young gen is NOT purely "last N turns." Last N is an override guarantee (those turns are
ALWAYS kept). But other turns can also be in young gen if their relevance score is high enough.
A critical decision from turn 3 that's still referenced by turn 20 may score 0.72 and stay KEEP.

---

## Turn Lifecycle — How Classification Determines Fate

The lifecycle is **not** purely linear (young → old → permanent in order). It's **score-driven
on every GC pass**. A turn's generation can change as its relevance score changes over time.

```
Turn arrives in conversation
  │
  ▼
SCORING ──── RelevanceScorer computes weighted score:
  │           recency + similarity + density + decision + reference
  │
  ▼
SWEEPER ──── Classifies based on score + overrides:
  │
  ├─── score >= keep_threshold OR override (lastN, system) ──→ KEEP
  │                                                              │
  │                                                              ▼
  │                                                         YOUNG GEN
  │                                                         (full verbatim)
  │
  ├─── archive_threshold <= score < keep_threshold ──────→ COMPACT
  │                                                              │
  │                                                              ▼
  │                                                         PROMOTION
  │                                                         (summarize with adjacent turns)
  │                                                              │
  │                                                              ▼
  │                                                         OLD GEN
  │                                                         (compressed summary)
  │
  └─── score < archive_threshold ────────────────────────→ ARCHIVE
                                                               │
                                                               ▼
                                                          EXTRACTION
                                                          (FactExtractor pulls facts/decisions)
                                                               │
                                                               ▼
                                                          PERMANENT GEN
                                                          (facts stored, turn removed)
```

### Score Changes Over Time (Why Turns Move Between Generations)

A turn's relevance score is **not static**. It changes every GC pass because:
- **Recency decays** — as new turns arrive, older turns score lower on recency
- **Reference changes** — if later turns stop referencing an old turn, its reference score drops
- **Similarity shifts** — as the conversation topic changes, old turns may become less similar
  to the current query

This means a turn classified as KEEP today could become COMPACT on a future GC pass
(its score dropped below keep_threshold). And a COMPACT turn could become ARCHIVE later
(score dropped below archive_threshold). This is the natural aging process — driven by
relevance, not just position.

Classification says what a turn *would* become; a **cut** is when it happens, for
the oldest turns in one batch (see [the-cut.md](the-cut.md)). A KEEP turn that a
cut takes is not lost: it is quoted word for word in the front.

---

## How Context Is Assembled (on gc.collect)

The context sent to the model is **a front, then the verbatim tail**:

```
┌──────────────────────────────────────────────────────┐
│ System prompt (the client's, unchanged)              │
│                                                      │
│ FRONT — one user-role message, present after a cut   │
│   facts recovered from archived turns                │
│   "[Memory: (turn 15) database: mysql]"              │
│   summaries and quoted turns, in conversation order  │
│   "[Compacted turns 6-12] Implemented JWT auth..."   │
│   "[Quoted exactly] (turn 5, user) We chose ..."     │
│                                                      │
│ TAIL — every later turn, exactly as the client sent  │
│   Turn 18, Turn 19, Turn 20                          │
│                                                      │
│ The new turn (unchanged)                             │
└──────────────────────────────────────────────────────┘
```

The front changes only when a cut is made, which is the subject of
[the-cut.md](the-cut.md): when a cut happens, what it contains, and why the prompt
may be changed only there.

**Facts are retrieved, not dumped, and the budget keeps the system stable.**
Permanent generation accumulates for the life of a session, so the front carries a
budgeted selection of it, chosen when the front is made: decisions first, then
facts, then preferences, with raw excerpts only in the room left over. The facts
are real tokens in the real prompt and count against the context like any other
message. Setting either memory budget to zero leaves facts out, which is how a
deployment opts out and how a benchmark measures what they are worth. Ranking sits
behind an interface, so a deployment with its own ranking supplies one without
changing the engine.

**The budget is a size, not a count.** The front carries facts up to a share of the
model's window (10% by default), so memory grows with the conversation only until
that share is spent and can never crowd the window. *Rejected: a count limit.* Facts
are chosen once per cut, with no question to rank them against, so a count can only
leave out facts the budget has room for; cost and the window are both measured in
tokens, which the size budget measures directly. *Rejected: sending everything*,
which lets memory grow without bound until it costs more than the turns it
replaced. When the budget is reached, what does not fit is shown as out of view;
condensing the permanent generation further is the answer to that.

**Selection and presentation use different orderings.** Importance decides what
survives the budget; the surviving facts are shown in conversation order, because
the front is read top to bottom and a fact from turn 2 belongs before one from
turn 40.

**The front says what it is.** It carries a header naming it as condensed history,
each fact carries its origin turn, each summary its provenance marker, and each
quote is marked as quoted. Distilled content presented as original conversation
would be the dishonesty the provenance markers exist to prevent. A `RAW` entry is
labelled an excerpt rather than a parsed fact, because extraction found nothing in
that turn.

**Superseded facts never reach the prompt.** Only active entries are candidates.
This is the step that makes contradiction tracking worth anything: without the
filter the model would read an old value and its correction as two equal claims.

### How old gen is swept, and why it is the bound that matters

The old generation is scored and swept as part of each cut, and entries —
summaries and quotes — that have gone cold are archived into the permanent
generation and removed. Without this a turn's journey would end at old gen: old
gen would only ever grow, the assembled context would gain one summary per aged
run forever, and old gen would become the leak that aging the young generation
exists to prevent. Candidates include entries produced earlier in the same pass;
one that is already cold is archived straight away and never enters old gen.

**Archiving reads what an entry says, not how it is shown.** An entry opens with a
marker telling the model what it is — "[Quoted exactly] (turn 2, user)",
"[Compacted … turns 13–16]" — and extraction parses the entry's text without it;
parsed with it, the marker read as the first words of the turn, the fact in that
sentence was lost, and a short stated decision fell out of the prompt as a raw
excerpt. The facts an archived entry yields carry every turn it covered, so a
reader can still trace turn 15 after its summary is gone.

**Two bands, not three.** A summary is kept or archived. The *re-compact into a
longer-horizon summary* band described under Alternatives Considered is
deliberately not built: summarising a summary compounds loss, because each pass
through a model drops detail and a second pass drops detail about detail that can
no longer be inspected. It also costs a model call per eviction where extraction
costs none. Deferred rather than rejected — a benchmark showing that re-compaction
retains more than extraction does would reopen it.

**What eviction buys is prompt position, not storage.** An old-generation summary
is in every prompt; its extracted facts are in a prompt only if they win a place in
the front's fact budget at the next cut. So even a summary whose extraction finds
nothing — kept verbatim as a `RAW` entry, at the same size it was — stops
occupying the context unconditionally, at the cost that a `RAW` entry ranks last
for that budget. Archiving happens before removal, so a failed archive leaves the
summary where it was: unlike a young turn, an evicted summary has no original to
fall back on.

**Scored against the assembled context, not against old gen alone.** Recency is
positional, so a summary's age is its distance from the newest real turn. Scoring
summaries only among themselves would make the oldest survivor always look recent,
and the coldest summary would be permanently safe — a pass that appears to work
while bounding nothing.

**No separate token budget, and this was settled by measurement rather than by
argument.** Summaries carry the first turn index of their run, so decay already
applies to them. Over 200 synthetic turns against a 700-token window: unswept, old
gen reached 72 summaries and 2226 tokens and was still climbing linearly, with the
compression ratio *worsening* from 0.20x to 0.51x as the conversation went on.
Swept, old gen held at 9 summaries and the assembled context held at 537 tokens —
constant, independent of conversation length — with the ratio improving from 0.61x
to 0.12x. A token budget would be a second mechanism able to disagree with the
score, and the score-driven bound is sufficient. What would change this: a decay
rate or threshold setting where the bound exists but arrives too slowly to keep
the context inside the window.

That last result is the difference between compressing context and *bounding* it,
and it is the claim the whole design rests on. Sweeping old gen is also only worth
doing because injection exists: the two were always one idea, since moving a cold
summary into the permanent generation is an improvement only if the permanent
generation reaches the prompt. Before injection it would have been a deletion.

---

## Why Permanent Generation Exists (Why Summaries Alone Aren't Enough)

### The Unbounded Growth Problem

Summaries compress ~5x but still grow linearly with conversation length:
- 100 turns → 20 summary blocks × 80 tokens = 1,600 tokens of summaries
- 500 turns → 100 summary blocks × 80 tokens = 8,000 tokens of summaries
- 1000 turns → 200 summary blocks × 80 tokens = 16,000 tokens of summaries

Eventually summaries themselves overflow the context window. The only option then is to
summarize the summaries — lossy compression of lossy compression. Quality degrades rapidly.

### The Permanent Gen Solution

Permanent gen stores **deduplicated facts**. Growth is bounded by the number of unique
decisions in the conversation — not by the number of turns.

500 turns about a tech stack might produce **50 unique facts** × ~10 tokens = 500 tokens.
That's constant relative to conversation length.

### The Compression Analogy

| Layer | Analogy | Growth |
|-------|---------|--------|
| Young gen (verbatim) | Raw 4K footage | Fixed (last N) |
| Old gen (summaries) | Compressed MP4 | Linear (unbounded) |
| Permanent gen (facts) | The script of the movie | Sublinear (bounded by unique knowledge) |

You can't keep accumulating MP4s forever. But the script stays small.

---

## Knowledge Contradiction Handling

When the user changes their mind, the system must update — not accumulate conflicting entries.

```
Turn 5:  "Let's use PostgreSQL"
         → KnowledgeEntry(content="Database is PostgreSQL", message_turn=5, status=ACTIVE)

Turn 15: "Actually, switch to MySQL"
         → KnowledgeEntry(content="Database is MySQL", message_turn=15, status=ACTIVE)
         → Turn 5 entry replaced with copy: status=SUPERSEDED
```

Rules:
- Superseded entries are NOT deleted — history is preserved
- Only ACTIVE entries are injected into context on retrieval
- The supersession chain is queryable (for the visualizer's Decision Survival Map)
- Contradiction detection uses topic_label matching, not just string equality

---

## Hallucination Risk Assessment

Generational memory introduces three hallucination vectors:

### 1. Lossy Summarization
Compressing "We compared PostgreSQL and MySQL for 3 turns, considering ACID compliance, hosting
costs, and team familiarity — chose PostgreSQL because the team knows it" into a short summary
can lose the *reasoning*. The LLM might hallucinate a different justification later.

**Mitigation:** Summaries preserve key decisions AND reasoning. Permanent gen stores decision +
justification as separate fields.

### 2. Knowledge Extraction Errors
If the extractor misparses "we decided NOT to use Redis" as KnowledgeEntry("Using Redis"),
the permanent gen injects a false entry forever.

**Mitigation:** Conservative extraction (high-confidence patterns only), contradiction detection
(user corrections supersede bad extractions), and the Context Visualizer lets humans see and
correct stored entries.

**What "conservative" means concretely, because it is the load-bearing word.** The
extractor refuses rather than guesses, and refusing costs nothing: a turn it
declines to parse is archived verbatim as a `RAW` entry, so the text survives and
nothing is claimed about it. A *mislabelled* entry is strictly worse than no entry,
and for a reason that is easy to miss — contradiction detection matches on the
topic label, so a label that is really a sentence fragment can never be
contradicted by anything and stays active permanently. A bad label does not just
look untidy; it disables the mechanism that was supposed to correct it.

The rules follow, and each one trades recall for precision deliberately:

- **A subject is one to four words.** Counting words, not characters and not
  "whatever precedes the verb". Without a bound, a topic grows until it happens to
  hit a full stop.
- **A declarative subject sits at the start of its sentence.** A matching verb
  further in belongs to a subordinate clause, and what precedes it is a fragment
  rather than a subject. This is the rule that does the most work.
- **Matching is per sentence, not per turn.** Otherwise a verb anywhere in a long
  turn matches, and a capture anchored to "the start" reaches back to the start of
  the whole turn. It also makes the end-of-input anchors mean end-of-*sentence*,
  which is what they were always written for.
- **A subject made only of pronouns, articles or interrogatives is rejected.** "it
  is fine" and "what are the constraints" state nothing retrievable later. The
  rejection is what sends the turn to a verbatim archive instead.
- **Openers and labels are not part of the sentence.** "OK,", "So," and a short
  label before a colon ("Decision:", "Change of plan:") are removed before the
  start-of-sentence rule applies; otherwise "OK, decision: the database is
  postgres" states nothing the extractor can read. The openers are a fixed list
  rather than "any leading clause", because a qualifier is not an opener: "In
  production, the database is postgres" is refused rather than filed without its
  qualifier.
- **Closing punctuation is removed before matching.** A topic slot at the end of a
  pattern ("…for the database") must meet the end of the sentence; with the full
  stop still there it never did, and every such decision was filed as general.
- **A question claims nothing.** "The database is postgres, right?" asks for
  confirmation; filed as a fact, it would supersede the answer. It is kept as text.
- **A trailing topic names one thing.** In "use Postgres for the database, and
  Redis for the cache" the slot catches "cache" for a decision mostly about the
  database, and a later cache decision would retire it. When the content before
  the slot lists several things, the decision is filed as general instead.

**What extraction cannot read is kept beside what it can.** A turn is rarely one
statement. Archived as only its facts, "the database is postgres; the API must
answer p99 under 200 ms" kept the database and lost the requirement — the loss the
verbatim fallback exists to prevent, hidden because the turn still showed a fact.
The sentences no entry came from are kept as a `RAW` entry beside the facts;
content-free ones ("OK.", "Thanks!") are not.

Labels are also normalised — lowercased, internal whitespace collapsed, a leading
article or possessive dropped — because contradiction detection compares them
exactly: "the Database is X" followed by "the database is Y", or "we decided to use
X for the database" followed by "the database is Y", has to be one topic with two
values rather than two unrelated facts.

The known remaining weakness is a subject that is a quantity or a long noun
phrase: "forty thousand a second is above what a single writer handles" yields a
real fact under an odd label. That is where a learned extractor earns its place;
pattern matching cannot tell a subject from a measurement.

### 3. Decontextualized Knowledge
"If we were building in Go, we'd use PostgreSQL" is hypothetical. Stripped of context, it
becomes false certainty: KnowledgeEntry("Database is PostgreSQL").

**Mitigation:** Extraction patterns only fire on decision-indicating language ("let's use",
"we'll go with", "I decided") — not hypotheticals ("if we", "we could", "maybe").
Source turn number always preserved for human traceability.

### The Epistemic Transparency Defense

All three risks are why the Context Visualizer exists. The system cannot guarantee zero
hallucination from compression — but it can make its internal state **visible** so the
human can exercise judgment. This is the project's core philosophy: the AI provides
transparency, the human provides judgment.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Score-driven classification, not position-driven | Generations are assigned by relevance score, not by turn index. A high-value old turn stays in young gen. Last N is an override safety net, not the primary mechanism. This prevents the sliding-window failure mode. |
| Three generations (not two, not four) | Maps naturally to: full detail / compressed / extracted knowledge. Fewer = information loss. More = unnecessary complexity for the same guarantees. |
| Heuristic extraction before LLM-based | Free, deterministic, testable. LLM-based extraction is more accurate and can be added behind the same interface. |
| Supersede entries, don't delete | Deletion destroys history. Supersession preserves the decision chain for audit and visualization. |
| Choose facts for the front at the cut, by type then recency | Injecting everything wastes tokens and grows with the session. Choosing per question would change the prompt on every request, breaking the provider's cache and the model's thinking blocks, so facts are chosen when a cut makes the front: decisions, then facts, then preferences, raw excerpts last. Accepted cost: a stored fact can miss the budget and not be sent, which the dashboard reports. |
| ARCHIVE always leaves a trace | A turn is run through the extractor before it leaves active context; what extraction does not read — the whole turn, or the sentences of it no entry came from — is preserved **verbatim** as a `RAW` knowledge entry. Extraction recall is imperfect, so the verbatim fallback is what makes the no-information-loss guarantee actually hold. |
| Young gen is NOT purely "last N" | Last N is a floor guarantee (those are always KEEP). But any turn scoring above keep_threshold also stays in young gen regardless of age. |
| Aging happens in `gc.update` (cold path), not `gc.collect` | Compaction and extraction are the expensive work; doing them per request would add latency to every forwarded call. The ager runs after the response; the hot-path collect only assembles what the ager already produced. |
| State-based aging (act on current classification), not a transition diff | Once an aged-out turn is removed from the young generation it can never be re-swept, so remembering its prior classification buys nothing. The observer (health monitor) needs the across-pass diff; the actor (`gc.update`) does not. |
| One-directional aging (no revival to verbatim) | Reviving a cooled turn would revive its lossy summary, not the original, and re-introduce the memory bloat compression exists to avoid. Relevance-on-return is served by the summary staying in context plus fact injection. |

---

## Alternatives Considered

The three-generation model was chosen against several simpler designs, each
rejected for a specific reason:

- **A single store ("one bucket of kept content").** Rejected. Recent verbose
  turns, compacted summaries, and durable facts have different shelf lives. One
  store either over-retains (keeping verbose turns forever) or over-compresses
  (losing facts to summaries). Separate generations let each have its own rules.
- **Two generations — young and permanent only, no old.** Rejected. Without the
  old generation, every aging turn faces a binary choice: stay verbatim or be
  reduced to a fact. Real conversations have material that is no longer fresh but
  still useful in summary form (background, partial threads); the old generation
  is where that lives.
- **Accumulating contradictions instead of superseding.** Rejected. An
  accumulating store would surface both "we chose Postgres" and "we chose MySQL"
  with equal weight, forcing the model to resolve the contradiction itself — the
  opposite of the retention commitment, which is to hand the model a *coherent*
  record of current decisions.
- **Unbounded permanent-gen injection.** Rejected. A long session's permanent
  generation accumulates; injecting every weakly-relevant entry would re-bloat the
  very context the cleanup just trimmed. Injection is bounded — entries below a
  minimum relevance are dropped and at most a capped number are injected, ranked
  by relevance — which is the contract that makes *never make things worse*
  implementable.
- **Old gen with no eviction.** Rejected. Without a token budget, old gen grows
  with the session, eventually making every collect slower and the cleaned
  context larger than the original. Old gen is itself swept: low-relevance
  summaries are archived (facts extracted first), borderline ones re-compacted
  into longer-horizon summaries.
- **Extract knowledge only on ARCHIVE.** Rejected. A decision made early may sit
  in a turn still classified KEEP. If that turn later ages out, the decision must
  already be in permanent gen — waiting for an explicit ARCHIVE risks losing it if
  the turn is summarised into old gen first. Extraction also fires as a turn ages
  out of young gen, closing that gap.

Supersession itself carries an accepted cost: it treats the *latest* contradicting
entry as current, which is right most of the time but not always — a user can
misspeak in a way the extractor reads as a real change. The system accepts being
occasionally wrong about which fact is current rather than interrupting the user
to disambiguate.

## Who Owns the Generations

**Each session owns its three generations; nothing else holds them.** A
session's young turns, old-gen summaries, and permanent-gen facts live together
in one immutable snapshot on its `SessionState` (see
[session/overview.md](../session/overview.md)). `GenerationalMemory` is a
stateless producer shared by every session: it *summarises* a run of turns and
*extracts* the facts from a turn, and stores neither result. The aging pass
collects what it produced into a delta and commits it to the session's snapshot.

This is what keeps one conversation's memory out of another's prompt. When the
generations were held inside one shared `GenerationalMemory`, a proxy serving two
users injected the first user's facts into the second user's prompt, presented
as the second user's own memory.

Supersession is part of filing, not of storage: when a fact is filed, every
active fact on the same topic in *that session* is marked superseded. Two labels
are shared rather than topics, and entries under them neither supersede nor get
superseded: `RAW` excerpts, which claim nothing, and "general", the label of every
decision that names no topic. Matched as a topic, "we decided to ship on Friday"
retired every earlier topic-less decision, however unrelated. Superseded
facts are kept, because the record of what a conversation contradicted is part
of what the transparency layer reports; only active facts reach the prompt.

## Events Emitted (for Context Visualizer)

| Event | When | Data |
|-------|------|------|
| `MESSAGE_PROMOTED_TO_OLD_GEN` | A run of cooled turns was summarised into old gen | session id, source turn indices, tokens before/after |
| `KNOWLEDGE_ENTRY_ADDED` | A fact was filed in permanent gen | session id, the entry |
| `KNOWLEDGE_ENTRY_SUPERSEDED` | A newly filed fact contradicted active ones on its topic | session id, new entry, superseded entries |
| `MESSAGE_ARCHIVED` | A turn or summary was archived after its facts were extracted | session id, message, extracted entries, source generation |

All four are emitted **after** the aging pass commits, so each describes a change
that is now in the session's state, and a pass that is discarded announces
nothing. Joining the young generation needs no event: young-generation membership
is just the turns in the session's snapshot.

These events feed the health monitor's transitions timeline. Where each turn and
fact stands *now* is read from the session's state rather than from events — see
the knowledge state in [visibility/overview.md](../visibility/overview.md).

---

## Where It Fits in the Pipeline (hot path vs cold path)

Aging and assembly are split across the two engine entry points, so the
per-request hot path stays cheap and never waits for aging.

```
COLD PATH · gc.age(session_id)                        runs after the response
═════════════════════════════════════════════════════════════════════════════
  the session's verbatim turns
                    │
                    ▼  (one pass per session at a time)
             RelevanceScorer ──► Sweeper      score + classify each verbatim turn
                    │
                    ▼  find the latest cut point: keeps the newest turns, splits no tool round
          batch = the turns before it          worth a cut? (compactable tokens ≥ min_cut_tokens)
                    │
       ┌────────────┼─────────────────────────────┐
       ▼            ▼                              ▼
     KEEP        COMPACT                        ARCHIVE
     quoted      runs summarised                facts extracted
     verbatim    (a model call)                 (KnowledgeExtractor)
                    │
                    └──► delta ──► smaller than before? ──► commit onto the CURRENT snapshot


HOT PATH · gc.collect(session_id)                     runs before forwarding
═════════════════════════════════════════════════════════════════════════════
  read the session's snapshot once — no lock, no scoring, no compaction:

        [ FRONT: facts + summaries + quotes ]   (only after a cut; frozen between cuts)
      + [ TAIL: later turns, verbatim ]
                                      │
                                      ▼
                          context ──► LLM
```

**`gc.age` (cold path) — the ager.** Runs after a response, off the request's
critical path, on the turns recorded so far. It decides whether a cut is worth
making and, if so, plans it against the current snapshot and commits it in one
step onto whatever the snapshot is by then, so turns recorded during a slow
compaction are kept. When a summary or an extraction fails, the cut retreats to
just before it.

**`gc.collect` (hot path) — the assembler.** Runs before the request is
forwarded. It reads the session's current snapshot and returns the front and the
tail. It never waits for an aging pass: if one is in flight, it returns the
context from before that pass, which is complete and only less compressed.

Two properties matter here:

- **State-based aging, not a transition diff.** `gc.update` acts on each turn's
  *current* classification and removes the acted-on turn. Because a turn that
  leaves the young generation can never be re-swept, there is nothing to guard
  against by remembering its previous classification. The health monitor, which
  *observes* aging without moving anything, is the component that needs the
  across-pass diff.
- **One-directional aging.** Turns age down the tiers (young → old → permanent)
  and never revive to verbatim. A topic that stays relevant keeps its turns
  scoring high, so they never leave the young generation; a topic that cooled and
  returns is served by its old-gen summary (always assembled in) plus fresh young
  turns. Nothing is lost — it is progressively compressed.