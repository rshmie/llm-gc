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
  compact_threshold <= score < keep_threshold → COMPACT   → Old Generation (after summarization)
  score < compact_threshold (default 0.3)   → ARCHIVE   → Permanent Generation (after extraction)
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
│  Contains:   Turns classified COMPACT (score between compact_       │
│              threshold and keep_threshold), compressed into summaries│
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
| Old | compact_threshold <= score < keep_threshold | Medium | Linear with conversation length | Partially — entries eventually archive |
| Permanent | score < compact_threshold (after extraction) | Low | Sublinear (deduplicated facts) | Effectively yes — unique decisions are finite |

The key insight: **summaries grow linearly, but knowledge entries grow sublinearly.** Whether you
discussed PostgreSQL across turns 5, 15, and 45 — the permanent gen holds ONE entry. This is
what prevents unbounded context growth.

One caveat to that bound: turns the extractor cannot read are preserved **verbatim** as `RAW`
entries — the no-information-loss guarantee. Those are full-size and not deduplicated, so the
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
  ├─── compact_threshold <= score < keep_threshold ──────→ COMPACT
  │                                                              │
  │                                                              ▼
  │                                                         PROMOTION
  │                                                         (summarize with adjacent turns)
  │                                                              │
  │                                                              ▼
  │                                                         OLD GEN
  │                                                         (compressed summary)
  │
  └─── score < compact_threshold ────────────────────────→ ARCHIVE
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
(score dropped below compact_threshold). This is the natural aging process — driven by
relevance, not just position.

---

## How Context Is Assembled (on gc.collect)

When the GC builds the optimized context to send to the LLM:

```
┌──────────────────────────────────────────────────────┐
│ 1. System prompt (unchanged)                          │
│                                                      │
│ 2. Relevant permanent gen facts (injected as context)│
│    "[Memory: Database is MySQL, decided turn 15]"    │
│    "[Memory: Auth uses JWT with 7-day refresh]"      │
│    Only facts RELEVANT to current query (similarity) │
│                                                      │
│ 3. Old gen summaries (compressed turns)              │
│    "[Summary turns 6-12: Implemented JWT auth...]"   │
│                                                      │
│ 4. Young gen turns (recent, full verbatim)           │
│    Turn 18, Turn 19, Turn 20                         │
│                                                      │
│ 5. Current query (unchanged)                         │
└──────────────────────────────────────────────────────┘
```

This resembles RAG — but instead of retrieving from an external knowledge base, it retrieves
from facts extracted from **this conversation's own history**. Same mechanism (embedding
similarity for retrieval), different source.

### How step 2 chooses, and what it costs

Retrieval, not a dump. Permanent generation accumulates for the life of a session,
so injecting all of it would recreate, somewhere new, the unbounded growth that
archiving exists to stop.

**Ranking sits behind an interface.** Retrieval quality is a measurable property
that belongs to a benchmark, and the component that *uses* a ranking should not
change when the ranking does. The shipped default ranks by word overlap with the
turn about to be sent, weighting a match on the fact's subject above a match
anywhere in its content. The rejected alternative was to make embedding similarity
the default: it is better at paraphrase, and it puts a model download and a cold
start on the collect path for an improvement nobody has measured yet. That is the
wrong order. The accepted cost is a real and named miss — "which datastore?" does
not match a fact stored under `database` — and a deployment that cares supplies its
own retriever rather than editing the engine.

**Facts with no overlap are excluded, not merely ranked last.** A prompt filled
with whatever happened to be stored spends tokens and invites the model to use
something unrelated to the turn. Returning fewer facts is the better failure.

**The block is capped, and the cap is what makes the system stable.** Injected
facts are real tokens in the real prompt, so they count against the context budget
like any other message — the same ruling the budget meter already makes about
old-generation summaries, for the same reason: a meter that omitted them would
under-report pressure in the reassuring direction. That creates a loop worth
naming: pressure causes archiving, archiving produces facts, facts add tokens,
tokens add pressure. Bounded, it converges, because past the cap further archiving
only ever reduces the total. Unbounded, memory grows with the session and cancels
the saving that archiving produced. Setting either budget to zero disables
injection, which is how a deployment opts out and how a benchmark measures what
injection is worth by running the same conversation both ways.

**Selection and presentation use different orderings.** Relevance decides what
survives the budget; the surviving facts are then shown in conversation order,
because a block is read top to bottom and a fact from turn 2 belongs before one
from turn 40.

**The block says what it is.** It carries a header naming it as recovered memory
and each fact carries its origin turn. Distilled facts presented as if they were
original conversation would be the same dishonesty the compacted-summary marker
exists to prevent, and the origin turn is what lets the prompt and the dashboard
agree about provenance. A `RAW` entry is labelled an excerpt rather than a parsed
fact, because extraction found nothing in that turn and calling it a fact would
overstate what is known.

**Superseded facts never reach the prompt.** Only active entries are candidates.
This is the step that makes contradiction tracking worth anything: without the
filter the model would read an old value and its correction as two equal claims.

### How old gen is swept, and why it is the bound that matters

The old generation is scored and swept on the same pressure-gated pass that ages
the young generation, and summaries that have gone cold are archived into the
permanent generation and removed. Without this a turn's journey ended at old gen:
`_old_gen` was append-only, so the assembled context grew by one summary per aged
run forever, and old gen became the leak that young gen had been fixed to avoid.

**Two bands, not three.** A summary is kept or archived. The *re-compact into a
longer-horizon summary* band described under Alternatives Considered is
deliberately not built: summarising a summary compounds loss, because each pass
through a model drops detail and a second pass drops detail about detail that can
no longer be inspected. It also costs a model call per eviction where extraction
costs none. Deferred rather than rejected — a benchmark showing that re-compaction
retains more than extraction does would reopen it.

**What eviction buys is prompt position, not storage.** An old-generation summary
is in every prompt; its extracted facts are in a prompt only when they match the
turn being sent. So even a summary whose extraction finds nothing — kept verbatim
as a `RAW` entry, at the same size it was — stops occupying the context
unconditionally. Archiving happens before removal, so a failed archive leaves the
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

Four rules follow, and each one trades recall for precision deliberately:

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

Labels are also normalised — lowercased, internal whitespace collapsed — because
contradiction detection compares them exactly, and "the Database is X" followed by
"the database is Y" has to be one topic with two values rather than two unrelated
facts.

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
| Retrieve permanent gen by similarity | Not all stored entries are relevant to every query. Injecting everything wastes tokens. Similarity retrieval = only inject what matters now. |
| ARCHIVE always leaves a trace | A turn is run through the extractor before it leaves active context; if extraction yields no entry, the turn is preserved **verbatim** as a `RAW` knowledge entry. Extraction recall is imperfect, so the verbatim fallback is what makes the no-information-loss guarantee actually hold. |
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

## Events Emitted (for Context Visualizer)

| Event | When | Data |
|-------|------|------|
| `KNOWLEDGE_ENTRY_ADDED` | New knowledge entry stored in permanent gen | entry content, source turn, knowledge type |
| `KNOWLEDGE_ENTRY_SUPERSEDED` | New entry supersedes an existing one | new entry, superseded entries |
| `MESSAGE_ARCHIVED` | A turn is archived after its knowledge is extracted | message, extracted knowledge entries |
| `MESSAGE_PROMOTED_TO_OLD_GEN` | A cooled turn is summarised into the old generation | messages compacted, tokens before/after |

Old-generation membership is now tracked inside `GenerationalMemory`: the ager
(`gc.update`) hands a cooled turn to `promote_to_old_gen`, which compacts it and
stores the summary, emitting `MESSAGE_PROMOTED_TO_OLD_GEN`.
`MESSAGE_ADDED_TO_YOUNG_GEN` remains a reserved type with no emitter — young-generation
membership is just the turns still present in the session's live message list, so it
needs no separate store or event.

These events feed the **Decision Survival Map** — showing which decisions from the conversation
are still alive, which were superseded, and which are at risk of being lost.

---

## Where It Fits in the Pipeline (hot path vs cold path)

Aging and assembly are split across the two engine entry points, so the
per-request hot path stays cheap.

```
COLD PATH · gc.update(session_id, new_turn)          runs after the response
═════════════════════════════════════════════════════════════════════════════
  new_turn ──► append to YOUNG GEN
                    │
                    ▼
             RelevanceScorer ──► Sweeper      score + classify each young turn
                    │
                    ▼      act on each turn's CURRENT classification
       ┌────────────┼─────────────────────────────┐
       ▼            ▼                              ▼
     KEEP        COMPACT                        ARCHIVE
     stays     promote_to_old_gen            archive_message
     young     → summary ─► OLD GEN           → facts ─► PERMANENT GEN
                    │                              │
                    └────────── turn leaves the YOUNG GEN ──────────┘


HOT PATH · gc.collect(session_id)                    runs before forwarding
═════════════════════════════════════════════════════════════════════════════
  read + stitch by turn_index — no scoring, no compaction:

       [ OLD GEN summaries ]  +  [ YOUNG GEN turns (verbatim) ]
                                      │
                                      ▼
                          cleaned context ──► LLM

  planned (not yet wired): inject relevant PERMANENT GEN facts;
                           reuse a pre-computed plan instead of assembling cold
```

**`gc.update` (cold path) — the ager.** Runs after a response, off the request's
critical path. It scores and sweeps the young generation, then acts on each
turn's current classification:

- **KEEP** → stays in the young generation (verbatim).
- **COMPACT** → `promote_to_old_gen` compacts it into an old-gen summary; the turn
  then leaves the young generation.
- **ARCHIVE** → `KnowledgeExtractor` pulls its facts into permanent gen; the turn
  then leaves the young generation.

Each turn is aged independently: its produce-action (compact or extract) runs
*before* the turn is removed from the young generation, so a failure leaves the
turn in place to be retried, never losing it and never leaving a summary whose
original is still live.

**`gc.collect` (hot path) — the assembler.** Runs before the request is
forwarded. It only reads and stitches: old-gen summaries + young-gen turns,
ordered by turn index. No scoring, no compaction on this path — the ager already
did that work. (Permanent-gen fact injection and pre-computed-plan reuse are
planned additions to this path, not yet wired.)

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