# Scoring: How Much Each Span Matters Now

## Why scoring exists

Every span in the old generation is shown at a detail level, and the level comes from its score. The score answers one question, again at every collection: **how much does this part of the conversation matter to the conversation as it stands now?** Because it is asked again, the answer can change — a span the user returns to scores higher than it did when it was made.

## How a span is scored

Four signals, in two groups:

| Signal | What it asks | Computed by | Weight |
|---|---|---|---|
| **Recency** | How recently was this span said? | `SpanScorer` | 0.6 to liveness: the larger of these two |
| **Relevance now** | Does what the user is discussing now reach it? | the `Relevance` seam (`LexicalRelevance` by default) | (the same 0.6) |
| **[Decision](decision-scorer.md)** | Does it state or settle a decision? | `DecisionScorer` | 0.2 |
| **[Density](density-scorer.md)** | Does it carry names, numbers, code, links? | `DensityScorer` | 0.2 |

```
                 ┌───────────────────────────────────────────────────────────────┐
  every span ───►│ LIVENESS  =  the larger of                                    │
  (its turns)    │                recency      e^(−0.05 × turns since its last turn)
                 │                relevance    how much of what the user is       │
  "now" ────────►│                             discussing now this span talks about
  (the user's    │                                                               │
   turns in      │ IMPORTANCE  = decision      decision phrases it states        │
   the tail)     │             + density       names, numbers, code it carries   │
                 │                                                               │
                 │ score = (0.6 × liveness + 0.2 × decision + 0.2 × density) / 1.0│
                 └───────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                        the planner: score → detail level
                        (see generational-memory/detail-levels.md)
```

### Liveness: recent, or reached by what is discussed now

A span is **live** — part of the conversation as it stands — if it is recent **or** if what the user is discussing reaches it. Liveness is the larger of the two, not their sum.

```
                         recency   relevance   liveness
 just condensed, unrelated  0.78       0.00       0.78   ← recent: stays summarised
 old, nobody mentions it    0.02       0.00       0.02   ← quiet: drifts to archived
 old, the user asks about   0.02       0.80       0.80   ← reached: rises again
```

- **Recency** decays slowly: a span loses half its recency every fourteen turns, measured from its last turn. Slow on purpose — the score should archive what has gone quiet, and leave bounding the front to the budget.
- **Relevance now** compares the span with **"now"**: the turns the user wrote in the tail (at a collection), or the user's newest message (at a recall). Not the assistant's replies — they restate and expand, and would make every span look relevant; not tool results, which are output rather than a subject.

### Importance: what a span states

Two measures lift spans that carry decisions and specifics:

- **[Decision](decision-scorer.md)** — phrases that state or settle a decision ("we decided", "let's go with", "the plan is").
- **[Density](density-scorer.md)** — the share of names, numbers, code and links in the text.

Importance alone tops out at 0.4, below the keep line (0.7): a span is never kept word for word just for stating a decision, unless it is also live.

### Every level stays reachable, at any age

Recency falls to zero with age; relevance does not. So the best score a span of any age can reach is 1.0, and every level — kept included — can be reached by a span the conversation returns to. A test computes this best score from the shipped weights and fails if a level is ever out of reach.

### Relevance now, word for word

Relevance is a seam (`Relevance`); the shipped implementation compares words:

```
 for each word the user said now (stopwords dropped):
     weight  = how rare the word is across this conversation's spans
               (a word every span uses — "service", "design" — says little)
     credit  = how often this span uses it, with diminishing returns
               1 mention → ½ · 2 → ⅔ · 10 → about 0.9

 relevance(span) = Σ weight × credit  /  Σ weight          (between 0 and 1)
```

The span that *discusses* a subject outranks one that *names* it in passing, and a weak match stays weak. Measured on the decision-survival check's ticketing conversation:

| What the user asked | Best span | Next best | A generic span |
|---|---|---|---|
| "How long do we hold seats…?" | the seat-hold answer, 0.67 | 0.58 | 0.06 |
| "What did we decide about the platform fee?" | the fee decision, 0.87 | 0.84 | 0.00 |
| "How would you phase the rollout?" | none above 0.09 | — | — |

### A message that refers back

```
 "As we discussed, how long do we hold seats?"
  └──────┬───────┘ └────────────┬─────────────┘
   says THAT it refers    says WHAT: "hold", "seats"
   back — any earlier     → reaches the span that discusses seat holds,
   span could be meant      and raises it at the next collection
```

A back-reference reaches its span through the words that name the subject, not through "as we discussed", which fits every earlier span alike. Words that only refer back — "mentioned", "earlier", "discussed", "again" — are not counted at all: each would be a word asked about that no span matches, and measured, they cut the fee span's score for "as I mentioned earlier, what was the fee again?" from about 0.5 to 0.13. Between collections the front does not change, so until the next one the model sees that span at its current level, and the front's header tells it to say so rather than guess.

### Known misses

- **Paraphrase.** "Datastore" does not find "database". An embedding implementation of the same seam fixes it without changing anything else.
- **A reference that names nothing.** "Do what I said earlier" reaches no span; no word in it is about anything.

## Where the score is used

```
 at every collection        score every span against the user's turns in the tail
                            → the planner re-picks every level
 for the next-collection    the same scoring, drafted spans included, with no model
 preview (dashboard)        call and no event
```

The default engine is wired in one place, `build_gc_service`, which the proxy runs and the decision-survival check measures — so the check reports on the engine people actually run.

```
                    build_gc_service(config, summary_provider)
                                      │
        ┌─────────────────────────────┼───────────────────────────────┐
        ▼                             ▼                               ▼
   SpanScorer                    GenerationalMemory                GCConfig
   liveness 0.6                  extractor + compactor             bands, margin,
   decision 0.2, density 0.2     (LLMCompactor, or NoOp            recap budget,
   word-overlap relevance        when no provider)                 batch size
        └──────────────────────► GcService ◄────────────────────────┘
```

## Signals, risk, and a design constraint

Every score is returned with its parts — recency, relevance, decision, density, and the liveness taken from them — so a reader can see *why* a span is where it is.

**Risk indicators must measure degradation that happens naturally, not degradation caused by GC.** Without LLM-GC, the window fills, the API truncates from the top, decisions vanish, and nobody sees it. With LLM-GC, the same pressure exists, but it is visible, and the score keeps what matters now in front of the model. If risk went up *because* LLM-GC condensed something, LLM-GC would be the problem it claims to solve. Its role is to make degradation visible and to slow it, never to cause it.

## Why this shape

- **Spans scored in one batch, against a "now" the caller supplies.** Word rarity needs every span at once, and in a flat list of messages a span of twenty turns would count as one position.
- **Recency and relevance inside the span scorer; decision and density as scorers of their own.** Decision and density read one span's text and nothing else, so each is a small class behind `BaseScorer`. Recency needs the newest turn, and relevance needs how rare each word is across every span, so both are computed where all the spans are seen at once.
- **The larger of recency and relevance, not their sum.** Summed, recency weighted enough to keep recent spans summarised capped what an old span could ever score; weighted lightly, every unrelated span was archived the moment it was made.
- **A slow decay.** At the turn-level rate (0.15 per turn), measured spans were archived about seven turns after they were said, with the front nowhere near its budget; the three short check conversations archived five or six of seven spans. At 0.05 they keep their summaries while there is room.
- **Several named signals, not one opaque number.** A single black-box score could not be explained; separate signals let a person audit a span's level.
- **Regex for the linguistic signals, for now.** No labelled data exists yet to train a model honestly; the regex scorers are the baseline, and the interfaces let a learned replacement swap in without touching the pipeline.
- **Draft weights.** 0.6 / 0.2 / 0.2 and the 0.05 decay were set against the check's conversations and will be tuned on it: a saving is kept only if no answer is lost.

## Alternatives rejected

- **Scoring each message on its own, with recency weighted at 0.6, then classifying it once.** This was the earlier pipeline. Every message old enough to be collected was 5–8 turns old, so recency barely varied between them, and it capped the score of any collected message at 0.68 — under the 0.7 keep line, so quoting an important old turn could never happen. Replaced by span scoring with liveness.
- **A reference scorer counting later messages that say "earlier" or "as I mentioned".** It never knew *which* message was meant, so every older message got the same lift. Relevance to what is discussed now is the targeted version.
- **Embedding similarity in the default engine.** It needs a large model download, a cold start, and CPU-heavy work on the scoring path, for a gain not yet measured against word overlap. It belongs behind the `Relevance` seam, as an optional extra.
- **Plain word coverage.** Ties: four spans of the ticketing conversation scored 1.00 on the platform-fee question, because coverage ignores how often a span repeats a word.
- **BM25 scaled to the best-matching span.** Some span is always the best, so a generic question ("how would you phase the rollout?") made one span a perfect match off a single incidental word.

## File map

```
src/llm_gc/
  scoring/
    span_scorer.py         # SpanScorer, liveness, the Relevance seam, LexicalRelevance
    base_scorer.py         # the abstract base the importance scorers implement
    decision_scorer.py     # decision phrases (regex)
    density_scorer.py      # names, numbers, code, links per token (regex)
    scorer_result.py       # one measure's score, reason and signals
    relevance_type.py      # the measures: recency, similarity, decision, density
  config/
    constants.py           # the regex patterns
```
