# Scoring Pipeline — Overview

## Why Does Scoring Exist?

LLM context windows are finite. A 50-turn conversation can easily exceed 128K tokens — and when it does, older context gets silently dropped. The model doesn't tell you what it forgot. It just starts hallucinating or contradicting earlier decisions.

**Scoring solves this**: instead of blindly dropping the oldest messages, we score every message for relevance and make intelligent decisions about what to keep, compress, or remove. The goal isn't just saving tokens — it's preserving the information that matters most while being transparent about what's being lost.

This is the **Mark** phase of the Mark-Sweep-Compact pipeline (inspired by how garbage collectors work in programming language runtimes like Java's JVM or Go's GC).

```
                         ┌──────────────────────────────────────────┐
                         │           SCORING PIPELINE               │
                         │           (The "Mark" Phase)             │
                         │                                          │
  conversation ─────────►│  For each message:                       │
  (list of messages)     │                                          │
                         │    ┌─────────────┐                       │
                         │    │  Recency     │──► How old is it?    │
                         │    ├─────────────┤                       │
                         │    │  Similarity  │──► Related to query? │
                         │    ├─────────────┤                       │
                         │    │  Density     │──► Info-rich or fluff?│
                         │    ├─────────────┤                       │
                         │    │  Decision    │──► Contains decisions?│
                         │    ├─────────────┤                       │
                         │    │  Reference   │──► Referenced later?  │
                         │    └─────────────┘                       │
                         │          │                               │
                         │          ▼                               │
                         │    weighted_sum ──► combined score [0,1] │
                         │                                          │
                         └──────────────────────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  SWEEPER decides: │
                              │  keep / compress  │
                              │  / remove         │
                              └──────────────────┘
```

## The Five Dimensions of Relevance

A single "relevance score" is too blunt — a message can be old but contain a critical decision, or recent but completely empty ("ok", "thanks"). We decompose relevance into five independent signals, each measuring a different aspect:

| Scorer | Question It Answers | Weight | Status |
|--------|-------------------|--------|--------|
| [Recency](recency-scorer.md) | How recent is this message? | 0.30 | ✅ Built |
| [Similarity](similarity-scorer.md) | How related is it to the current query? | 0.30 | ✅ Built |
| [Density](density-scorer.md) | How information-rich is it? | 0.15 | ✅ Built |
| [Decision](decision-scorer.md) | Does it contain decisions or preferences? | 0.15 | ✅ Built |
| [Reference](reference-scorer.md) | Is it referenced by later messages? | 0.10 | ✅ Built |

### Why These Weights?

Recency and similarity get the highest weight (0.30 each) because they're the strongest general-purpose signals — recent messages and messages related to the current topic are almost always relevant.

Decision and density get moderate weight (0.15 each) — they capture content quality rather than temporal/topical relevance.

Reference gets the lowest weight (0.10) because it's a supporting signal — useful but not sufficient on its own.

These weights are configurable via `GCConfig` and will be auto-tuned in later phases.

### Combined Score Formula

```
combined_score = 0.30 × recency
              + 0.30 × similarity
              + 0.15 × density
              + 0.15 × decision
              + 0.10 × reference
```

Every scorer outputs a normalized score in **[0.0, 1.0]**, so the combined score is also in [0.0, 1.0].

## Score → Action Mapping

The combined score feeds into the Sweeper, which classifies each message:

```
Score Range      Action           Generation
─────────────    ──────────────   ──────────────────────────────
0.7 – 1.0        KEEP             Young Gen — full detail preserved
0.3 – 0.7        COMPRESS         Old Gen — summarized to save tokens
0.0 – 0.3        REMOVE           Dead — facts extracted, then dropped
```

Override rules exist regardless of score:
- Last N turns are always kept (configurable, default 5)
- System prompt is always kept
- Messages with explicit decisions get a score boost

## Architecture: How Scorers Are Composed

All scorers inherit from `BaseScorer` (an abstract base class) and follow the **Strategy Pattern** — they share the same interface but implement different algorithms. This means:

- New scorers can be added without changing existing code
- Scorers can be swapped, composed, or disabled independently
- Each scorer is testable in isolation

```
              BaseScorer (ABC)
              ┌──────────────────────┐
              │ scorer_name          │
              │ score() → ScorerResult│ ◄── abstract method
              └──────────┬───────────┘
                         │ inherits
     ┌───────────────┬───┼───────────┬────────────────┐
     │               │   │           │                │
RecencyScorer  SimilarityScorer  DensityScorer  DecisionScorer  ReferenceScorer
(math-based)   (embedding-based) (regex-based)  (regex-based)   (regex-based)
```

Every scorer returns a `ScorerResult` — a Pydantic model with:
- `score` — the normalized relevance score [0.0, 1.0]
- `scorer_name` — which dimension this measures (a `RelevanceType` enum)
- `reason` — human-readable explanation of the score
- `signals` — raw data used to compute the score (for debugging and the visualizer)

The `signals` dict is what makes the Context Visualizer possible — it gives the developer full transparency into *why* a message got its score.

## Signals, Risk, and a Critical Design Constraint

Scorer signals feed into the Context Visualizer's risk indicators — hallucination risk, quality degradation warnings, decision survival maps. There is one non-negotiable constraint on how these risk signals are framed:

**Risk indicators must measure degradation that happens naturally — not degradation caused by GC.**

Context degradation is a fact of LLM conversations with or without LLM-GC:

- **Without LLM-GC**: The context window fills up. The API silently truncates from the top. Decisions vanish. The model's attention to old turns weakens. The user has no visibility into any of this. Hallucination risk is invisible.
- **With LLM-GC**: The same degradation is happening, but now the user can *see* it. And the GC is actively preserving what matters — decisions get score boosts, important context gets compressed instead of silently dropped, facts are extracted to permanent memory before messages are removed.

If the hallucination risk goes up *because* LLM-GC removed something, then we're not solving the problem — we are the problem. The risk heuristics must measure the natural degradation state: what percentage of known decisions are still in full context, how much of the conversation is filler vs. substance, how far back the model's effective attention reaches. LLM-GC's role is to make that degradation visible and to slow it down — not to cause it.

This connects to the project's core philosophy of **epistemic transparency**: the system doesn't claim to eliminate hallucination risk. It makes the risk visible so the human can exercise judgment.

## Design Rationale & Alternatives Considered

The "how it works" above is one point in a space of choices. The decisions that
shaped it, and the credible alternatives rejected:

**Multi-signal composition, not a single opaque score.** Folding recency,
similarity, density, decision, and reference into one black-box function would
still produce a number — but a number nobody could explain. That defeats the
project's visibility commitment: the developer could not see *why* a turn scored
the way it did, and so could not judge whether to trust the classification.
Decomposing relevance into five independently-published signals is what makes the
score auditable.

**Regex heuristics for the linguistic signals, not a learned model — yet.**
Density, decision, and reference detection ask whether text *means* something
specific; that can be regex, hand-written heuristics, or a learned model. A
learned scorer was rejected for the initial build for a blunt reason: no labelled
data exists yet, and producing it honestly requires running the system on real
conversations first. A learned-but-untrained scorer would be either dishonest
(not actually learning) or pointless (random scores). The regex scorers are the
honest baseline, and the `BaseScorer` interface is what lets a learned
replacement swap in later without touching the pipeline.

**Embeddings for similarity, not heuristics alone.** Pure recency and keyword
heuristics miss semantic relevance — a turn about "the database" is still relevant
when the conversation moves to "Postgres" with no shared keyword. Embedding-based
cosine similarity captures that cheaply and runs locally with no provider
dependency.

**Conservative recall on the regex signals.** The detectors are deliberately
tuned to prefer false positives (flagging a decision that isn't there) over false
negatives (missing a real one). The cost asymmetry mirrors the quality-preservation
principle: a wrongly-flagged turn costs a few tokens of extra retention; a missed
decision costs a lost fact.

**Weights normalised as ratios, not validated against a strict sum of 1.0.**
Configured weights are normalised at construction (each divided by their sum),
with a warning if the input sum is more than 1% from 1.0. This avoids two real
problems: floating-point configs equal to 1.0 by intent but bit-different in
practice (`0.1 + 0.2 = 0.30000000000000004`), and the common case of bumping one
signal without re-tuning the rest. Any non-negative configuration is valid; none
is silently rejected.

## Current Limitations & Future Improvements

**Today:** Recency uses math; density, decision, and reference use regex pattern matching. This is fast and predictable but misses nuance — regex can't understand "I think we should probably go with option B" as a decision.

**Next:** Fact extraction feeding back into scoring — messages whose extracted facts are still relevant get a score boost.

**Later, optionally:** Learned relevance scoring replaces the regex heuristics with trained models. The scorer interface stays the same — only the algorithm inside changes. This is why the Strategy pattern matters here: swapping regex for ML requires zero changes to the pipeline.

## File Map

```
src/llm_gc/
  scoring/
    __init__.py            # Public exports
    base_scorer.py         # Abstract base class — Strategy Pattern contract
    scorer_result.py       # Pydantic model for scorer output
    relevance_type.py      # Enum: RECENCY, SIMILARITY, DENSITY, DECISION, REFERENCE
    recency_scorer.py      # Exponential decay scorer
    similarity_scorer.py   # Semantic similarity scorer (sentence-transformers)
    density_scorer.py      # Information density scorer (regex)
    decision_scorer.py     # Decision phrase scorer (regex)
    reference_scorer.py    # Back-reference scorer (regex)
  config/
    constants.py           # Regex patterns and default configuration values
    gc_config.py           # GCConfig Pydantic model (thresholds, weights)
```