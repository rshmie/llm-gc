# Recency Scorer

## Why Does This Exist?

In any conversation, recent messages are almost always more relevant than older ones. If you're debugging a database error on turn 45, turn 3 where you discussed project setup is probably not useful anymore.

But a hard cutoff ("drop everything older than 20 turns") is too blunt — turn 8 might contain a critical architectural decision that's still relevant. We need a **gradual decay**: recent messages get high scores, older messages get progressively lower scores, but nothing is killed by age alone. Other scorers (decision, reference) can rescue old-but-important messages.

This is the simplest and most reliable signal in the scoring pipeline.

## How It Works

The recency scorer uses **exponential decay** — the same mathematical function used in radioactive decay, capacitor discharge, and many natural "fading" processes.

### The Formula

```
score = e^(-decay_rate × age)

where:
  age        = distance from the end of the conversation (newest = 0, second newest = 1, ...)
  decay_rate = how fast relevance fades (default: 0.1)
  e          = Euler's number (~2.718)
```

### Why Exponential Decay (Not Linear)?

Linear decay (`score = 1 - age/total`) drops at a constant rate — message 5 and message 50 lose the same amount of relevance per turn. That doesn't match reality: the difference between 1-turn-ago and 5-turns-ago is huge, but the difference between 45-turns-ago and 50-turns-ago is negligible. Both are "old."

Exponential decay captures this: it drops sharply for recent messages (where recency matters most) and flattens out for older ones (where they're all roughly "old").

```
Score
1.0 │●
    │ ●
0.8 │  ●
    │   ●
0.6 │    ●
    │     ●
0.4 │       ●
    │         ●
0.2 │            ●
    │                ●
0.0 │──────────────────────●──●──●──●──
    0    5    10   15   20   25   30   35
                    Age (turns)
```

### Decay Rate Controls Aggressiveness

The `decay_rate` parameter (default 0.1) controls how fast older messages lose relevance:

| Age | decay_rate=0.05 (gentle) | decay_rate=0.1 (default) | decay_rate=0.2 (aggressive) |
|-----|--------------------------|--------------------------|----------------------------|
| 0 | 1.000 | 1.000 | 1.000 |
| 5 | 0.779 | 0.607 | 0.368 |
| 10 | 0.607 | 0.368 | 0.135 |
| 20 | 0.368 | 0.135 | 0.018 |
| 50 | 0.082 | 0.007 | 0.000 |

- **Lower decay_rate** (0.05): Older messages retain more relevance. Better for long, coherent conversations where early context stays relevant.
- **Higher decay_rate** (0.2): Aggressive aging. Better for rapid topic-switching conversations where old context is rarely revisited.

The default of 0.1 is a balanced middle ground — a message 10 turns old scores ~0.37, which is in the "compress" zone when combined with other signals.

## Worked Example

Given a 5-message conversation:

```
Index 0: "Let's build an API"           ← age 4 → score = e^(-0.1 × 4) = 0.670
Index 1: "Use FastAPI with Pydantic"     ← age 3 → score = e^(-0.1 × 3) = 0.741
Index 2: "ok sounds good"               ← age 2 → score = e^(-0.1 × 2) = 0.819
Index 3: "Add JWT authentication"        ← age 1 → score = e^(-0.1 × 1) = 0.905
Index 4: "How do I set up routes?"       ← age 0 → score = e^(-0.1 × 0) = 1.000
```

Note: recency alone doesn't decide what to keep — "ok sounds good" (age 2) scores higher than "Let's build an API" (age 4), even though the latter is far more important. That's why we have density and decision scorers to compensate.

## Design Decisions

**Why `conversation.index(message)` instead of passing age directly?**
The scorer computes age from the message's position in the conversation list. This keeps the scorer interface uniform — every scorer receives the same `(message, conversation)` pair. The small cost of `list.index()` is negligible for conversation-sized lists (rarely more than a few hundred messages).

**Why is decay_rate a constructor parameter, not in GCConfig?**
Each scorer owns its own tunable parameters. `GCConfig` holds system-wide settings (thresholds, model name). Scorer-specific parameters live on the scorer instance — this keeps configuration scoped and makes it easy to run multiple scorer instances with different settings (useful for benchmarking).

**Why not use timestamps instead of position?**
Position-based age is simpler and doesn't require timestamps on messages. In an LLM conversation, turns happen sequentially — position *is* the meaningful measure of recency. Timestamps would add complexity without improving the signal.

## Signals Emitted

```python
signals = {
    "age": 4,              # Position from end (0 = newest)
    "decay_rate": 0.1,     # The decay rate used
    "raw_score": 0.6703    # The computed score (before any pipeline-level normalization)
}
```

These signals feed into the Context Visualizer's turn relevance heatmap.

## Limitations & Future Improvements

- **No session awareness**: Treats the entire conversation as one sequence. Session boundaries do not reset age calculations, so turn 1 of a new session is not treated as "new" when it is turn 50 overall.
- **Fixed decay rate**: The same rate applies to all conversations. Benchmarking across conversation types would show whether one rate fits all; a learned per-conversation decay rate is a further step.
- **No topic-aware aging**: A message about the current topic should age slower than one about an abandoned topic. The similarity scorer partially addresses this, but a combined recency-similarity signal could be more powerful.

## File Location

- Implementation: `src/llm_gc/scoring/recency_scorer.py`
- Tests: `tests/scoring/test_recency_scorer.py`