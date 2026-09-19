# Decision Scorer

## Why Does This Exist?

Decisions are the single most dangerous type of information to lose from a conversation. Consider:

> **Turn 8**: "Let's go with PostgreSQL for the database. JWT for auth with 7-day refresh tokens."
>
> **Turn 35**: User asks "Can you set up the authentication middleware?"

If turn 8 was garbage collected because it was "old", the model has no memory that JWT was chosen. It might suggest session-based auth, or ask "what auth approach would you like?" — wasting the user's time and breaking trust.

**Decisions are the highest-priority content to preserve.** They represent irreversible choices that downstream work depends on. The decision scorer exists to ensure these messages get a relevance boost so the sweeper never discards them before their facts are extracted to permanent memory.

This is directly tied to LLM-GC's core philosophy of **epistemic transparency** — the system must know what it knows, and decisions are the foundation of that knowledge.

## How It Works

The decision scorer counts **decision-indicator phrases** in the message text using regex pattern matching, then normalizes against a configurable threshold.

### The Formula

```
decision_score = min(decision_signal_count / max_decision, 1.0)

where:
  decision_signal_count = number of decision phrases found (via regex, case-insensitive)
  max_decision          = normalization threshold (default: 3)
```

### What Counts as a Decision Phrase?

The regex pattern (`DECISION_SCORER_PATTERN` in `constants.py`) detects phrases across three categories:

```
Category                  Phrases Detected
────────────────────      ──────────────────────────────────────────────────
Explicit decisions        "decided to", "let's go with", "let's use",
                          "we'll go with", "we'll use", "go with",
                          "the plan is", "chose", "settled on", "we agreed"

Conclusions               "concluded", "concluded that", "in conclusion",
                          "the answer is", "the solution is", "final choice",
                          "to summarize"

Commitments               "I will use", "we should use", "going to use",
                          "going with", "stick with", "switching to"

State indicators          "keep", "delete", "summarize", "uncertain"
```

All matching is **case-insensitive** — "Let's GO with" and "let's go with" both match.

### Why Count-Based (Not Binary)?

A message with three decision phrases ("Let's use Python. We'll go with FastAPI. The plan is to deploy on AWS.") is a more critical decision message than one with a single phrase ("let's use Python"). The count captures this intensity — more decision phrases means more decisions at stake.

`max_decision = 3` means: three or more decision phrases in a single message scores 1.0 (maximum importance). This threshold was chosen because most real decision messages contain 1–3 decision phrases. Messages with more than 3 are rare but automatically capped at 1.0.

## Worked Example

**Message**: `"Let's go with option A. I've decided to use Python. The final choice is REST over GraphQL."`

Phrases found:
1. `Let's go with` — explicit decision
2. `decided to` — explicit decision
3. `final choice` — conclusion

Decision signal count: **3**
Score: min(3 / 3, 1.0) = **1.00**

---

**Message**: `"I think we should use Redis for caching."`

Phrases found:
1. `we should use` — commitment

Decision signal count: **1**
Score: min(1 / 3, 1.0) = **0.33**

---

**Message**: `"Hello, how are you doing today?"`

Phrases found: **0**
Score: **0.00**

## How It Interacts With Other Scorers

The decision scorer's weight (0.15) might seem low, but it works by **boosting** messages that other scorers might undervalue:

```
Example: An old decision message

  Recency score:   0.15  (old → low recency)
  Density score:   0.70  (contains specific terms)
  Decision score:  1.00  (contains decisions!)
  Reference score: 0.33  (referenced once later)

  Combined: 0.30(0.15) + 0.15(0.70) + 0.15(1.00) + 0.10(0.33) = 0.33

  Without decision scorer:
  Combined: 0.30(0.15) + 0.15(0.70) + 0.10(0.33) = 0.18  → REMOVED

  With decision scorer:
  Combined: 0.33  → COMPRESSED (kept as summary, not deleted)
```

The decision scorer prevents old decision messages from falling into the "remove" zone. Decisions are extracted to permanent memory when a message is archived, but the scorer is what keeps them in full context until then.

## Design Decisions

**Why regex instead of an LLM call to detect decisions?**
LLM calls are expensive (~100ms+), require API access, and add a dependency on external services for a core scoring function. Regex is ~microseconds, deterministic, and works offline. The trade-off: regex misses nuanced decisions like "I'm leaning towards option B" or "Probably PostgreSQL." A trained classifier would address this — the scorer interface stays identical.

**Why these specific phrases?**
They were chosen to be high-precision (few false positives) rather than high-recall (catching every possible decision). "let's go with" is almost always a decision. "I think" is often just hedging. We'd rather miss some soft decisions than flag every opinion as a decision. The phrase list can be expanded as benchmarking surfaces false negatives.

**Why `max_decision = 3` as default?**
Empirically, most decision-bearing messages in real conversations contain 1–3 decision phrases. Setting the threshold at 3 means a single decision phrase still gets a meaningful score (0.33), while messages with 3+ hit the maximum. This was a judgment call, still to be validated against real conversation data.

**Why is "uncertain" in the pattern?**
Expressions of uncertainty ("I'm uncertain about the database choice") are meta-decisions — they signal that a decision is pending or unresolved. These are worth preserving because they provide context for later decisions and prevent the model from assuming certainty where none exists.

## Signals Emitted

```python
signals = {
    "decision_signal_count": 3,    # Number of decision phrases matched
    "decision_score": 1.0          # Final normalized score
}
```

In the Context Visualizer, these signals will power the **decision survival map** — a view showing which decisions have been made, which are still in context, and which are at risk of being garbage collected.

## Limitations & Future Improvements

- **Pattern matching misses implicit decisions**: "I'm going with B" → detected. "B seems right, let's do that" → partially detected ("let's" without "go with" / "use"). "Yeah, option B" → missed entirely. Natural language decisions are far more varied than any regex can capture.
- **No decision extraction**: The scorer detects that a message *contains* decisions but doesn't extract *what* was decided. That is the fact extractor's job, on the path into permanent memory.
- **No negative decision detection**: "Don't use MongoDB" is as important as "Use PostgreSQL" — negative decisions constrain future choices. The current pattern partially handles this ("delete") but misses most negations. Future improvement.
- **No decision conflict detection**: If turn 5 says "use PostgreSQL" and turn 20 says "actually, switch to MongoDB", the scorer treats both as equally important decisions. Detecting that turn 20 supersedes turn 5 requires understanding decision relationships, which the scorer does not model.

## File Location

- Implementation: `src/llm_gc/scoring/decision_scorer.py`
- Regex pattern: `src/llm_gc/config/constants.py` → `DECISION_SCORER_PATTERN`
- Tests: `tests/scoring/test_decision_scorer.py`