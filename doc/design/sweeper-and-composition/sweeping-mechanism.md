# Sweeper — Classification Mechanism

## Why Does This Exist?

The Scoring Pipeline produces a relevance score for every message - a number between 0 and 1. But a number alone doesn't tell the system **what to do** with that message. The Sweeper is the decision layer: it takes scores and assigns actionable classifications.

Without the Sweeper, the pipeline would know "message 5 scored 0.42" but have no mechanism to say "message 5 should be compressed." The Sweeper bridges the gap between measurement and action.

This is the **Sweep** phase of the Mark-Sweep-Compact pipeline. In Java GC terms: marking identifies liveness, sweeping classifies objects for collection. Here, scoring identifies relevance, sweeping classifies messages for compression or retention.

## How It Works

### Architecture: Strategy Pattern + Orchestrator

The Sweeper uses a two-layer design:

```
                    ┌────────────────────────────────────────────┐
                    │              SWEEPER                        │
                    │           (Orchestrator)                    │
                    │                                            │
  conversation ───►│  1. Apply override rules (universal)        │
  + scores         │     • System prompt → always KEEP          │
                    │     • Last N turns → always KEEP           │
                    │                                            │
                    │  2. Delegate remaining to strategy          │
                    │     ┌──────────────────────────────┐       │
                    │     │   ThresholdSweepStrategy      │       │
                    │     │   (score + token length →     │       │
                    │     │    KEEP or COMPACT)           │       │
                    │     └──────────────────────────────┘       │
                    │                                            │
                    │  3. Aggregate: counts, token totals, time  │
                    │  4. Emit SWEEP_COMPLETED event             │
                    │                                            │
                    └───────────────────┬────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │    SweepResult    │
                              │  (entries + metrics)│
                              └──────────────────┘
```

**Why two layers?**

Override rules (system prompt, last N turns) are universal — they apply regardless of which strategy is active. If each strategy implemented overrides independently, you'd duplicate logic and risk one forgetting. The Sweeper owns universal rules; the strategy owns classification logic.

This follows the same principle as a bouncer at a door: the bouncer checks the VIP list (overrides) first, then anyone not on the list goes through normal admission (strategy).

### The Strategy Pattern

The Sweeper delegates classification to a pluggable strategy. Today there's one (`ThresholdSweepStrategy`). Tomorrow there could be an ML-based strategy. The interface stays the same.

```
BaseSweepStrategy (ABC)
  │
  ├── ThresholdSweepStrategy  (score + token length)
  │
  └── (future) MLSweepStrategy (learned classification)
```

**Why Strategy Pattern here?** Sweeping is fundamentally an algorithm — like how JVM has multiple GC algorithms (G1, ZGC, Shenandoah). The sweep algorithm may evolve. The Strategy Pattern means evolution adds new classes without modifying existing code (Open/Closed Principle).

## Classification Categories

Classification is three-way, decided by where the combined score falls relative
to two thresholds:

| Classification | Condition | What Happens Downstream |
|---------------|-----------|------------------------|
| **KEEP** | Score >= `keep_threshold` OR token count <= `min_compactable_tokens` | Message stays in context as-is. No modification. |
| **COMPACT** | `archive_threshold` <= score < `keep_threshold` | Message is passed to the Compactor for compression. Substance preserved in fewer tokens. |
| **ARCHIVE** | Score < `archive_threshold` | Message leaves the context entirely. The Composer routes it to `GenerationalMemory.archive_message(...)`, which extracts its facts into the permanent generation. |

**Conservative default**: When in doubt, KEEP. A message incorrectly kept costs a few extra tokens. A message incorrectly compacted risks information loss. The asymmetry favors keeping.

**Why `min_compactable_tokens`?** A 3-token "ok" classified as COMPACT would produce a summary of similar length — wasted effort. Messages below 30 tokens stay as KEEP regardless of score because the compression savings are negligible.

**How ARCHIVE stays safe.** The classification itself is score-only — the sweeper
does not check whether a message's content is preserved anywhere before marking it
ARCHIVE. The guarantee is delivered downstream instead: `archive_message` runs the
extractor and, if it returns nothing, stores the **entire message verbatim** as a
`RAW` knowledge entry. Raw entries are exempt from topic supersession, so repeated
raw archives never overwrite one another. A message can therefore be archived
without its content being lost, whether or not the heuristics understood it.

> **Known gap: archived knowledge is not read back into context.** Assembly is
> old-generation summaries plus young-generation turns. Nothing injects relevant
> permanent-generation entries into the prompt, so today ARCHIVE means the message
> leaves the context and its facts go into a store only the dashboard reads. The
> knowledge is preserved, but the model does not see it. Closing this needs a
> relevance query over the permanent generation at assembly time.

## Override Rules

These overrides apply before any threshold logic. They are universal (strategy-independent):

| Rule | Classification | Reason |
|------|---------------|--------|
| Last N turns (configurable, default 5) | KEEP | Recent messages are almost always contextually needed. Compacting them would be premature. |
| System prompt (`role == "system"`) | KEEP | The system prompt defines model behavior. Modifying it would change the model's personality and constraints. |

When an override fires, the `SweepEntry` records `override_applied=True` and `override_reason` explaining which rule triggered.

## Core Design Principle: Quality Non-Negotiable

**LLM-GC must never cause quality loss.** If there is any doubt about whether a message can be safely compressed, the sweeper defaults to KEEP.

The difference from Java GC: in JVM GC, an object is either reachable or garbage - binary. In conversation context, a message is on a **spectrum of relevance**. 
A message scoring 0.25 isn't zero — some signal found it partially relevant. This means:

- COMPACT only when confident the message's meaning can be preserved in fewer tokens
- Never REMOVE - removal requires proof that information lives elsewhere
- The system errs on the side of keeping too much rather than losing something important

If llm-gc causes hallucination risk or quality degradation, it's worse than doing nothing. The user would be better off letting the context window fill naturally.

## ThresholdSweepStrategy — Classification Logic

The concrete strategy uses a simple, interpretable rule:

```python
if score >= keep_threshold OR token_count <= min_compactable_tokens:
    → KEEP
elif archive_threshold <= score < keep_threshold:
    → COMPACT
else:
    → ARCHIVE
```

**Why threshold-based?** It's the simplest sound approach — a clear, interpretable rule. The user can understand exactly *why* a message was classified the way it was ("score 0.45 < keep_threshold 0.7 and 100 tokens > min 30 → COMPACT"). No black box.

### Decision Tree

```
Is message too short to compact? (token_count <= 30)
  YES → KEEP (always — compacting saves nothing)
  NO  → Is score >= keep_threshold? (default 0.7)
          YES → KEEP (relevant enough to stay in full)
          NO  → COMPACT (low relevance, long enough to compress)
```

## Configuration

| Parameter | Default | Lives In | Purpose |
|-----------|---------|----------|---------|
| `keep_threshold` | 0.7 | `GCConfig` | Score at or above this → KEEP |
| `min_compactable_tokens` | 30 | `GCConfig` | Messages at or below this token count → always KEEP |
| `last_n_turns_to_keep` | 5 | `GCConfig` | Last N messages in conversation → always KEEP (override) |

### Why These Defaults?

- **keep_threshold = 0.7**: Combined score of 0.7 means at least some scorers found the message relevant. Conservative — most messages stay. Can be tuned down for aggressive compression.
- **min_compactable_tokens = 30**: Based on testing — a message like "Hi, it sounds good. Definitely let's go with PostgreSQL" is ~28 tokens. Compressing it would likely produce a summary of similar length. Not worth the effort.
- **last_n_turns_to_keep = 5**: The last 5 turns are almost always contextually needed for the next response. Compacting them would be premature.

## Why No Sliding Window / Why No REMOVE

### No REMOVE

`SweepClassification` has three values — KEEP, COMPACT, ARCHIVE — and no REMOVE.
Unlike Java GC, where an object is binary (reachable or unreachable), messages
have continuous relevance. Dropping one outright requires answering "is this
message's information preserved elsewhere?", and answering it *reliably*.
Heuristic fact extraction exists, but it is not a proof: it can miss, and a miss
means the information is simply gone.

ARCHIVE is as far as the sweeper goes. It takes a message out of context and
routes it through extraction into the permanent generation, so there is at least
an attempt to preserve what mattered — which is why the archive threshold is set
conservatively, and why the gap noted above matters.

Example: Turn 3 says "let's go with Redis." Turn 25 says "maybe PostgreSQL is better." Without relationship awareness, the sweeper might remove Turn 25 (lower recency score) while keeping Turn 3 — preserving the *wrong* decision.

## Event Emission

The Sweeper emits a `SWEEP_COMPLETED` event via the EventBus after every sweep:

```python
EventType.SWEEP_COMPLETED
data = {
    "sweep_entries": [...],          # Full list of classified messages
    "classification_counts": {...},  # {KEEP: N, COMPACT: M}
    "current_message_turn": int      # Total messages in conversation
}
```

This feeds into the Context Visualizer (turn lifecycle view) and benchmarking tools.

## How the Sweeper's Confidence Grows

The sweeper's interface stays stable. What changes is the quality of the inputs
it gets, and each improvement lets it classify with more confidence rather than
differently:

| Input | Effect on classification |
|---|---|
| Relevance scores and token count (today) | KEEP, COMPACT, ARCHIVE by threshold |
| Relationship awareness — knowing two messages cover the same decision | Archival that can tell a superseded turn from a unique one |
| Compactor feedback — did compression actually preserve meaning? | Thresholds tuned against real compaction quality |
| A learned relevance model | More accurate scores, so more confident classifications |

## Worked Example

Given a 6-message conversation with `keep_threshold=0.7`, `min_compactable_tokens=30`, `last_n_turns_to_keep=2`:

```
Turn 0: [system] "You are a coding assistant"
         score=0.15, tokens=20
         → KEEP (override: system prompt)

Turn 1: [user] "Let's build a REST API with FastAPI and PostgreSQL..."
         score=0.35, tokens=80
         → COMPACT (score < 0.7, tokens > 30, not in last 2)

Turn 2: [assistant] "Great choice! Here's the project structure..."
         score=0.45, tokens=200
         → COMPACT (score < 0.7, tokens > 30, not in last 2)

Turn 3: [user] "ok sounds good"
         score=0.10, tokens=4
         → KEEP (tokens <= 30, not worth compacting)

Turn 4: [user] "Now add JWT authentication with refresh tokens"
         score=0.85, tokens=60
         → KEEP (override: last 2 turns)

Turn 5: [assistant] "I'll implement JWT auth. Here's the approach..."
         score=0.90, tokens=300
         → KEEP (override: last 2 turns)
```

Result: `total_keep_tokens=384`, `total_compact_tokens=280`, potential savings from compressing turns 1-2.

## Design Rationale & Alternatives Considered

**Three tiers, not two.** The simplest cleanup is binary — keep or drop. It was
rejected because it collapses three genuinely different kinds of turn into one
decision: a turn still in its useful window (leave as-is), a turn no longer fresh
but still useful in summary (compact), and a turn whose verbatim form is spent but
which carries a decision or fact that must survive (archive). Keeping all three
wastes context; dropping all three loses decisions. The three-way split — KEEP /
COMPACT / ARCHIVE — is what lets each kind be handled correctly, and is what makes
the retention commitment deliverable.

A two-tier keep/compact split (no archive) was also rejected: compaction preserves
gist but cannot be relied on to preserve specific facts — *which* database was
chosen, *what* key format was decided. Decisions need a separate path that
extracts them into a durable store before the verbose conversation is dropped.

**Conservative by default.** The quality-preservation principle makes the cost
asymmetry explicit: a message kept unnecessarily costs a few bounded tokens; a
message removed wrongly risks an unpredictable wrong answer. So defaults lean KEEP
over COMPACT and COMPACT over ARCHIVE, and on a near-boundary score the more
conservative class wins. Aggressive defaults were rejected outright: maximising
token savings by default would invert the asymmetry the system is built around.
Token savings are a byproduct of preserving what matters, not the goal.

**A single tuning dial over two correlated knobs.** The design intent is to expose
cleanup aggressiveness as *one* control that moves both thresholds together, rather
than letting users set `keep_threshold` and `compact_threshold` independently. The
thresholds are correlated: moving one without the other silently shifts the
COMPACT-vs-ARCHIVE balance in ways most users do not intend. A single dial keeps
the conservative-to-aggressive axis coherent by construction; the per-threshold
values remain available as advanced overrides for users who need finer control.

## Limitations & Future Improvements

- **No relationship awareness**: The sweeper doesn't know if two messages discuss the same topic or contradict each other.
- **Fixed threshold**: The same `keep_threshold` applies to all message types. An adaptive threshold (stricter for assistant messages which are longer, gentler for user messages which contain intent) could be more effective.
- **No compaction quality feedback**: The sweeper classifies blindly — it doesn't know whether compression will actually preserve meaning. Closing that loop needs the compactor to report back.
- **Single strategy**: Only threshold-based classification. A learned strategy that trains on compaction outcomes is the obvious next one.

## File Map

```
src/llm_gc/
  engine/
    sweep/
      __init__.py                    # Public exports
      sweep_classification.py        # SweepClassification enum (KEEP, COMPACT)
      sweep_entry.py                 # Per-message classification result
      sweep_result.py                # Aggregate sweep output with metrics
      base_sweep_strategy.py         # Strategy Pattern ABC
      threshold_sweep_strategy.py    # Concrete strategy: threshold-based classification
      sweeper.py                     # Orchestrator: overrides + strategy + events

tests/engine/sweep/
  test_threshold_sweep_strategy.py   # Strategy classification logic tests
  test_sweeper.py                    # Orchestrator, overrides, aggregation tests
```