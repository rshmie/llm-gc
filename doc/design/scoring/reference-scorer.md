# Reference Scorer

## Why Does This Exist?

Some messages are important not because of what they contain in isolation, but because **later messages refer back to them**. If turn 12 says "as we discussed in the PostgreSQL setup...", then the PostgreSQL setup turn is still contextually relevant — even if it's old and the recency scorer gave it a low score.

This is a **backward-looking importance signal**: instead of scoring a message by its own content (like density or decision scorers do), the reference scorer scores it by how the *rest of the conversation* treats it. A message that nobody ever refers back to was probably not that important. A message that gets referenced three times is clearly load-bearing.

In garbage collection terms: this is like reference counting. An object (message) stays alive as long as other live objects (later messages) hold references to it.

## How It Works

For a given message, the reference scorer checks **all later messages** in the conversation for reference-indicator phrases. Each later message that contains a reference phrase increments the count.

### The Formula

```
reference_score = min(reference_count / max_references, 1.0)

where:
  reference_count = number of later messages containing reference phrases
  max_references  = normalization threshold (default: 3)
```

**Important nuance**: the scorer counts *how many later messages reference* (not how many reference phrases appear total). One later message with "as you mentioned earlier, like we discussed" counts as 1, not 2. This is intentional — what matters is how many distinct conversation turns found this message worth referring back to.

### What Counts as a Reference?

The regex pattern (`REFERENCE_SCORER_PATTERN` in `constants.py`) detects three categories of backward references:

```
Category                  Phrases Detected
────────────────────      ──────────────────────────────────────────────────
Explicit references       "as I mentioned", "as you said", "like you said",
                          "as we discussed", "like we discussed",
                          "as I said", "like I said", "as mentioned",
                          "as noted", "as we said", "as we mentioned",
                          "as we noted"

Temporal references       "earlier", "previously", "before",
                          "going back to", "back to your point",
                          "back to what you said", "referring back to"

Reiteration               "to reiterate", "to repeat", "referring to",
                          "regarding what"
```

All matching is **case-insensitive**.

### A Subtle Point: Which Message Gets the Score?

The reference scorer scores the **original message being evaluated**, not the message that contains the reference phrase. If message A is being scored and message C (a later message) says "as we discussed", then message A's reference count goes up — because C is evidence that A was important enough to refer back to.

```
Message A: "Let's use PostgreSQL"        ← This gets the score boost
Message B: "ok sounds good"
Message C: "As we discussed, PostgreSQL   ← This contains the reference phrase
            needs pgvector extension"        (but C is not the one being scored)
```

## Worked Example

**Scoring message at index 1 in a 5-message conversation:**

```
Index 0: "We need a database for the project"
Index 1: "Let's use PostgreSQL with pgvector"    ← scoring this message
Index 2: "ok"
Index 3: "As you mentioned earlier, PostgreSQL needs setup"   ← has reference phrase
Index 4: "Going back to the database choice, what about Redis?"  ← has reference phrase
```

Later messages (index 2, 3, 4):
- Index 2: "ok" → no reference phrase → skip
- Index 3: "As you mentioned earlier..." → match → reference_count = 1
- Index 4: "Going back to..." → match → reference_count = 2

Score: min(2 / 3, 1.0) = **0.67**

---

**Scoring the last message in a conversation:**

The last message has no later messages → reference_count = 0 → score = **0.00**

This is correct — a message that was just sent can't have been referenced yet. Its importance will emerge over time as the conversation continues.

## Design Decisions

**Why count referencing messages, not total reference phrases?**
If one message says "as we discussed earlier, like I mentioned before, referring back to what I said" — that's one message with three reference phrases, but it's still one act of referring back. Counting messages instead of phrases prevents a single verbose reference from inflating the score.

**Why `max_references = 3` as default?**
In real conversations, being referenced by 3+ different later messages is rare and signals very high importance. A single reference (score 0.33) gives a meaningful but modest boost. Two references (0.67) puts the message firmly in the "compress, don't delete" zone. Three or more (1.0) makes it maximally important from a reference perspective.

**Why look at ALL later messages, not just the next N?**
A reference from turn 50 to turn 5 is just as meaningful as a reference from turn 6 to turn 5 — arguably more so, because it means the original message stayed relevant across a long stretch of conversation. Limiting the lookback would miss these long-range references.

**Why doesn't this scorer identify which specific message is being referenced?**
The current approach checks whether later messages contain reference phrases, but doesn't match *which* earlier message they're referring to. "As we discussed" could be referring to any earlier message. This is a known limitation — matching specific cross-references requires deeper NLP (coreference resolution). For now, the scorer applies the reference boost to whichever message is currently being scored, which is imprecise but directionally correct.

## Signals Emitted

```python
signals = {
    "reference_count": 2,       # Number of later messages with reference phrases
    "reference_score": 0.67     # Final normalized score
}
```

In the Context Visualizer, these signals will help show conversation "threads" — which messages are being referred back to, and how far back the references reach.

## Limitations & Future Improvements

- **No targeted reference resolution**: The scorer detects that a later message *makes a reference*, but can't determine *which specific earlier message* it's referring to. "As you mentioned" could point to any prior message. True NLP coreference resolution would fix this.
- **False positives on temporal words**: "earlier" and "before" match as reference phrases, but "I've never done this before" isn't a backward reference — it's just using the word "before" in a different sense. These words are included because they're more often references than not in conversational context, but they add noise.
- **No entity-based reference detection**: If turn 5 introduces "PostgreSQL" and turn 20 says "the PostgreSQL setup", that's a reference by shared entity — but the current regex approach can't detect it. The architecture doc lists "shared named entities between turns" as a future detection method.
- **Last message always scores 0.0**: By definition, the most recent message can't have been referenced yet. This is correct behavior but means the reference scorer contributes nothing to the newest message's combined score. Other scorers (recency, density) compensate.
- **O(n) per message**: For each message being scored, the scorer iterates through all later messages. Scoring the full conversation is O(n²) in the worst case. For conversation lengths under a few hundred messages, this is negligible. If it becomes a bottleneck, pre-computing reference counts in a single pass would be straightforward.

## File Location

- Implementation: `src/llm_gc/scoring/reference_scorer.py`
- Regex pattern: `src/llm_gc/config/constants.py` → `REFERENCE_SCORER_PATTERN`
- Tests: `tests/scoring/test_reference_scorer.py`