# Density Scorer

## Why Does This Exist?

Not all messages carry the same amount of information. Compare:

- **"ok"** — zero information, pure acknowledgment
- **"thanks!"** — zero information, social signal
- **"Check `Python` 3.12 at https://python.org — NASA uses it for 42% of their tooling"** — packed with specific, referenceable information

If we only scored by recency, "ok" from 2 turns ago would score higher than a detailed technical decision from 8 turns ago. The density scorer prevents this by measuring **how much information a message actually carries** relative to its size.

In garbage collection terms: filler messages are "garbage" almost immediately — they consume tokens but contribute nothing to the model's understanding. Dense messages are "live objects" that should survive longer.

## How It Works

The density scorer counts **information-carrying signals** in the message text, divides by token count, and normalizes the result.

### The Formula

```
density_ratio = signal_count / token_count
density_score = min(density_ratio / max_density, 1.0)

where:
  signal_count = number of information-carrying patterns found (via regex)
  token_count  = total tokens in the message (via tiktoken)
  max_density  = normalization ceiling (default: 0.5)
```

### What Counts as a "Signal"?

The regex pattern (`DENSITY_SCORER_PATTERN` in `constants.py`) detects four types of information-carrying content:

```
Pattern                          What It Catches             Example
────────────────────────────     ──────────────────────      ─────────────────
\b[A-Z][a-zA-Z]+\b              Capitalized words           Python, FastAPI, NASA
\b\d+\.?\d*\b                   Numbers (incl. decimals)    128000, 3.14, 42
`[^`]+`                          Code fragments (backticks)  `async def`, `pip install`
https?://\S+                     URLs                        https://python.org
```

**Why these four?** They're proxies for *specificity*. Messages with proper nouns, numbers, code, and links tend to contain concrete, referenceable information. Messages without them ("sounds good", "let me think about that") tend to be conversational filler.

### Why ratio, Not Raw Count?

A 500-token message with 10 signals is less dense than a 20-token message with 10 signals. Raw count would favor long messages regardless of quality. The ratio measures *concentration* of information.

### How `max_density` Works

`max_density` (default 0.5) is the normalization ceiling — any ratio at or above it scores 1.0.

```
density_ratio = 0.25, max_density = 0.5  →  score = 0.25 / 0.5 = 0.50
density_ratio = 0.50, max_density = 0.5  →  score = 0.50 / 0.5 = 1.00
density_ratio = 0.75, max_density = 0.5  →  score = min(1.5, 1.0) = 1.00  (capped)
```

This prevents unreasonably high ratios (like a 2-token message "Python 3.12") from distorting the score beyond 1.0.

## Worked Example

**Message**: `"Use FastAPI with Pydantic v2 for validation. Check https://fastapi.tiangolo.com"`

Signals found:
1. `FastAPI` — capitalized word
2. `Pydantic` — capitalized word
3. `Use` — capitalized word
4. `Check` — capitalized word
5. `2` — number
6. `https://fastapi.tiangolo.com` — URL

Signal count: **6**
Token count (via tiktoken): **~12**
Density ratio: 6 / 12 = **0.50**
Score: min(0.50 / 0.50, 1.0) = **1.00**

**Message**: `"ok sounds good let me try that"`

Signals found: **0** (no capitalized words, numbers, code, or URLs)
Token count: **~7**
Density ratio: 0 / 7 = **0.00**
Score: **0.00**

The contrast is exactly what we want — the first message is worth keeping, the second is safe to discard.

## Design Decisions

**Why regex instead of NLP / named entity recognition?**
Regex is fast (~microseconds), deterministic, and requires no model loading. For now it's the right trade-off: good enough to separate "ok" from "Use PostgreSQL with pgvector", and NLP-based detection can replace it later. The scorer interface stays the same — only the internal detection logic changes.

**Why count capitalized words as signals?**
In technical conversations, capitalized words strongly correlate with named entities — tools (`FastAPI`), languages (`Python`), services (`AWS`), concepts (`JWT`). This is a heuristic, not perfect — it will match sentence-starting words too. But across a conversation, the signal-to-noise ratio is good enough. The architecture doc's original design also listed "named entity count" as a density indicator.

**Why is `max_density` configurable?**
Different conversation styles have different natural density. A code-heavy conversation might have higher baseline density than a brainstorming session. Making it configurable lets benchmarking find optimal values per conversation type.

**Edge case — empty messages:**
If `token_count` is 0 (empty message), the scorer returns 0.0 instead of dividing by zero. This is handled explicitly in the implementation.

## Signals Emitted

```python
signals = {
    "signal_count": 6,          # Number of regex pattern matches
    "token_counts": 12,         # Total tokens in the message
    "density_ratio": 0.5,       # signal_count / token_counts
    "density_score": 1.0        # Final normalized score
}
```

These signals feed into the Context Visualizer — particularly useful for identifying which messages are "waste" (high token count, low density).

## Limitations & Future Improvements

- **Regex is a blunt instrument**: Can't distinguish "Use Python" (specific recommendation) from "Use whatever you think is best" (vague). Both have a capitalized word, but only the first carries specific information.
- **No semantic understanding**: The scorer doesn't know that "PostgreSQL" is more information-dense than "Something" — both are capitalized words. NLP-based entity recognition would fix this.
- **Sentence-start capitalization**: English capitalizes the first word of every sentence regardless of whether it's a named entity. This inflates signal count slightly. Acceptable for now — the ratio normalizes it.
- **No code block detection**: Currently only detects inline backtick code. Multi-line code blocks (triple backticks) contain high-density content but aren't counted as a single signal. Future improvement.

## File Location

- Implementation: `src/llm_gc/scoring/density_scorer.py`
- Regex pattern: `src/llm_gc/config/constants.py` → `DENSITY_SCORER_PATTERN`
- Tests: `tests/scoring/test_density_scorer.py`