# Similarity Scorer

## Why Does This Exist?

Conversations drift. A 40-turn conversation might start with database design, move through API architecture, touch on authentication, then land on deployment. 
When the context window fills up, the GC needs to know: **which old messages are still relevant to what the user is asking about right now?**

Consider:

> **Turn 3**: "Let's use bcrypt with a work factor of 12 for password hashing."
>
> **Turn 25**: User discusses API rate limiting for 10 turns.
>
> **Turn 38**: "How should we handle password reset tokens?"

Turn 3 is old (recency score: low). It's not referenced by later messages (reference score: low). But it's **semantically related** to the current question about password reset - both are about authentication/security. 
The similarity scorer is the only signal that can rescue this message from being garbage collected.

**Without the similarity scorer**, relevance is purely based on age, density, decisions, and back-references. A topically relevant but old message has no defender. With it, the scoring pipeline can recognize: "this message is about the same *topic* the user is asking about now."

## How It Works

The similarity scorer uses **sentence embeddings** - vector representations of text meaning - to compute how semantically close a message is to the current query (the last message in the conversation).

### The Process

```
1. Identify the query: conversation[-1] (last message, any role)
2. Encode both texts into vectors using a sentence-transformer model
3. Compute cosine similarity between the two vectors
4. Clamp to [0.0, 1.0] (negative similarity → 0.0)
```

### The Formula

```
cosine_similarity = dot(A, B) / (norm(A) * norm(B))

similarity_score = max(0.0, cosine_similarity)

where:
  A    = embedding vector of the message being scored
  B    = embedding vector of the current query (conversation[-1])
  dot  = dot product (sum of element-wise multiplication)
  norm = Euclidean norm (magnitude of the vector)
```

Cosine similarity measures the angle between two vectors in high-dimensional space:
- 1.0 = vectors point in the same direction (identical meaning)
- 0.0 = vectors are perpendicular (unrelated topics)
- -1.0 = vectors point in opposite directions (clamped to 0.0)

### Why Cosine, Not Euclidean Distance?

Cosine similarity measures **direction** (what the text is about), not **magnitude** (how long or emphatic it is). Two sentences about authentication will point in a similar direction regardless of whether one is 5 words and the other is 50 words. 
Euclidean distance would penalize the length difference. For topic relevance, direction is what matters.

## The Embedding Model

The scorer uses `sentence-transformers/all-MiniLM-L6-v2`:
- **Architecture**: 6-layer BERT, fine-tuned for semantic similarity
- **Output**: 384-dimensional vector per sentence
- **Size**: ~80MB
- **Speed**: <10ms per encoding on CPU
- **Quality**: Good enough for topic-level similarity; not suitable for fine-grained entailment

The model is injected via the constructor (dependency injection) — the scorer doesn't create or manage it. This means:
- One model instance is shared across the application
- Tests can inject a fake model (no 80MB download needed)
- The model can be swapped in the future without touching scorer logic

### Why Local Embeddings (Not an API)?

| Consideration | Local Model | API (e.g., OpenAI) |
|--------------|-------------|---------------------|
| Latency | <10ms | 100–500ms |
| Cost | Free (after download) | Per-token pricing |
| Frequency | Called for every message every GC cycle | Would be expensive at scale |
| Quality | Good for topic similarity | Higher quality, overkill for this signal |
| Availability | Always works (offline) | Requires network, can fail |

The scorer runs on **every message in the conversation per GC cycle**. API calls would add 100ms+ per message, potentially seconds per cycle. Local embeddings make this practical.

## Worked Example

**Conversation**:
```
[0] user:      "Set up PostgreSQL with connection pooling"
[1] assistant: "Here's the pgbouncer config..."
[2] user:      "Now let's add Redis caching for sessions"
[3] assistant: "Redis config with TTL..."
[4] user:      "How do I handle database migrations?"   ← query (conversation[-1])
```

**Scoring message [2]** ("Now let's add Redis caching for sessions") against query [4] ("How do I handle database migrations?"):

Both are about infrastructure/databases but different subtopics. The embedding model captures the partial overlap (both mention data storage concerns) but also the difference (caching vs migrations).

```
embedding_msg   = model.encode("Now let's add Redis caching for sessions")   → [0.3, 0.8, 0.1, ...]
embedding_query = model.encode("How do I handle database migrations?")       → [0.7, 0.4, 0.2, ...]

cosine_similarity ≈ 0.45
similarity_score  = max(0.0, 0.45) = 0.45
```

**Scoring message [0]** ("Set up PostgreSQL with connection pooling") against query [4]:

Both are squarely about database operations. The vectors will be closer.

```
cosine_similarity ≈ 0.72
similarity_score  = 0.72
```

Message [0] scores higher despite being older — exactly the behavior we want. The recency scorer penalizes it for age; the similarity scorer rescues it for topical relevance.

## Edge Cases

**Message is the query itself**: If the message being scored IS `conversation[-1]`, return 1.0 immediately without computing embeddings. Comparing a message to itself is trivially 1.0 and wastes computation.

**Single-message conversation**: The only message is both the target and the query → handled by the same `message is query_message` check → returns 1.0.

**Empty message content**: The model will encode it (producing a non-zero vector from tokenizer special tokens). The cosine similarity will naturally be low against any meaningful query. No special handling needed — the math does the right thing.

**Negative cosine similarity**: Rare with sentence-transformers (most unrelated texts score near 0, not negative). Clamped to 0.0 because our scoring contract requires [0.0, 1.0] and negative values don't carry actionable signal for the sweeper.

## How It Interacts With Other Scorers

Similarity has the joint-highest weight (0.30, tied with recency). This reflects that topical relevance is one of the two strongest general signals for "should we keep this?"

```
Example: An old but topically relevant message

  Recency score:    0.10  (very old → low recency)
  Similarity score: 0.85  (same topic as current query!)
  Density score:    0.60  (moderately information-rich)
  Decision score:   0.00  (no decision phrases)
  Reference score:  0.00  (not referenced later)

  Combined: 0.30(0.10) + 0.30(0.85) + 0.15(0.60) + 0.15(0.00) + 0.10(0.00)
          = 0.03 + 0.255 + 0.09 + 0 + 0
          = 0.375  → COMPRESSED (preserved as summary)

  Without similarity scorer:
  Combined: 0.30(0.10) + 0.15(0.60) + 0.15(0.00) + 0.10(0.00) = 0.12  → REMOVED

  With similarity scorer:
  Combined: 0.375  → COMPRESSED (summary preserved)
```

The similarity scorer prevents topically relevant messages from being removed just because they're old. It's the counterbalance to recency — together they represent "when was it said?" vs "is it still about what we're discussing?"

## Design Decisions

**Why use the last message as query, regardless of role?**
The last message — whether user or assistant — best represents the current topic. If the last message is an assistant response about authentication, that *is* the current topic. Using only the last *user* message would miss cases where the user says "continue" or "ok" — semantically empty messages that don't represent the topic.

**Why dependency injection for the model?**
The model takes ~1 second to load and uses ~80MB of memory. In production, exactly one instance should exist, shared across all components that need embeddings. The scorer shouldn't own model lifecycle — it just uses whatever it's given. This also makes testing trivial: inject a fake model that returns predictable vectors.

**Why clamp negatives to 0.0 instead of rescaling [-1, 1] → [0, 1]?**
Rescaling would mean a cosine of 0.0 (completely unrelated) maps to 0.5 — implying "moderately similar." That's misleading. Unrelated should be 0, and only genuinely similar content should score above 0. The clamp preserves this semantic meaning.

**Why not cache embeddings?**
The same message gets scored in every GC cycle. Re-encoding it every time is wasteful. However, caching adds complexity (cache invalidation, memory management) that we don't need yet. Caching belongs where the pipeline is wired together, not in the scorer. For now, correctness over performance.

## Signals Emitted

```python
signals = {
    "cosine_similarity": 0.72,    # Raw cosine value (can be negative)
    "similarity_score": 0.72      # Clamped score [0.0, 1.0]
}
```

In the Context Visualizer, these signals will power a **topic relevance heatmap** - showing which messages in the conversation are semantically close to the current query and which have drifted off-topic.

## Limitations & Future Improvements

- **Single-query comparison**: The scorer compares against only the last message. A user might have a multi-turn topic that the last message alone doesn't capture. Future improvement: compare against a "topic window" (last N messages averaged).
- **No embedding caching**: Every GC cycle re-encodes all messages. A cache keyed by content hash would remove the redundant computation.
- **Model quality ceiling**: `all-MiniLM-L6-v2` is a lightweight model. It captures topical similarity well but misses subtle relationships (e.g., "database migrations" and "ALTER TABLE" might not score as high as expected). A larger model could be swapped in if quality metrics justify the memory/speed trade-off.
- **Language sensitivity**: The model was primarily trained on English text. Multilingual conversations will score lower than they should. Not a concern for now but relevant if the project expands scope.
- **No query decomposition**: "How do I handle database migrations and also set up Redis caching?" is one query about two topics. The scorer treats it as a single vector - messages about either topic should score high, but the averaged vector might dilute both signals.

## File Location

- Implementation: `src/llm_gc/scoring/similarity_scorer.py`
- Tests: `tests/scoring/test_similarity_scorer.py`