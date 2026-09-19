# Knowledge Entry Model

## What It Is

A `KnowledgeEntry` is an extracted unit of knowledge from a conversation turn. It represents
a piece of information that must survive even after the original turn is archived (removed
from context).

The word "Knowledge" is deliberate — these are not necessarily objective facts. They include:
- Project state ("Database is MySQL")
- Decisions with reasoning ("Chose MySQL over PostgreSQL for cheaper hosting")
- User preferences ("Prefers concise responses")

All are stored identically. The `KnowledgeType` enum differentiates their semantic meaning
for the visualizer and retrieval logic.

---

## Data Model

```python
@dataclass(frozen=True)
class KnowledgeEntry:
    message_turn: int              # which turn this was extracted from
    content: str                  # the knowledge itself (includes reasoning if relevant)
    topic_label: str              # grouping key for contradiction detection
    knowledge_type: KnowledgeType # FACT, DECISION, or PREFERENCE
    knowledge_status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
```

### Why `frozen=True`?

A KnowledgeEntry is a historical record. The message it was extracted from cannot change
for that turn. If the user changes their mind later, that produces a *new* entry that
supersedes the old one. The old entry remains frozen as history.

### Why `@dataclass` instead of Pydantic?

KnowledgeEntry is internally constructed by the system (from messages it already validated).
Pydantic is for validation boundaries (external/untrusted input). Dataclasses are for
internal value objects where the data is already trusted.

---

## KnowledgeType Enum

| Value | Meaning | Example |
|-------|---------|---------|
| `FACT` | A piece of project/domain state | "Project is a fintech app" |
| `DECISION` | A choice made (content includes reasoning) | "Database is MySQL — chose over PostgreSQL for cheaper hosting" |
| `PREFERENCE` | A behavioral preference | "User prefers concise responses" |

The type does NOT affect storage or contradiction logic. It's a semantic label for:
- The visualizer's Decision Survival Map (filters by `DECISION`)
- Future retrieval heuristics (preferences might be weighted differently)

---

## KnowledgeStatus Enum

| Value | Meaning |
|-------|---------|
| `ACTIVE` | Current, valid knowledge — injected into context on retrieval |
| `SUPERSEDED` | Replaced by a newer entry — preserved for history, never injected |

---

## Contradiction Detection

When a new entry shares the same `topic_label` as an existing ACTIVE entry, the old
entry is superseded:

```
Turn 5:  KnowledgeEntry(content="Database is PostgreSQL", topic_label="database", status=ACTIVE)
Turn 8:  KnowledgeEntry(content="Database is MySQL", topic_label="database", status=ACTIVE)
         → Turn 5 entry becomes SUPERSEDED
```

Because entries are frozen (immutable), supersession creates a *new copy* of the old entry
with `knowledge_status=SUPERSEDED` using `dataclasses.replace()`. The original object is
never mutated.

### Limitation

Contradiction detection relies on exact `topic_label` match. "database" and "db-choice"
would be treated as different topics. The FactExtractor is responsible for
consistent labeling; semantic matching would improve on it.

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| Single class, no inheritance | `DecisionFact` subclass was removed — the `KnowledgeType` enum differentiates semantics without needing separate classes. Reasoning is part of `content` string. |
| `frozen=True` | Historical records must be immutable. Supersession creates new copies. Prevents bugs from shared mutable references. |
| `@dataclass` over Pydantic | Internal value object — no validation boundary needed. Lighter weight. |
| `reasoning` folded into `content` | Avoids a field that's empty for FACT and PREFERENCE types. Extractor formats content naturally. |
| Named `KnowledgeEntry` not `Fact` | "Fact" implies objective, constant truth. These are extracted knowledge units that can be superseded. |
