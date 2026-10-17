# Knowledge Entry Model

## What It Is

A `KnowledgeEntry` is one unit of knowledge read from a conversation turn — a fact, a decision or a preference. Every turn is read once, from its own words, when a collection makes its span; the entries are filed in the session's permanent generation and kept for the session's life. They are what an archived span shows beside its opening line, and the record of what the conversation later changed its mind about.

## How an entry reaches the model

```
 turn 5: "Let's use PostgreSQL for the database."
            │ read once, when its span is made
            ▼
 KnowledgeEntry(topic="database", content="postgresql", message_turn=5, ACTIVE)
            │
            ├─ span archived or summarised ──► on its line:
            │     [Archived] (turns 5–6) Let's use PostgreSQL for the database. · database: postgresql
            │
            └─ turn 15 says "switch to MySQL" ──► a new entry supersedes it, and turn 5's span,
                  at whatever level it is shown, carries:
                  ↳ Replaced later (turn 15): database: mysql, not postgresql.
```

A kept span shows its own words, which already state its facts, so its entries are not listed beside it. A folded span shows nothing of its own. A superseded entry is never shown as a fact — only named in the note on its span.

---

The word "Knowledge" is deliberate — these are not necessarily objective facts. They include:
- Project state ("Database is MySQL")
- Decisions with reasoning ("Chose MySQL over PostgreSQL for cheaper hosting")
- User preferences ("Prefers concise responses")

All are stored identically. The `KnowledgeType` enum differentiates their semantic meaning for the dashboard.

---

## Data Model

```python
@dataclass(frozen=True)
class KnowledgeEntry:
    message_turn: int              # which turn this was read from
    content: str                  # the knowledge itself (includes reasoning if relevant)
    topic_label: str              # grouping key for contradiction detection
    knowledge_type: KnowledgeType # FACT, DECISION, or PREFERENCE
    knowledge_status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    speaker: Literal["user", "assistant"] | None = None   # who stated it
```

### Why `frozen=True`?

A KnowledgeEntry is a historical record. The message it was read from cannot change for that turn. If the user changes their mind later, that produces a *new* entry that supersedes the old one. The old entry remains frozen as history.

### Why `@dataclass` instead of Pydantic?

KnowledgeEntry is internally constructed by the system (from messages it already validated). Pydantic is for validation boundaries (external/untrusted input). Dataclasses are for internal value objects where the data is already trusted.

---

## KnowledgeType Enum

| Value | Meaning | Example |
|-------|---------|---------|
| `FACT` | A piece of project/domain state | "Project is a fintech app" |
| `DECISION` | A choice made (content includes reasoning) | "Database is MySQL — chose over PostgreSQL for cheaper hosting" |
| `PREFERENCE` | A behavioral preference | "User prefers concise responses" |

The type does not affect storage, rendering or contradiction logic. It is a semantic label the dashboard uses, for example to list decisions on their own.

---

## KnowledgeStatus Enum

| Value | Meaning |
|-------|---------|
| `ACTIVE` | Current knowledge — shown on its span's line when the span is summarised or archived |
| `SUPERSEDED` | Replaced by a newer entry — kept for history; shown only as the "replaced later" note on its span |

---

## Contradiction Detection

When a new entry shares the same `topic_label` as an existing ACTIVE entry, the old entry is superseded:

```
Turn 5:  KnowledgeEntry(content="Database is PostgreSQL", topic_label="database", status=ACTIVE)
Turn 8:  KnowledgeEntry(content="Database is MySQL", topic_label="database", status=ACTIVE)
         → Turn 5 entry becomes SUPERSEDED
```

Because entries are frozen (immutable), supersession creates a *new copy* of the old entry with `knowledge_status=SUPERSEDED` using `dataclasses.replace()`. The original object is never mutated.

An entry whose topic is `general` — a decision that names no topic, like "we decided to ship on Friday" — supersedes nothing (`KnowledgeEntry.can_supersede`). Matched as a topic, it would retire every earlier topic-less decision, however unrelated.

### Limitation

Contradiction detection relies on an exact `topic_label` match. "database" and "db-choice" are different topics. The reader keeps labels consistent: the model is shown the facts on record and files a changed value under the topic it changes. An entry restating its topic's active value unchanged is not filed, so a repetition never reads as a change. Patterns, used without a model, label by the sentence's subject and cannot see the labels in use, so a change worded differently is missed.

Each entry records its `speaker`, from the turn it was read from. An entry the assistant stated on a topic the user decided is not filed: only the user changes their own decision, and the recommendation stays in its exchange's summary. A tool result and a subagent's relayed report arrive under the user role, but their speaker is the assistant: they are what the assistant's side found, not what the user said. Read as the user's, a subagent's finding — "the config uses MySQL" — would replace the decision the user made on the same topic. The model reading facts is shown the report labelled `subagent report`, never as the user.

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| Single class, no inheritance | The `KnowledgeType` enum differentiates semantics without separate classes. Reasoning is part of the `content` string. |
| Read from the original turns, once per collection | A fact read from a summary is read second-hand; reading when the spans are made means no level ever waits on a model. Each entry names exactly one turn. |
| Facts on their span's line, not in a block of their own | Each turn is represented once in the front, in conversation order; a separate block of facts would repeat what summarised spans already say, under a second budget. |
| `frozen=True` | Historical records must be immutable. Supersession creates new copies. Prevents bugs from shared mutable references. |
| `@dataclass` over Pydantic | Internal value object — no validation boundary needed. Lighter weight. |
| `reasoning` folded into `content` | Avoids a field that's empty for FACT and PREFERENCE types. The reader writes the value alone. |
| Named `KnowledgeEntry` not `Fact` | "Fact" implies objective, constant truth. These are units of knowledge read from turns, and they can be superseded. |
