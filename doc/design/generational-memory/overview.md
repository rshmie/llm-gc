# Generational Memory Model

## The problem

LLM conversations grow linearly. A 100-turn conversation at 300 tokens a turn is 30,000 tokens, and every request sends all of it again. Context windows have hard limits, every token costs money and time, and long prompts dilute the model's attention. When a conversation gets long, the usual answers are:

- **Truncate from the top** — early turns are lost for good.
- **A sliding window (keep the last N)** — the same loss, at a predictable moment.
- **Summarise everything** — smaller, but the detail inside each summary is gone, and summaries of summaries lose more each time.

All three share one failure: **information is destroyed, and nothing can bring it back** when the conversation returns to it. The system cannot promise that a decision from turn 5 still reaches the model at turn 500.

LLM-GC's answer is to **never destroy anything**, and instead decide, again and again, **how much detail of each part of the conversation the model is sent**.

---

## The idea in one picture

```
 THE CONVERSATION — every turn, kept whole (the source of truth)
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ t0 t1 │ t2 t3 │ t4 t5 │ t6 t7 │ t8 t9 │ … │ t40 t41 t42 t43 t44               │
 └─────────────────────────────────────────────────────────────────────────────┘
  span A  span B  span C  span D  span E          the tail
 ◄──────────── old generation: spans ─────────────►  ◄── young generation ──►

 WHAT THE MODEL RECEIVES ON EACH REQUEST
 ┌ the FRONT — one message, rebuilt only when a collection commits ───────────────┐
 │ span A   ·  out of view   (folded with its neighbours into one line)           │
 │ span B   ·  archived      opening line + the facts read from it                │
 │ span C   ▤  summarised    opening line + facts + a summary                     │
 │ span D   █  kept          every word, quoted exactly                           │
 │ span E   ▤  summarised    … + "↳ Replaced later (turn 22): cloud: AWS"         │
 └───────────────────────────────────────────────────────────────────────────────┘
   t40 t41 t42 t43 t44   ← the tail, exactly as the client sent it

 PERMANENT GENERATION — facts read from every turn, with what replaced what
```

- Every turn is kept, word for word, for the life of the session.
- The newest turns are sent exactly as written.
- Older turns are grouped into **spans** — one exchange each — and every span is shown at one of four **detail levels**.
- At every collection, LLM-GC re-scores every span by how much it matters **now** and re-picks its level. A span the conversation returns to is shown in full again, from its own words.

---

## The three generations

A **turn** in these docs is one message of the conversation: a user message, a reply, or a tool result. One prompt to a coding agent is many turns, since each tool call and its result is a turn of its own; the dashboard's Session tab, which follows the person, counts prompts instead. The settings say "messages" for the same thing.

| Generation | Holds | Sent to the model as |
|---|---|---|
| **Young** (the tail) | The newest turns — at least `last_n_messages_to_keep` of them | Exactly as the client sent them |
| **Old** | Every older turn, grouped into spans, each with its original words | The front: each span at its detail level |
| **Permanent** | The facts, decisions and preferences read from every turn, active or replaced | On their span's line, and as "replaced later" notes |

Turns move from young to old once, when a collection takes them. They never move into the permanent generation: **the permanent generation holds knowledge about turns, not turns.** A span never leaves the old generation either; it is only shown with more or less detail.

```
   young generation                old generation                   permanent generation
 ┌─────────────────┐  collection  ┌──────────────────────────┐      ┌──────────────────────┐
 │ newest turns,   │ ───────────► │ spans, each holding its  │ ───► │ facts read from each │
 │ word for word   │  (once)      │ turns, shown at a level  │ read │ turn when its span   │
 └─────────────────┘              │ that can rise or fall    │ once │ is made              │
                                  └──────────────────────────┘      └──────────────────────┘
```

---

## Spans: what a collection makes

A **span** is one exchange: something the user said or asked, and everything that came back before they spoke again.

Think of the minutes of a meeting. Each agenda item — a question raised, and the discussion that settled it — is one entry. Minutes can record an item in full, as a short paragraph, or as a one-line decision, but they never record half an item. LLM-GC treats a conversation the same way: the span is the item, and its detail level is how fully it is recorded.

```
 turn  who         what was said                                     span
 ────  ─────────   ───────────────────────────────────────────────   ──────────────────
   4   user        What should the seat hold be?                     ┐ turns 4–5
   5   assistant   Seven minutes: long enough to pay, short enough…  ┘
   6   user        Let's use Stripe for payments, with a 4% fee.     ┐ turns 6–7
   7   assistant   Agreed. Stripe takes the fee before the payout…   ┘
   8   user        Check the current fee table, please.              ┐
   9   assistant   [tool call] read fees.csv                         │ turns 8–11
  10   user        [tool result] the file's contents                 │ a tool result is
  11   assistant   The table already lists 4% for…                   ┘ not a new question
```

### What a span carries

When a collection makes span 6–7, it stores everything any level will ever need, once:

```
 span 6–7
 ├ its turns          turns 6 and 7, word for word        kept for the life of the session
 ├ its opening line   "Let's use Stripe for payments,     the user's first sentence:
 │                     with a 4% fee."                    what was asked or decided
 ├ its facts          fee: 4%                             read from the turns
 └ its summary        "The user chose Stripe with a 4%    one model call
                       fee; the assistant agreed, noting
                       Stripe takes it before the payout."
```

The [four detail levels](#the-four-detail-levels) are four ways of showing this one span, so choosing between them later costs no model call.

- **No summary is made** when the span is too short for any summary to be shorter than its words; such a span is kept or archived, never summarised.
- **Facts are read from the original turns**, never from the summary, so a mistake in a summary cannot become a fact.

### Why the unit is an exchange, not a single message

- **A question and its answer only mean something together.** "Seven minutes" on its own answers nothing; "What should the seat hold be?" on its own has no answer. Condensed apart, the answer's summary loses what it answered, and the question cannot be shortened at all. Together, the opening line names the subject in the user's own words and the summary says what was settled.
- **It is how a conversation is returned to.** A user comes back to "the seat-hold question", not to turn 5. Relevance compares what they ask now with whole exchanges, and raises a whole exchange: question, answer and all.
- **It is always safe to quote back.** The provider requires every tool result to follow its tool call immediately. A span never separates them, so any span can be shown word for word without breaking the request.
- **Fewer, better summaries.** One per exchange rather than one per message: fewer model calls, and each summary reads as one story.
- **Fixed boundaries make every later choice free.** A span's boundaries never change, so its summary, facts and opening line are made once. Raising or lowering it later only changes how it is rendered.

A collection never cuts between them either: it cuts only where a span may begin (see [the-cut.md](the-cut.md#why-this-shape)).

### The precise rule

```
 user ─ assistant ─ (tool call ─ tool result)* ─ assistant │ user ─ assistant │ …
 └──────────────────── one span ──────────────────────────┘ └── next span ───┘
```

A tool result arrives as a user-role message, but it continues the exchange; it does not start a new one. So does a subagent's relayed report: Claude Code adds a background subagent's findings to the main conversation under the user role, and the report joins the exchange that started the subagent. So does a turn Claude Code wrote all of, such as the notice that a background task finished. Only a turn the person wrote opens an exchange, including one Claude Code wrote part of, such as a skill command and the instructions it expands to (`proxy/overview.md`), but not one that carries tool results. What a person types after interrupting a tool round comes in the turn carrying the round's results: those words are theirs — what they are discussing now, and the source of their facts — but the turn opens no exchange, because a tool result stays in the span of the call it answers. Whose words a turn holds and where an exchange opens are decided apart.

```
 user ─ assistant → Agent ×2 ─ (tool result: "started") ─ assistant … ─ report ─ assistant ─ report ─ assistant
 └──────────────── one exchange: the question, the subagents' reports, the answers to them ──────────────┘
``` An exchange larger than 8,000 tokens (a long agentic
run) is divided between tool rounds, never between a tool call and its result,
and never between a question and its first answer. See [the-cut.md](the-cut.md)
for when a collection makes spans.

---

## The four detail levels

Each level shows everything the level below it shows, so **showing a span with
more detail never takes anything away**.

```
 kept         [Quoted exactly] (turn 6, user) Let's use Stripe for payments, with a 4% fee… [Quoted exactly] (turn 7, assistant) Agreed. Stripe takes the fee… ▲ more detail summarised   [Summarised] (turns 6–7) Let's use Stripe for payments, with a 4% fee. · fee: 4% The user chose Stripe with a 4% fee; the assistant agreed, noting… │ archived     [Archived] (turns 6–7) Let's use Stripe for payments, with a 4% fee. · fee: 4% │ out of view  [Out of view] (turns 0–3) 2 earlier exchanges, folded…      ▼ fewer tokens
```

Kept, summarised and archived are chosen by the score. **Out of view is not:** it
is the floor that keeps the front bounded in a conversation so long that even one
archived line per span no longer fits the budget. The oldest spans are then folded
behind a single line that says they exist; each still keeps its turns. Ordinary
sessions never reach it. How a level is chosen, what each costs, and why folding
works this way are in [detail-levels.md](detail-levels.md#out-of-view-the-floor-under-every-other-level).

---

## How a span's level is chosen

**The score decides the level; the budget caps the total.**

```
 every span ──► SCORE: how much does it matter now? liveness   = the larger of  recency (slow decay) relevance to what the user is discussing + importance = decisions and specifics it states │ ▼ PLAN:  score ≥ 0.7 → kept · ≥ 0.3 → summarised · below → archived (a span moves only once its score clears a line by 0.05) │ ▼ over the recap budget?  lower the coldest span one level, repeat; when nothing can go lower, fold the oldest
```

- **Liveness** is the larger of recency and relevance, not their sum: a span is
  part of the conversation as it stands if it is recent **or** if what the user is
  discussing reaches it. This is what lets an old span come back. See
  [scoring/overview.md](../scoring/overview.md).
- **The recap budget** is one share of the model's window (30% by default). The
  score is deliberately conservative — it archives only what has gone quiet — and
  the budget is what bounds the front.
- **Levels are re-decided at every collection, both ways.** A span drops as it
  goes quiet and rises when the conversation returns to it:

```
                    collection 3   collection 6   collection 9   collection 12
 span D relevance       0.55           0.25           0.20       0.80  ← the user asks about it span D level         summarised     archived       archived     kept  ← restored from its own words
```

---

## How the context is assembled

The context sent to the model is **the front, then the verbatim tail**:

```
┌──────────────────────────────────────────────────────┐ │ System prompt (the client's, unchanged)              │ │                                                      │ │ FRONT — one user-role message, present after a cut   │ │   a header naming the levels                         │ │   every span, in conversation order, at its level    │ │                                                      │ │ TAIL — every later turn, exactly as the client sent  │ │                                                      │ │ The new turn (unchanged)                             │ └──────────────────────────────────────────────────────┘
```

The front changes only when a collection commits; between collections its bytes
are identical on every request, which lets the provider cache it and keeps the
model's thinking blocks valid. It opens the conversation as a user turn, which the
provider requires.

**The front says what it is.** Its header reads: *"Earlier turns, condensed by
LLM-GC: quoted exactly, summarised, archived to an opening line and its facts, or
out of view. If a detail you need is not shown, say so rather than guess."* Every
span's line carries its marker and the turns it stands for. Condensed content
presented as original conversation would be the dishonesty the markers exist to
prevent, and the closing instruction turns a missing detail into an honest "I don't
have that" instead of a confident guess.

---

## Why nothing is lost

```
 a detail stated at turn 7 │ ├─ the span holding turn 7 keeps turn 7's words, always ├─ its facts are read from turn 7's words, once └─ its level can rise again: kept shows the words themselves
```

Three properties together make the no-loss promise hold:

1. **Originals are kept.** A turn leaves the young generation only into a span that
   holds it as it was recorded; committing a collection refuses any plan in which a
   departing turn would not land in exactly one span.
2. **Levels are nested.** Raising a span never shows less than before.
3. **Levels are reversible.** Moving up restores from the original turns or the
   cached summary, never from anything derived from a lower level.

What a level below *kept* loses is **detail in the prompt**, not information: the
span can always be shown in full again. On the decision-survival check, every
probe answer reached the model on all five conversations, 37 of 37, matching the
whole conversation sent unchanged ([evidence](../../../evals/decision_survival/evidence/detail-levels-2026-10-06.json)).

---

## The permanent generation: what was decided, and what replaced it

Every turn's facts are read when its span is made. They are filed in the
permanent generation, where a newer fact on the same topic **supersedes** the
older one instead of sitting beside it:

```
Turn 5:  "Let's use PostgreSQL" → KnowledgeEntry(topic="database", content="postgresql", message_turn=5, ACTIVE)

Turn 15: "Actually, switch to MySQL" → KnowledgeEntry(topic="database", content="mysql", message_turn=15, ACTIVE) → turn 5's entry is replaced by a copy marked SUPERSEDED
```

- Superseded entries are kept: the record of what a conversation contradicted is
  part of what the transparency layer reports.
- An archived or summarised span shows its **active** facts on its line.
- A span whose fact was later replaced carries a note at every level it is shown —
  `↳ Replaced later (turn 15): database: mysql, not postgresql.` — because its own
  words, or its opening line, may still state the old decision.
- An entry whose topic is "general" (a decision that names no topic) supersedes
  nothing: matched as a topic, "we decided to ship on Friday" would retire every
  earlier topic-less decision, however unrelated.

The entry model is in [knowledge-entry-model.md](knowledge-entry-model.md).

---

## Hallucination risk assessment

Condensing a conversation introduces three ways for the model to be told something
false or incomplete.

### 1. Lossy summarisation

A summary of "we compared PostgreSQL and MySQL across three turns, weighing ACID
guarantees, hosting cost and team familiarity, and chose PostgreSQL because the
team knows it" can lose the *reasoning*, and the model may later invent a
different one.

**Mitigation:** a summarised span also shows its opening line — the user's own
sentence — and its facts, so the decision itself does not depend on the summary.
When the conversation returns to the span, it is raised to kept and the model
reads the original words.

### 2. Fact-reading errors

A reader can file a false fact — "we decided NOT to use Redis" read as "Redis" — or,
worse, mark a value replaced when it was not. Either is shown to the model as settled,
beside its span.

**How facts are read.** By the model that writes the summaries, once per collection:
one call reads every span the collection makes, given the facts already on record as
`topic: value`.

```
 facts on record                     the collection's turns cloud: Google Cloud                 [turn 8] user: Change of plan: we're moving to AWS. payments: Stripe                    [turn 9] assistant: Done — the plan now assumes AWS. └──────────────┬──────────────┘ ▼  one call {"facts": [{"turn": 8, "topic": "cloud", "value": "AWS"}]} ▼  filed under the topic on record cloud: AWS        ↳ the span stating Google Cloud now carries "Replaced later (turn 8)"
```

- **A change is filed under the topic it changes**, so filing it marks the old value
  replaced. A turn that only repeats or explains a fact on record is left out, and a
  fact restating its topic's active value is never filed: it would read "MySQL 8, not
  MySQL 8".
- **Only the user changes a decision the user made.** Every fact records who stated
  it, taken from its turn rather than from the model. A value the assistant states on
  a topic the user decided is not filed; the recommendation stays in its exchange's
  summary. Measured on the check, twice: the user chose 9 shards and the assistant
  recommended 3. Filed as a replacement, the recap's "Replaced later" note made the
  model answer 3. Filed beside the decision, "index shards: recommended 3" was the
  only labelled note — the user's 9 sat inside a quoted sentence — and the model
  still led with 3. And when the user had asked "is that reasonable?", the reader
  took their 9 for a question, filed nothing for it, and filed the recommendation
  under another name, "shard layout", where the rule could not see it. The reader is
  now told that a user who asks about their own choice has still made it, and to
  file a recommendation that disagrees with a value the user stated under exactly
  that value's topic. The rule is enforced when facts are filed, so it holds whatever
  the model writes, as long as the topics match. Recommendations on topics the user
  never decided are filed, as recommendations.
- **The answer is untrusted output.** It must be JSON in the expected shape, or the
  read fails and is reported; a fact naming a turn the reader was not shown, or that
  is not a short topic and value, is dropped. Facts are filed in the order they were
  said, whatever order the model lists them in: filing is sequential, and a later
  statement is what replaces an earlier one.
- **Only the person's words are read as theirs.** Claude Code can write into the person's message: a skill's instructions, a command's output (`proxy/overview.md`). The reader is shown such a turn as the person's words alone, because a fact's speaker comes from the turn the model cites, never from the model. Shown whole, a skill's "the database is MySQL" would be filed as the person's decision, and nothing the assistant said could replace it. A turn Claude Code wrote all of is shown under the label `client`, and its facts are the assistant's side. *Rejected: showing every part under its writer's label* — the model cites only a turn number, so a fact from Claude Code's part would still be filed as the person's. *Accepted cost:* a value only Claude Code wrote becomes a fact when a later turn settles on it, as a value in tool output does.
- **No text in the read can forge a turn.** The turns and the facts on record are escaped the way a summary's transcript is (`compaction.md`), so a tool result that quotes the closing marker, then a line reading `[turn 3] user: …`, stays the tool result's text. It matters more here than in a summary: a fact's speaker comes from the turn the model cites, and a fact the user stated cannot be replaced by anything the assistant says.
- **The read has room to think.** Its output cap is the list's room plus a large
  allowance for the thinking the model does before it, and its timeout lets the cap
  be reached. An agent's turns carry whole files and command output: on a Claude
  Code session, a 5,000-token collection spent all of a 3,500-token cap thinking and
  returned no list. A cap is billed only as used, while an answer cut off before it
  starts is paid for in full and returns nothing. Turns are read in calls of at most
  one span's size, because the thinking grows with what is read. *Accepted cost:* a
  collection can take a couple of minutes; it runs off the request path, and
  requests meanwhile go whole, never worse than without LLM-GC.
- **A failed read** makes the collection's spans without facts, marks them, and is
  reported once; the dashboard names the messages that have no facts for that
  reason, instead of showing an empty list as if they settled nothing. They are not
  read again by a later collection: filing follows the order facts are read in, so
  an old exchange read after a newer one would replace the newer value. *Revisit if*
  unread spans show up once reads have room to think; reading them again needs
  filing by turn order.

Without a model, patterns read the user's own turns only — see the rules below.
Each reader declares whose turns it reads, so a side left unread is reported as
unread, never as a side that settled nothing.

#### Who can change a decision, in plain words

Beside each condensed exchange, the recap carries short notes like `database:
Postgres`, and when a later message changes one, a warning beside the old value:
"↳ Replaced later: database: MySQL, not Postgres". That warning is the strongest
thing LLM-GC tells the model, so it must only ever say something that happened. The
one rule: **only you change a decision you made.** The assistant can suggest; its
suggestion stays in its exchange's summary, where the model reads it in context, and
never becomes a note on a topic you decided.

```
 you:   "Let's use 9 shards."               → note: index shards: 9   (you) AI:    "I'd recommend 3 instead."          → no note: a suggestion against your decision stays in this exchange's summary you:   (never replied about it) recap: index shards: 9, and no "Replaced later" — as in the conversation itself
```

| What happens | What the model is told |
|---|---|
| You decide 9; the AI suggests 3; you never reply | your 9 is the note; the suggestion is in that exchange's summary |
| You ask "let's use 9 — is that reasonable?"; the AI suggests 3 | the same: asking about your choice does not undo it |
| You decide 9; the AI suggests 3; you say "ok, go with 3" | your new decision replaces your 9: "Replaced later: index shards: 3, not 9" |
| You decide 9; the AI disagrees; you say "no, keep 9" | your 9 stays; repeating it is not a change, so no note |
| The AI proposes Postgres; you say nothing | "database: recommended Postgres" — a suggestion, and it reads as one |
| The AI proposes Postgres; you say "sounds good" | your agreement is your decision: "database: Postgres" |
| The AI changes its own earlier suggestion | its new suggestion replaces its old one |
| You said "use pytest"; the AI later switched to unittest on its own | your "pytest" stays the note; the switch is in that exchange's summary |

In every case the exchange's own words are still there — quoted, summarised, or as
its opening line — and the notes are hints on top. When LLM-GC is unsure, it adds
no note: a missing hint costs a little, and a false one cost an answer on the check.
*Accepted cost:* the last row. When the assistant itself changes something you
decided, as an agent acting on code can, your decision stays the note and the change
is only in the summary, so the model can take the note as current. *Revisit if* the
Claude Code sessions show that case often: the reader would then mark what the
assistant *did*, as distinct from what it suggested, and only the first would replace
your decision.


**Why the model reads them.** Patterns over every turn were measured on the check:
they captured the answer to 8 of 37 probes and missed four of five changes of plan
("we're switching the index to OpenSearch", "payments go through Adyen now"). On one
conversation they filed 47 of 50 facts from the assistant's explanations, mostly
fragments ("costs: real but modest at this size"), and three of those "replaced" one
another, each putting a false "Replaced later" note in the recap. A change of plan
can be worded any number of ways; patterns chasing them never catch up.

**Why one call per collection, not inside each summary.** A short exchange — "Change
of plan: we're moving to AWS." "Done." — is never summarised, because a summary could
not be shorter; yet it is exactly the one that must replace a value. One call over
all the collection's spans reads it, and is shown the facts on record once. *Accepted
cost:* one more model call per collection, reading its turns again; collections are
infrequent and off the request path.

**What still guards against a wrong fact:** every fact names its turn, and the turn
is kept; the dashboard shows each replacement as the old value and the new one, so a
wrong one is plain; and a later correction replaces it.

The patterns, for a deployment without a model, each trading recall for precision
deliberately:

- **A subject is one to four words.** Without a bound, a topic grows until it
  happens to hit a full stop.
- **A declarative subject sits at the start of its sentence.** A matching verb
  further in belongs to a subordinate clause.
- **Matching is per sentence, not per turn.**
- **A subject made only of pronouns, articles or interrogatives is rejected.** "it
  is fine" states nothing retrievable later.
- **Openers and labels are not part of the sentence.** "OK,", "So," and a short
  label before a colon ("Decision:") are removed first; a qualifier is not an
  opener: "In production, the database is postgres" is refused rather than filed
  without its qualifier.
- **Closing punctuation is removed before matching.**
- **A question claims nothing.** "The database is postgres, right?" asks for
  confirmation; filed as a fact, it would supersede the answer.
- **A trailing topic names one thing.** When the content before "for the …" lists
  several things, the decision is filed as general.

Labels are normalised — lowercased, whitespace collapsed, a leading article or
possessive dropped — because supersession compares them exactly.

Patterns read only the user's own words: there they measured right — a choice, a
name, a date — while from the assistant's explanations they read mostly fragments.

### 3. Decontextualised knowledge

"If we were building in Go, we'd use PostgreSQL" is hypothetical; stripped of
context it becomes false certainty.

**Mitigation:** the reader is told that options only considered, hypotheticals and
advice nobody took up are not facts, and patterns fire only on decision language
("let's use", "decided to"); every fact names the turn it came from, and the turn
itself is kept.

### The epistemic transparency defence

All three risks are why the dashboard exists, and why the front's header asks the
model to say when a detail is not shown. The system cannot guarantee that
condensing never misleads — but it can make its own state **visible**, to the
person and to the model. The AI provides transparency; the human provides
judgement.

---

## Who owns the generations

**Each session owns its three generations; nothing else holds them.** A session's
tail, spans and facts live together in one immutable snapshot on its
`SessionState` (see [session/overview.md](../session/overview.md)).
`GenerationalMemory` is a stateless producer shared by every session: it groups,
summarises and reads turns, and stores nothing. A collection gathers what it
produced into a delta and commits it to the session's snapshot.

This is what keeps one conversation's memory out of another's prompt: a producer
that held what it produced would hand one user's facts to the next.

---

## Events emitted

| Event | When | Data |
|-------|------|------|
| `MESSAGE_PROMOTED_TO_OLD_GEN` | A new span was made at the summarised level | session id, the span's turns, tokens before / after |
| `MESSAGE_ARCHIVED` | A new span was made archived, or an older span was lowered to archived or out of view | session id, the span's turns, how many facts were filed from them, `young` or `old` |
| `KNOWLEDGE_ENTRY_ADDED` | A fact was filed | session id, its turn, its kind, its status, who stated it |
| `KNOWLEDGE_ENTRY_SUPERSEDED` | A new fact retired active ones on its topic | session id, the new fact's turn, the turns of the facts it retired |
| `AGING_DEGRADED` | Part of a collection fell short | session id, the degradation |
| `GC_FINISHED` | Every pass, whatever it did | session id, the pass's report |

All but `AGING_DEGRADED` and `GC_FINISHED` are emitted **after** the commit, so
each describes a change now in the session's state. Where each turn and fact
stands *now* is read from the session's state, not from events — see
[visibility/overview.md](../visibility/overview.md).

No payload carries what was said, only turns, counts and named values; why is in [events/overview.md](../events/overview.md#what-a-payload-carries).

---

## Where it fits: hot path and cold path

```
COLD PATH · gc_age(session)                                after the response ═════════════════════════════════════════════════════════════════════════════ turns older than the tail │ ▼  draft spans (grouping, facts, opening lines — no model call) ▼  score every span, plan every level, estimate the saving │      not worth a collection? stop here, having paid for nothing ▼  summarise the new spans (model calls) ▼  plan again with the real summaries ▼  the cut pays for itself? commit, onto the CURRENT snapshot

HOT PATH · gc_collect(session)                             before forwarding ═════════════════════════════════════════════════════════════════════════════ read the session's snapshot once — no lock, no scoring, no model call [ FRONT ]  +  [ TAIL ]  ──►  LLM
```

**The cold path** runs after a response, off the request's critical path. It
decides whether a collection is worth making, plans it against the snapshot it
started from, and commits in one synchronous step onto whatever the snapshot is by
then, so turns recorded during a slow summary are kept. At most one pass runs per
session.

**The hot path** runs before the request is forwarded. It returns the front and the
tail from the current snapshot and never waits for a pass: if one is in flight, it
returns the context from before it, which is complete.

---

## Design decisions

| Decision | Rationale |
|----------|-----------|
| Keep every original turn | Nothing can be restored that was not kept. The proxy receives the originals with every request anyway; the session keeps them so the library has them too. Accepted cost: the originals are held in memory for the session's life. |
| Spans are exchanges | A question and its answer belong together: the question names what the answer is about, and is the span's opening line. Tool rounds are never split. |
| Four nested levels | Each level shows everything the level below shows, so raising a span never loses information. Measured: when the summarised level showed only the summary, the check's no-summary rehearsal kept 9 of 37 answers; nested, 24 of 37. |
| The score decides the level, the budget caps the total | A score that also decided *when* to drop detail would drop it on a schedule, whether or not the front needed the room. |
| Levels re-decided at every collection, both ways | A guess about what will matter later cannot be made reliably once; re-deciding corrects it. |
| A margin around every level line | A score hovering at a line would otherwise show a span in full at one collection and as facts at the next. |
| Facts read from the original turns, once per collection | A fact read from a summary is read second-hand. Reading when the spans are made means choosing a level later never waits on a model. |
| Facts read by the model that summarises; patterns, on the user's words, without one | Measured: patterns captured 8 of 37 answers and invented replacements from sentence fragments; the model files a change of plan under the topic it changes. |
| Supersede, don't delete | The decision chain is part of what transparency reports. |
| Collections on the cold path, assembly on the hot path | Summaries are the expensive work; the hot path never waits for them. |

---

## Alternatives considered

- **Moving turns one way down fixed tiers** (verbatim → summary → facts, never
  back). This was the earlier design. A turn's tier was decided once, by a score
  that was mostly its age at the moment of the cut, and a summary was archived into
  facts when that score decayed. Measured on the decision-survival check, it lost 7
  of 37 answers: each was a detail inside a summary that had been archived and then
  sent cut to 180 characters. Nothing could bring the detail back, because the
  original had not been kept. Replaced by kept originals and reversible, nested
  levels.
- **Deciding the form of a turn once, by score bands on each turn.** Measured on
  the check's conversations, no collected turn could ever score into the top band
  (recency, weighted on its own, capped the score at 0.68), and whether a turn was
  summarised or reduced to facts depended about as much on *when* the collection
  ran as on what the turn said. Replaced by span scoring against what is being
  discussed now, re-decided at every collection.
- **Separate budgets for facts and for summaries, with facts in their own block.**
  Two limits on one resource: the tighter one silently decides, and nobody tunes
  the one they forgot. Replaced by one recap budget, with facts on their span's line.
- **Keeping text extraction could not read as a raw entry.** It existed so archiving
  lost nothing, but it was sent cut to a fixed length, ranked last, and duplicated
  what the span now holds whole. Removed: the span keeps the turn itself.
- **Re-summarising old summaries into longer-horizon ones.** Summarising a summary
  compounds loss, and each pass costs a model call. Folding the oldest spans behind
  one line keeps the front bounded without rewriting anything.
- **A model that fetches old turns itself, through a recall tool.** It would need
  model round-trips inside the user's request, beside the client's own tools.
- **Retrieving old turns into every request.** The prompt would change on every
  request, resetting the provider's cache and the model's thinking blocks each time.
- **Filling the budget with as much detail as fits.** Nothing would be condensed
  until the history outgrew the budget — a capacity trigger, which is not what
  LLM-GC is: savings come from the score.
- **Facts matched by patterns over every turn.** Free and deterministic, but measured
  on the check they captured 8 of 37 answers, missed four of five changes of plan, and
  filed fragments of the assistant's explanations as facts that then replaced one
  another. Replaced by facts the model reads, with patterns kept for the user's own
  words when there is no model.
- **Facts written in the same call as each summary.** No extra call, but an exchange
  too short to summarise is never read, and a short "change of plan" exchange is the
  one that must replace a value.
- **Accumulating contradictions instead of superseding.** "We chose Postgres" and
  "we chose MySQL" with equal weight would leave the model to resolve the
  contradiction itself.

Supersession carries an accepted cost: it treats the latest contradicting entry
as current, which is right most of the time but not always — a user can misspeak
in a way the reader takes for a real change. The system accepts being
occasionally wrong about which fact is current rather than interrupting the user,
and the original words remain one level up.
