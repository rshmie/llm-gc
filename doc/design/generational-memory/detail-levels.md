# Detail Levels: How Each Span Is Shown

## How it works

Every span in the old generation is shown in the front at one of four **detail levels**. A level is chosen by the span's score, capped by the recap's budget, and chosen again at every collection.

### The four levels

Each level shows everything the level below it shows. **Showing a span with more detail never takes anything away.**

```
 level          what the model reads                                              made when
 ─────────────  ────────────────────────────────────────────────────────────────  ──────────────────
 kept           [Quoted exactly] (turn 6, user) every word of the user's turn     always there
                [Quoted exactly] (turn 7, assistant) every word of the reply
                                                                                  ▲ more detail
 summarised     [Summarised] (turns 6–7) the opening line · its facts              once, when the
                the summary                                                         span is made
                                                                                  │
 archived       [Archived] (turns 6–7) the opening line · its facts                once, when the
                                                                                    span is made
                                                                                  │
 out of view    [Out of view] (turns 0–13) 8 earlier exchanges, folded…           —
                                                                                  ▼ fewer tokens
```

- **The opening line** is the first sentence the user wrote in the exchange — what was asked, or what was decided, in their own words.
- **Its facts** are the active facts read from the span's turns, as `topic: value`.
- **A folded span** shares one line with every neighbouring folded span; see [Out of view](#out-of-view-the-floor-under-every-other-level) below.
- **A span whose fact was later replaced** carries one more line at every level it is shown: `↳ Replaced later (turn 22): cloud: AWS, not Google Cloud.` Its words, or its opening line, may still state the old decision; the note stops the model reading it as current.

### Out of view: the floor under every other level

Kept, summarised and archived are chosen by the score. **Out of view is not.** It is what happens when a conversation has grown so long that even the cheapest showing of every span no longer fits the recap's budget.

**What it is.** A span out of view is still stored — its turns, its summary, its facts — but the front no longer gives it a line of its own. A run of the oldest such spans shares one line, which says only that they exist:

```
[Out of view] (turns 0–41) 21 earlier exchanges, folded to stay within the budget.
```

The model learns that those turns happened and that it cannot see them, but not what was said in them. Their facts are not sent either.

**Why it has to exist.** Every other level costs something for each span. Even archived, an opening line and its facts, costs 20–40 tokens, and a conversation can keep growing. Without a floor, the front would grow with the conversation until it filled the window it exists to protect:

```
 exchanges in the old generation               100       1,000      3,000
 front, every span archived (~30 tokens each)   3,000    30,000     90,000  ← over the
                                                                              60,000 budget
 front, with the oldest folded                  3,000    30,000     60,000  ← bounded
```

(The default budget is 30% of the window: 60,000 tokens of a 200k window.) The folded line costs about 25 tokens whether it stands for two exchanges or two thousand. It is the one level whose cost does not grow with what it covers, and that is what makes the budget a hard bound however long the conversation runs.

**When it happens.** Last, and only under budget pressure:

```
 1  every span gets the level its score earns
 2  over budget?  the coldest span loses one level, again and again —
                  warm spans too, once the cold ones are at their floor
 3  every span already at its cheapest level, still over budget?
                  fold the oldest spans, one at a time, until it fits   ← only here
```

At the default budget that takes around two thousand exchanges, all of them already archived. An ordinary session never folds anything; one that does has a budget too small for it, or opening lines and facts too long.

**Why the oldest, not the coldest.** Folding the coldest would scatter many `[Out of view]` lines through the front, each costing tokens and each harder for the model to place. Folding the oldest keeps one contiguous run behind one line, and the conversation has moved furthest from it. *Accepted cost:* folding ignores the score, so a very old span the user asks about can be folded with its neighbours if the budget leaves no room at all.

**What it is not.**
- **Not deletion.** The span keeps its turns. At the next collection a folded span returns to whatever its score earns, and is folded again only if the budget still demands it.
- **Not silent.** The model is told the turns exist and that it cannot see them, and the front's header tells it to say so rather than guess when a detail it needs is not shown. The dashboard marks those turns *out of view* and their facts *stored, not sent*.

### A level is used only if it is cheaper than a richer one

The planner knows what each level of each span would cost, in tokens, as rendered. A level that costs as much as a richer one is never used:

```
 a long exchange    kept 4,210  >  summarised 640  >  archived 40      every level usable
 a short statement  kept 17     <  archived 18                         kept: its facts would
   "The database                                                       make the archived line
    is postgres."                                                      longer than its words
```

A span without a summary — too short for a summary to beat its words, or one whose summary failed — cannot be summarised; in that band it is kept rather than archived, so a failure never costs detail. The same holds for facts: a span whose facts could not be read is never archived, since archived it would be only its opening line. It is summarised if it has a summary and kept otherwise, and a span archived that way before is raised at the next collection.

### The planner

The planner is a pure function: scores, current levels and costs in, a level for every span out. No model call, no I/O.

```
 1  BAND       each span gets the level its score earns
                  score ≥ keep_threshold (0.7)     → kept
                  score ≥ archive_threshold (0.3)  → summarised
                  below                            → archived
               moving off its current level only once the score clears the line by
               level_margin (0.05) — up past 0.75 or 0.35, down below 0.65 or 0.25
                     │
 2  SETTLE     a level that is not cheaper than a richer one becomes the richer one
                     │
 3  FIT        while the spans cost more than the recap budget:
                  lower the coldest span one level (the older first on equal scores)
               the coldest goes all the way down before a warmer span loses detail
                     │
 4  FOLD       if every span is at its floor and it still does not fit:
                  fold the oldest spans, together, behind one line, until it fits
```

**The margin, across collections.** A score that hovers at a line does not flip the span back and forth:

```
 score        0.33   0.29   0.31   0.28   0.24
 level        summ.  summ.  summ.  summ.  archived      ← only once it falls below 0.25
```

Without the margin, the model would see the span's summary at one collection and only its opening line at the next, and its answers about it would flip with it.

**Reaching the budget never raises a level.** Savings come from the score; the budget only bounds the total. With room to spare, a span the score archives stays archived.

### The recap budget

One budget covers everything the front shows for its spans: `max_recap_ratio` of the window of the model the conversation's requests name (30% by default), less the front's own header. A share, never an absolute: 200 tokens is nothing to a 128k window and everything to a 200-token one. A session that has not learned its model's window uses the configured one.

**The share is taken of the window counted as the recap is.** The window is in the model's tokens; the recap is measured in LLM-GC's, and every model's tokenizer counts the same text its own way (on one Claude Code session, about 1.48 of Claude's tokens for each of LLM-GC's). Compared directly, a 30% setting would allow about 44% of that window, and a different share on every model. So the window is first converted at the **rate** the conversation's own replies measure: the provider's count of a request's whole prompt over LLM-GC's count of the same request, taken from the newest reply. Until a reply measures it, the window is used as it is.

```
the conversation's model     its window              rate      in LLM-GC's count   recap budget (30%)
claude-opus-5-5              1,000,000 (reported)    1.48          675,675              202,702
claude-haiku-4-5               200,000 (reported)    1.48          135,135               40,540
a name the gateway made up     200,000 (assumed)     not yet       200,000               60,000
```

*Rejected: a table of rates per model* — kept by hand, and wrong for a renamed or new model. *Rejected: counting with the provider's tokenizer* — Claude's is not public, and asking the provider to count is a network call per request. *Rejected: giving the engine the window already converted* — the dashboard's window meter divides the provider's count of a request by the window, both in the model's tokens, and would read too full; the engine keeps the window as the model states it and the rate beside it. *Accepted cost:* the rate is the whole prompt's, mostly the system prompt and tools on Claude Code, not the recap's own; prose and code may convert at somewhat different rates.

### The front, rendered

```
[Earlier turns, condensed by LLM-GC: quoted exactly, summarised, archived to an
 opening line and its facts, or out of view. If a detail you need is not shown,
 say so rather than guess.]

[Earlier conversation, in order]
[Out of view] (turns 0–5) 3 earlier exchanges, folded to stay within the budget.
[Archived] (turns 6–7) Let's use Stripe for payments, with a 4% fee. · fee: 4%
[Summarised] (turns 8–9) The product owner is Lena Hoffmann. · product owner: Lena Hoffmann
The assistant set out who approves customer-facing changes and how sign-off works…
[Quoted exactly] (turn 10, user) What should the seat hold be?
[Quoted exactly] (turn 11, assistant) Seven minutes…
```

The header is 45 tokens and names the levels in the words their markers use; each line's marker explains itself. The front owns every marker: a summary is rendered from what it says, not from the compactor's own wrapper.

## Why it exists

A condensed conversation has to answer two questions about every older part of it: how much of it does the model see now, and can it see more later? Detail levels answer both: the level is how much, and because the span keeps its words, any level can be raised again.

## Why this shape

- **Four levels, nested.** Kept, summarised and archived cover the range from every word to one line; out of view is the floor that keeps the front bounded however long the conversation runs. Nested, because an exchange's opening line is often the decision itself — a summarised level that dropped it would show less than the archived level below it. Measured with the check's rehearsal, whose summaries state nothing: 9 of 37 answers survived when the summarised level showed only the summary, 24 of 37 nested.
- **The score decides the level; the budget caps the total.** Each has one job. A score that also decided when to drop detail would drop it on a schedule; a budget that also raised levels would make LLM-GC a capacity trigger.
- **Bands with a margin, re-decided every collection.** Bands keep the rule readable — a person can see why a span is where it is. The margin keeps a span near a line steady. Re-deciding corrects yesterday's guess.
- **Coldest first, all the way down.** Protects the spans that matter most now; the older span goes first on a tie because the conversation has moved further from it.
- **Fold the oldest, contiguously.** One line names a whole folded run. Folding the coldest instead would scatter many short lines through the front.
- **Every band reachable at any age.** A test computes the best score an old span can reach and fails if a level is out of reach — the defect that once made quoting an important old turn impossible.

## Alternatives rejected

- **A sweeper classifying each turn once into keep, compact or archive.** This was the earlier design. The class was decided once, by a score dominated by the turn's age at the moment of the cut, and could not be revisited; an archived turn became facts and a raw excerpt cut to 180 characters. Replaced by levels that are re-decided every collection and can rise again.
- **Archiving old summaries whenever their score decayed.** Recency took every summary below the archive line within a few collections, so detail was dropped on a schedule with the window mostly empty — the cause of the 7 answers the check lost. Replaced by the budget deciding when anything must lose detail.
- **Separate budgets for summaries and for facts, with facts in a block of their own.** Two limits on one resource: the tighter one silently decides. Replaced by one budget, with facts on their span's line.
- **A size rule keeping every short message whole.** Replaced by comparing what each level actually costs, which keeps short spans whole for the right reason and nothing else.
- **Filling the budget with as much detail as fits.** Nothing would be condensed until the history outgrew the budget.
- **Folding the coldest spans instead of the oldest.** Many scattered lines, each costing tokens and harder for the model to place.

## Revisit if

- The check shows levels flipping across collections → widen `level_margin`.
- Ordinary sessions fold spans → opening lines or facts are too long, or the budget too small.
- Spans stay low after budget pressure eases → the margin is measured from the current level, so a lowered span needs its score to clear the next line by the margin to rise; a stored "earned level" per span would fix it.
