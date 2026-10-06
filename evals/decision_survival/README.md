# The decision-survival check

**Does a fact a conversation settled early still reach the model after LLM-GC has
cleaned the conversation up?** This check answers that with numbers, so a claim of
"smaller prompts without quality loss" points at a method, a dataset and a result
anyone can rerun, rather than at an impression from watching the dashboard.

It is also the instrument for changing LLM-GC: a new default, a different way of
handling text extraction cannot read, or a smaller front is kept only if it does
not cost facts this check can see.

## What a run does

Each scripted conversation is played through the real proxy, running the same
engine `run_proxy.py` serves (`build_gc_service`). Then the conversation's questions
are asked twice: once through LLM-GC, once with the whole conversation sent
straight to the model.

```
                     scripted conversation (every turn written in advance)
                                        │
        ┌───────────────────────────────┴──────────────────────────────┐
        ▼                                                              │
  proxy, in-process ── each scripted turn ──► canned local reply       │
  (cuts happen here,     (no model call; the script supplies          │
   as in production)      the assistant's turn)                        │
        │                                                              │
        ▼                                                              ▼
  the questions ──► model, as LLM-GC sent it          the questions ──► model, whole conversation
        │           (front + recent turns)                    │         (the control)
        ▼                                                     ▼
  answers + where each answer sat in the prompt          answers
        └────────────────────────── graded per question ─────────┘
```

- **The engine is the proxy's.** A check that wired its own engine would measure
  something nobody runs.
- **Only the questions reach a model**, plus LLM-GC's own summary calls at each
  cut. Scripted turns are answered locally and their reply discarded, so a run is
  cheap and every run sees exactly the same conversation.
- **Aging finishes before the next turn.** In production it runs after each
  response while the user reads; here the harness waits for it, so each request
  sees the state the previous one left, and the run is repeatable.
- **The control arm sends the identical request** (same model, effort, messages)
  without the proxy. Without it, a question the model gets wrong even with the
  whole conversation would be blamed on LLM-GC.

## The dataset

`conversations.json` holds five design conversations. Three are short (an ingest
service, clinic booking, a fraud model): 18 turns of answers around 160 words. They
carry the same six kinds of fact, because each kind fails in its own way:

| Kind | Example | Why it is hard |
|---|---|---|
| A decision later changed | Postgres, then "change of plan: MySQL" | The old value must not come back; `reject` lists it |
| A stated value | "the retention window is 30 days" | A requirement, not phrased as a decision |
| A second fact in the same message | "…also the API must answer within 200 ms" | Extraction or a summary can keep one and drop the other |
| A number inside a choice | "let's use Kafka … with 12 partitions" | The choice is easy to keep, the number easy to lose |
| A person | "the team lead is Priya Raman" | A name, with nothing to score it as important |
| A decided date | "we decided to ship on November 14" | Phrased as a decision, so it is the best-protected kind |

The kinds sit at **different turns in different conversations** (the change of
mind comes at turn 10, 6 and 14), so a result for a kind is not just a result for
an age.

**Two long conversations** (event ticketing, product search), 30 turns each, add
what the short ones cannot reach. Their assistant answers run to 1,000 words or
more, as real ones do, so a second collection archives what the first one kept or
compacted, the path by which a live session put the user's own requirements out of
view. They add two kinds of fact: **a detail only the assistant stated** ("hold a
selected seat for 7 minutes"), and **a requirement phrased as a constraint** ("must
respond in under 300 ms"), which extraction cannot read.

Their user turns are written by hand; their assistant turns were written once by a
model with `author.py` and frozen into the dataset. Each answer was checked as it
was written: one that left out the figure it had to state, or mentioned another
question's answer, was asked for again. Re-running `author.py` writes new text, so
results are only comparable across runs made on the same frozen dataset.

Rules every conversation follows, enforced by the loader and the tests:
- turns alternate user and assistant and end with the assistant;
- **each accepted answer appears in exactly one turn**, the one the probe names,
  so an answer the model gives can only have come from that turn. The one
  exception is named in the probe's `also_in`: the reply to a change of mind may
  repeat the new value, as any real reply to "we're moving to AWS" would;
- no question contains its own answer;
- the content is synthetic. Results files store the prompt and replies, so a
  real conversation must never be added.

## Grading

All of a conversation's questions go in **one** message, numbered, asking for the
current value in a few words, or "unknown". One message, so every question sees the
same prompt: asked one at a time, each request could trigger a clean-up before the
next.

Each answer gets one verdict, by whole-word matching against the probe's `accept`
and `reject` lists. Parenthetical asides are read as history, not as a second
answer: "MySQL 8 (changed from Postgres 16)" is right, while "MySQL or Postgres"
is unclear. Models add such asides even when asked for the current value only.

| Verdict | Meaning |
|---|---|
| right | names an accepted value |
| wrong | names something else, or only a replaced value |
| unknown | says the conversation does not say |
| unclear | names the current value *and* a replaced one |
| missing | no numbered line for this question |

And each probe records **where its answer was in the prompt LLM-GC sent**:
`verbatim` (in a turn sent word for word), `condensed` (only in the block LLM-GC
puts at the start), or `absent`.

## Reading the numbers

- **Lost to LLM-GC** is the headline: right with the whole conversation, not right
  through LLM-GC. It is the quality cost of the saving.
- **Absent from the prompt** says *why* most losses happen: the fact did not
  survive the clean-up at all. Fixing these is an engine change (what is kept,
  quoted, summarised).
- **Present but missed** is a different problem: the fact was in the prompt and the
  model still answered wrong, so the fix is in how the condensed block presents
  it, not in what is kept.
- **Prompt tokens** come from the provider's own count on the two question
  requests, so the saving is measured, not estimated.
- Warnings are printed and stored for any clean-up that degraded (a refused or
  failed summary), any request the proxy forwarded without GC, and any reply that
  did not finish normally. A run with warnings measured less than it claims.

## Running it

```
python -m evals.decision_survival.run --rehearse    # offline: no key, no cost
ANTHROPIC_API_KEY=... python -m evals.decision_survival.run
```

`--rehearse` replaces every model with an offline stand-in: summaries that state
nothing, and a "model" that answers right exactly when the answer is in what it
was sent. It checks the whole path and shows what survives *without* summaries.
It says nothing about a real model, and its numbers are never evidence.

A live run asks the questions with `claude-sonnet-5-5` at effort `medium` (change
them with `--model`, `--effort`; keep them fixed across runs you compare), and
summarises with LLM-GC's own compactor model. The engine settings that tuning changes
are flags, `--min-cut-tokens` (300) and `--keep-last` (5), so two runs differ only in
what is being tuned. A full run takes about five minutes; its cost is mostly the long
conversations, each about 50,000 tokens when sent whole, plus the summary calls at
their collections.

Add `--dashboard` to watch the run on its own dashboard, at http://127.0.0.1:9911,
separate from the proxy's. Each conversation is a session; the page stays open after
the report, so you can see where each question's answer ended up: kept, compacted,
archived, or out of view.

Every run writes a results file to `runs/` (not committed): settings, the commit
measured (marked `-dirty` when uncommitted changes were part of it), each verdict
and its answer, both replies, and the prompt LLM-GC sent.

**Evidence.** A run cited anywhere, in this file or a design document, is copied
to `evidence/` under a descriptive name and committed with the change that cites
it. 

| Run | Short conversations | Long conversations | What it shows |
|---|---|---|---|
| [size-budget-2026-10-05](evidence/size-budget-2026-10-05.json) | 18/18, prompts 17–24% smaller | 12/19, prompts 64% smaller | The recap's facts limited by size alone. On the long conversations every fact the user stated survives (11/11), and 1 of 8 details stated only inside an assistant answer does: each of the seven was in an excerpt of an archived summary, sent only up to its first 180 characters. |
| [detail-levels-2026-10-06](evidence/detail-levels-2026-10-06.json) | 18/18, prompts 11–15% smaller | 19/19, prompts 42% smaller | Every span kept with its original turns and shown at the level its score earns, levels nested. Every answer reaches the model; summaries are 95% of the front on the long conversations, so the saving is the summaries' size - tuning the summary target and how readily old spans are archived is what can win it back. |

Questions asked by `claude-sonnet-5-5` at effort `medium`, `min_cut_tokens 300`,
`keep_last 5`.

## What it does not tell you

- **Small and single-sample.** 37 probes, one answer each. Models and summaries
  vary from run to run, so a difference of one or two probes between two runs is
  not a result. Repeat both runs before acting on a close one.
- **Clean conversations.** Written to be clear, with every fact stated once and
  plainly. Real conversations restate, hedge and contradict themselves; this check
  is a floor for those, not a forecast.
- **Text only.** No tool calls, images or thinking blocks in the history; those
  paths are tested elsewhere and need real traffic.
- **Recall, not reasoning.** It asks for values. A model can recall every fact and
  still reason worse from a condensed history; this check would not see that.
- **String matching.** A right answer in unexpected words ("two days" for "48
  hours" is accepted; "a couple of days" is not) grades as wrong. Read the stored
  answers before trusting a surprising loss.
