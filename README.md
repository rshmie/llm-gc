# LLM-GC: Garbage Collection for LLM Context

> **Status: work in progress.** The engine and an Anthropic-compatible proxy are built
> and tested — relevance scoring, three-tier classification, generational memory with
> fact extraction and supersession, a summarising compactor, collections that keep the
> provider's prompt cache and thinking blocks valid, streaming, and a live dashboard
> (`python scripts/run_proxy.py --serve --dashboard` → http://localhost:9901). A
> decision-survival check measures what the model can still answer after LLM-GC has
> condensed a conversation; its current results are under [Measured results](#measured-results).
>
> What is **not** done yet: the OpenAI-compatible endpoint, a verified run of Claude Code
> through the proxy, and keeping long assistant answers whole enough to survive — the
> check shows LLM-GC still losing most of their details. Nothing is pip-installable yet.

If you've spent 50+ turns in a Claude Code session, you've probably noticed the model starts forgetting things. The database you agreed on, the auth approach you picked, the thing you explicitly said *not* to do — it's all still in the conversation, but buried under 130K tokens of old file reads, stale command outputs, and resolved debugging context. The model either hallucinates a wrong answer or tells you "I don't have that context."

The root cause is that every API call resends the entire conversation history from scratch. Nothing is ever cleaned up, so old material piles up and crowds out the things that still matter.

## The problem

Two distinct things go wrong, and only one of them is about money.

**Context loss.** The decision from turn 5 is still technically present at turn 50 — it's just one short message inside 100K tokens of less relevant material. Liu et al. (2023), [*Lost in the Middle*](https://arxiv.org/abs/2307.03172), showed that information buried in the middle of a long context is recalled less reliably than information at either end. So the decision can be *in* the window and still not usable. Worse, a naive fix makes it worse: on real sessions, a sliding window that keeps the last 10 turns loses **7.6 of 8 early decisions**.

**Context waste.** Across 7 real Claude Code sessions, 99% of token spend was input — the resent history — and 84% of that input at peak was stale context. You are paying, on every single call, to resend material that is crowding out the material you need.

This is recognisably a memory-management problem, and memory management has decades of prior art. Java, Go and C# all identify what is still needed, discard what is not, and compact what remains.

LLM-GC applies that shape to the context window. It scores each turn for relevance, compresses what is aging, and extracts decisions into a permanent store rather than deleting them. **The goal is preserving what the model needs to remember — not saving tokens.** Savings are a byproduct of removing waste.

But cleaning context is only half of it. The other half is **transparency**. Today you have no signal that the model has lost your database decision, no indicator that context quality is degrading, and no warning before the wrong answer arrives. The model answers in exactly the same confident tone whether its context is clean or 84% noise. LLM-GC surfaces what is happening to your context so that *you* can judge when to trust the output — the system provides the data, the human provides the judgment.

## How it works

LLM-GC intercepts LLM API calls, optimizes the conversation context, and forwards the cleaned version. Three operations, inspired by classic GC algorithms:

1. **Mark** — Score every turn older than the newest few for relevance
2. **Sweep** — Classify each: **keep** (quote it whole), **compact** (summarise it), or **archive** (extract its facts)
3. **Compact** — Assemble the prompt: one recap message (quotes, summaries and facts), then the recent turns word for word; the system prompt is never touched

### Generational memory

Context is managed in three generations, inspired by generational garbage collection:

| Generation | What it holds | Analogy |
|---|---|---|
| **Young** | Recent turns in full detail | Java's young/eden space |
| **Old** | Older turns, quoted whole or summarised | Java's tenured generation |
| **Permanent** | Extracted facts and decisions, never discarded | Java's permanent generation |

As turns age, they move through generations: in full (young) → quoted or summarised (old) → facts extracted (permanent). The old generation and the facts travel together in the recap at the start of every request, so they reach the model; facts are sent up to a share of the model's window.

### Context Visualizer — epistemic transparency

AI models have no reliable internal sense of what they know and don't know. Getting a model to report its own uncertainty accurately is an open research problem (calibration), so you cannot lean on the model to tell you when its grounding is weak.

Humans, by contrast, do this naturally — **when they have the data**. Evaluating whether a source should be trusted is a cognitive skill the research literature calls **epistemic vigilance** (Sperber et al., 2010). LLM-GC's premise is to stop trying to automate the judgment and instead give the human what their own vigilance needs to work on.

Right now, developers have no data:
- You can't see that the model has effectively "forgotten" a decision you made 40 turns ago
- You can't see that the context is so bloated the model is statistically more likely to hallucinate
- You can't see which parts of the conversation are still "alive" in the model's attention vs. buried noise
- You have no signal telling you "this is the point where you should worry about answer quality"

LLM-GC's Context Visualizer solves this. Every GC component already computes rich internal state — relevance scores, generation classifications, waste percentages, decision tracking. Instead of using that data only internally, the visualizer **exposes it to the developer** as a live dashboard:

- **Named signals, no aggregate score** — token budget pressure, and what the last GC pass actually did (including when it was bypassed or failed — an unhealthy pass is visible, never hidden). Each signal is named for what it literally measures; nothing looks more authoritative than its math supports. The *transformation ratio* is computed but deliberately **not** displayed: it is honest arithmetic that would still be misread as a risk score if it sat on the page as the one number to glance at
- **Generation lifecycle & transitions timeline** — where every turn lives right now (young / old / permanent) and the last N promote / archive / supersede events — including the moment a newer fact contradicted and superseded an older one, which no current-state view can ever show
- **What the model knows, turn by turn** — every turn as it was said, beside what the model still has of it: word for word, quoted, inside a summary, as facts, or out of view — and every fact marked as sent to the model or only stored. A fact that is kept but never sent is the case this view exists to expose
- **Turn relevance** — each recent turn's score against the keep and archive thresholds, and what the next cut will make of it
- **Post-Session Analysis** *(planned)* — After a session ends: full timeline of when quality peaked, when it degraded, which decisions survived, where the danger zones were

There is deliberately **no single "health score" or "hallucination risk" number**. A number labelled "risk" claims an accuracy no heuristic can back without ground-truth hallucination labels — and a dishonest number is a new black box, which is the opposite of the point. If a learned predictor can be defensibly grounded in real labels, an aggregate ships *alongside* the signals; until then, the signals do the part that can be done honestly, and you do the part that can't yet be automated honestly. **Aggregation is earned, not assumed.** Most tools optimize context silently. LLM-GC shows you what's happening and lets YOU decide if the AI's output is trustworthy right now.

## The GC Analogy

LLM-GC borrows principles from garbage collection — a concept pioneered in Lisp (1959) and refined across decades of language runtime design in Java, Go, C#, and others. The generational model, mark-sweep-compact algorithm, and concurrent pre-computation are adapted from these well-established systems and applied to a new domain: LLM context management.

| GC Concept | LLM-GC Equivalent |
|---|---|
| Heap memory | Context window |
| Objects | Conversation turns |
| Live objects | Relevant context (still needed) |
| Dead objects | Stale context (safe to remove) |
| GC roots | Current query + active topic |
| Reachability analysis | Relevance scoring |
| Mark phase | Score each turn for relevance |
| Sweep phase | Remove low-relevance turns |
| Compaction | Summarize + consolidate remaining context |
| Young generation | Recent turns (full detail) |
| Old generation | Older turns (compressed) |
| Permanent generation | Key facts (never discarded) |
| GC pause | Latency overhead (design target: <25ms warm; not yet benchmarked) |
| Concurrent GC | Async pre-computation between turns |
| GC tuning flags | Configurable thresholds and strategies |

### Where the analogy breaks — and why that matters more than where it holds

A runtime garbage collector is **sound**. Reachability is decidable and exact: trace from the roots, and an object is either reachable or it is not. The collector never frees a live object. That guarantee is precisely why Java's GC can run silently — it has nothing to report, because it cannot be wrong.

Relevance is not decidable. The root set for a conversation is the *next question*, and it hasn't been asked yet. A turn that looks dead at turn 10 can be the most important thing in the conversation at turn 12, and no analysis at turn 10 can determine that. Every relevance decision is a bet against a future that doesn't exist yet.

Three things follow, and they shape the whole design:

- **Cleanup cannot be silent.** A system making unsound, unverifiable decisions about what the model is allowed to see — and hiding them from the person who gets handed the answer — has replaced one opaque actor with two. Transparency here isn't a feature chosen because it's appealing. It's the only honest way to ship an operation that can't be proven correct.
- **Removal has to be recoverable.** Because a bet can be wrong, what's swept must stay reachable in some form. This is why the system compresses rather than deletes: a turn leaving the young generation survives as a summary, and its decisions survive as extracted facts. A wrong bet costs fidelity, not knowledge.
- **Conservatism is calibration, not timidity.** Keeping an unneeded turn costs a bounded number of tokens. Removing a needed one costs an unpredictable amount and stays invisible until it surfaces as a wrong answer. Asymmetric costs demand an asymmetric policy.

### Beyond the GC analogy

| Concept | What LLM-GC Adds |
|---|---|
| Transparency | The developer sees what the model still knows, at what fidelity, and what it has lost — the system provides data, the human provides judgment. This is the point of the project |
| Instrumented from the inside out | Every component emits structured events; visualizer, logger and benchmarks all consume the same bus. This is *observability* — it describes LLM-GC rather than your conversation, and it is the supporting surface, not the headline |
| Signals, not scores | Named, individually-explainable signals (budget pressure, GC breakdown, generation lifecycle); an aggregated risk number ships only if it can be grounded in real hallucination labels |

## Installation

> The steps below set up a local development environment to build and test LLM-GC. This is
> what works today: the proxy and the dashboard run from `scripts/`. A `pip install llm-gc`
> package is not available yet.

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — the dependency and virtual environment manager this
  project uses

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

(See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for
Windows and other options.)

### Setup

```bash
# 1. Clone the repository
git clone git@github.com:rshmie/llm-gc.git
cd llm-gc

# 2. Install dependencies (creates .venv automatically, installs the
#    project in editable mode, and resolves the dev extras: pytest, ruff)
uv sync --extra dev

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify the setup
pytest
```

If the test suite passes, the environment is ready.

### Watch a conversation on the dashboard

Run the proxy with the dashboard beside it, then talk through it:

```bash
python scripts/run_proxy.py --serve --dashboard   # add --stub-upstream to run offline
python scripts/chat.py                            # in a second terminal
# open http://localhost:9901
```

A short guide explains the page the first time you open it. **The conversation**
view shows every turn as it was said beside what the model
still has of it — word for word, quoted, inside a summary, as facts, or out of
view — and which facts the model is actually sent. **LLM-GC at work** shows how
the collector scores each turn, every pass and its outcome, and the proxy's
counters. **Run GC now** cuts the selected conversation without waiting for the
batch minimum. `--min-cut-tokens 300` makes cuts happen within a short
conversation.

### Everyday commands

```bash
uv sync --extra dev   # re-sync .venv after pulling changes to pyproject.toml
pytest                # run the test suite
ruff check .          # lint
```

## Using it

### With any Anthropic client

Point the client at the proxy and change nothing else. The client's own API key is
forwarded, never stored:

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:9900")
```

`python scripts/chat.py` is exactly that, as a terminal chat. Claude Code can be pointed
the same way (`ANTHROPIC_BASE_URL=http://127.0.0.1:9900 claude`), but that path has not
yet been verified on real sessions.

### As a library

```python
from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.llm import AnthropicClient
from llm_gc.models import Message
from llm_gc.utils import count_tokens

service = build_gc_service(GCConfig(context_window=200_000),
                           summary_provider=AnthropicClient(api_key="sk-ant-..."))

async def on_turn(session_id: str, text: str, role: str, index: int) -> list[Message]:
    turn = Message(role=role, content=text, token_count=count_tokens(text), turn_index=index)
    await service.gc_update(session_id, turn)     # record the turn, then run a pass
    return await service.gc_collect(session_id)   # what to send: the recap, then recent turns
```

## What LLM-GC does and doesn't do

**In scope — built today:**
- Relevance scoring per conversation turn (four signals by default, a fifth optional)
- Three-tier sweep classification (keep / compact / archive)
- Generational memory (young / old / permanent) with fact extraction and supersession; cold summaries are archived into facts, and facts go back into the prompt
- A summarising compactor that refuses any summary that does not shrink what it replaces
- Cuts: the prompt changes only at its front, in batches, so the provider's prompt cache and the model's thinking blocks stay valid
- An Anthropic Messages proxy, streaming included, that finds each conversation by its content and never makes a reply wait on GC
- Per-session state that reads never wait for
- A live dashboard: what the model knows of each turn, and what LLM-GC did
- Configurable thresholds
- A decision-survival check: scripted conversations asked about their own facts, through LLM-GC and sent whole

**In scope — not built yet:**
- The OpenAI-compatible endpoint
- Claude Code through the proxy, verified on real traffic
- Keeping long assistant answers whole enough that their details survive
- Broader benchmarks, on real sessions rather than scripted ones
- Post-session analysis (session timeline, decision survival, GC effectiveness report)

**Does not handle (not our problem):**
- Model routing (use LiteLLM or your proxy)
- Semantic caching (separate concern)
- Rate limiting / budget management
- Authentication / API key management
- Prompt engineering or fine-tuning
- Cross-session memory (use Mem0/Zep for that)

LLM-GC does one thing: keep the context window clean so the model remembers what matters. Token savings are a natural byproduct of removing what doesn't.

### How this differs from what already exists

Context management is not a new idea, and the honest framing is worth stating plainly.

**[MemGPT / Letta](https://arxiv.org/abs/2310.08560)** paged LLM context against an
operating-system memory hierarchy in 2023 — the closest relative to this project's
framing. **[Mem0](https://github.com/mem0ai/mem0)** and **[Zep](https://www.getzep.com/)**
extract facts and resolve contradictions; Zep's temporal edge invalidation is the same
idea as supersession here. **LangChain** has summary-buffer memory. **Providers ship
their own compaction**, and it uses a frontier model to summarize, which is very likely
better at preserving meaning than five weighted heuristics.

So LLM-GC doesn't claim a better summarizer. What it does differently is **govern and
report** the operation rather than perform it silently:

- **Per-turn and selective**, not summarize-everything-at-a-threshold
- **Recoverable** — facts survive in the permanent generation; compaction is one-way
- **Auditable** — a record of what happened to each turn and why
- **Under your policy** — thresholds, conservatism, never-drop rules are yours, not the provider's
- **Cross-client** — a proxy works for anything speaking the provider's API

LLM-GC's compactor uses the *same mechanism* provider compaction uses: a model
summarises. It is not an alternative to compaction; it is a governance and transparency
layer around it.

## Measured results

The first numbers come from profiling **real Claude Code sessions**: they measure the
problem and a naive baseline. The second set, further down, measures LLM-GC itself.

```
STAGE 0 — THE PROBLEM (7 real sessions, 1,742 API calls, 130M cumulative tokens)
  Avg input tokens at peak:                              132,579
  Share of token spend that is input (resent history):   99%
  Share of input that is stale context at peak:          84%

STAGE 1 — NAIVE BASELINE (sliding window: keep last 10 turns, drop the rest)
  Token savings:                                         46-95% per session
  Decisions preserved:                                   7.6 of 8 LOST
```

**That second line is the whole argument.** Saving tokens is easy — a five-line sliding
window gets 46–95%. Saving them *without throwing away the decisions that made the
conversation worth having* is the hard part, and it's the only reason to build anything
more complicated than truncation.

### What LLM-GC keeps, measured

The decision-survival check plays scripted conversations through the proxy, then asks
the model about details they established, once through LLM-GC and once with the whole
conversation; a question answered right only with the whole conversation is a fact
LLM-GC lost. Method: [evals/decision_survival/README.md](evals/decision_survival/README.md).

| Conversations | Right through LLM-GC | Right sent whole | Prompt through LLM-GC |
|---|---|---|---|
| 3 short (18 turns, ~160-word answers) | 17/18 | 18/18 | 17–24% smaller |
| 2 long (30 turns, ~1,500-word answers) | 5/19 | 19/19 | 64–66% smaller |

Short conversations keep almost everything. On long ones LLM-GC condenses hard and loses
most details — chiefly those stated only in long assistant answers, which end up as
excerpts the recap cuts short. That is the work in progress, and these numbers will be
replaced as it lands.

Savings will also be judged against **cached-token pricing**, not raw token counts:
rewriting the start of the prompt invalidates the provider's prompt cache, so a real
reduction in tokens can still be an increase in cost. A savings number that ignores that
isn't an honest number.

## Configuration

The library takes a `GCConfig`; the proxy script exposes the settings worth turning while
you watch the dashboard:

| `GCConfig` | `run_proxy.py` flag | Default | What it decides |
|---|---|---|---|
| `min_cut_tokens` | `--min-cut-tokens` | 1000 | tokens of older messages to compact or archive before a collection runs |
| `last_n_turns_to_keep` | `--keep-last` | 5 | newest messages always sent word for word |
| `context_window` | `--context-window` | 128,000 (proxy: 200,000) | the model's window; the fact budget is a share of it |
| `keep_threshold` | | 0.7 | relevance at or above which a message is kept whole |
| `archive_threshold` | | 0.3 | relevance below which a message is archived |
| `min_compactable_tokens` | | 30 | messages this short are kept whole rather than summarised |
| `max_memory_ratio` | | 0.1 | share of the window the recap may spend on facts; 0 turns facts off |

An unknown setting is an error, so a misspelt one never looks configured.

## GC strategies

v1.0 ships **one** strategy, `generational`: full mark-sweep-compact across three
generations, with relevance scoring, fact extraction and memory retrieval. It is the
only strategy that is fully specified, and shipping one well-specified strategy beats
shipping three half-specified ones.

Two others were considered and rejected, for different reasons:

- **`sliding_window`** (drop everything older than N turns) — rejected outright. It
  ignores content entirely, which is what loses 7.6 of 8 decisions in the benchmark
  above. It remains useful as the baseline to beat, not as an option to offer.
- **`summary`** (keep recent turns, roll everything older into a summary) — rejected
  *for v1.0* on different grounds: it would need its own sweeper, compactor contract and
  config surface. It may return as a configurable variant once the generational pipeline
  is mature.

## Documentation

The design docs are the substantive part of this repository. Each one covers how a
subsystem works, why it exists, and **why it has that shape — including the credible
alternatives that were rejected and the reason.**

- [Overview](doc/overview.md) — why this project exists, the core principles, how the pieces fit
- [Scoring](doc/design/scoring/overview.md) — the relevance signals and the default weighting
- [Sweeping](doc/design/sweeper-and-composition/sweeping-mechanism.md) — three-tier KEEP / COMPACT / ARCHIVE classification
- [The cut](doc/design/generational-memory/the-cut.md) — when a collection happens, and how the recap is assembled
- [Compaction](doc/design/sweeper-and-composition/compaction-mechanism.md) — summarising compacted runs, and refusing a summary that does not help
- [Generational memory](doc/design/generational-memory/overview.md) — young / old / permanent, and supersession
- [Session management](doc/design/session/overview.md) — per-conversation state and its concurrency contract
- [Visibility](doc/design/visibility/overview.md) — transparency vs. observability, and why there is no aggregate risk score
- [Events](doc/design/events/overview.md) — the pub/sub bus every component instruments through
- [Storage](doc/design/storage.md) *(planned)* — schema versioning, so an upgrade never destroys retained knowledge
- [Failure handling](doc/design/failure-handling.md) *(planned)* — the circuit breaker, and keeping degradation visible

A full index is in [doc/README.md](doc/README.md).

## License

Apache 2.0
