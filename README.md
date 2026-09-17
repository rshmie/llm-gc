# LLM-GC: Garbage Collection for LLM Context

> **Status: work in progress, and not yet useful as a tool.** The engine is built and
> tested — relevance scoring, three-tier sweep classification, generational memory with
> knowledge extraction, the full pipeline behind `GarbageCollector.collect()`, a
> context-health monitor, and a dashboard scaffold
> (`python -m llm_gc.visualizer` → http://localhost:9901). 282 tests pass.
>
> What does **not** work yet: aged turns move into the old generation but never leave it,
> and extracted facts are never injected back into the assembled context. So today
> LLM-GC *relocates* context rather than reclaiming it — measured on a 40-turn
> conversation, the assembled context crosses the window around turn 15 and reaches 1.5×
> by turn 39. **Every savings figure below is a target, not a measurement.** Closing that
> loop, plus the real summarizing compactor, is the current work. The transparent proxy
> is not built; nothing is pip-installable yet.

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

1. **Mark** — Score every conversation turn for relevance to the current query
2. **Sweep** — Remove dead turns, compress stale ones, extract key facts before discarding
3. **Compact** — Assemble clean context: system prompt + extracted memories + compressed summaries + recent turns

### Generational memory

Context is managed in three generations, inspired by generational garbage collection:

| Generation | What it holds | Analogy |
|---|---|---|
| **Young** | Recent turns in full detail | Java's young/eden space |
| **Old** | Older turns compressed into summaries | Java's tenured generation |
| **Permanent** | Extracted facts and decisions, never discarded | Java's permanent generation |

As turns age, they move through generations: kept in full (young) → compressed (old) → facts extracted (permanent) → original discarded. Key decisions survive indefinitely in the permanent generation and are retrieved when relevant.

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
- **Decision Survival Map** *(planned)* — Every key decision tracked: alive (young gen), compressed (old gen), preserved (permanent gen), or lost
- **Turn Relevance Heatmap** *(planned)* — Every message color-coded by current relevance
- **Post-Session Analysis** *(planned)* — After a session ends: full timeline of when quality peaked, when it degraded, which decisions survived, where the danger zones were

There is deliberately **no single "health score" or "hallucination risk" number**. A number labelled "risk" claims an accuracy no heuristic can back without ground-truth hallucination labels — and a dishonest number is a new black box, which is the opposite of the point. If the planned learned predictor (Phases 11-12) can be defensibly grounded in real labels, an aggregate ships *alongside* the signals; until then, the signals do the part that can be done honestly, and you do the part that can't yet be automated honestly. **Aggregation is earned, not assumed.** Most tools optimize context silently. LLM-GC shows you what's happening and lets YOU decide if the AI's output is trustworthy right now.

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
> what works today. The end-user install (`pip install llm-gc`, the proxy, the SDK wrapper) is
> the target experience once the project ships — see [Planned Usage](#planned-usage) below.

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

### Try the dashboard scaffold

The first visible surface of the visualizer already runs today:

```bash
python -m llm_gc.visualizer
# open http://localhost:9901
```

Click **"Run GC now"** — each click feeds a batch of synthetic conversation
turns through the real GC pipeline and updates every signal live: token
budget pressure, the last pass's KEEP/COMPACT/ARCHIVE breakdown, generation
lifecycle, and the promote/archive/supersede transitions timeline.

### Everyday commands

```bash
uv sync --extra dev   # re-sync .venv after pulling changes to pyproject.toml
pytest                # run the test suite
ruff check .          # lint
```

## Planned usage

### With Claude Code (primary use case)

```bash
pip install llm-gc

# If using Claude Code directly:
llm-gc start
export ANTHROPIC_BASE_URL=http://localhost:9900

# If using Claude Code through a company proxy (e.g. port 6655):
llm-gc start --forward-to http://localhost:6655
# Then update .claude config to point base_url at http://localhost:9900

# That's it. Use Claude Code as normal.
# Context Visualizer runs at http://localhost:9901 — open it to see live session health.
```

### Library (full control)

```python
from llm_gc import GarbageCollector

gc = GarbageCollector(strategy="generational")
result = gc.collect(messages=my_messages, query=current_query)

# Send optimized messages to LLM
response = client.messages.create(messages=result.messages)

print(f"Decisions preserved: {len(result.memories_used)}")
print(f"Saved {result.tokens_saved} tokens ({result.savings_pct}%)")
```

### SDK wrapper (one import change)

```python
from llm_gc import Anthropic  # drop-in replacement
client = Anthropic()
# Everything else identical. GC is automatic.
```

## What LLM-GC does and doesn't do

**In scope — built today:**
- Relevance scoring per conversation turn (five signals, weighted)
- Three-tier sweep classification (keep / compact / archive)
- Generational memory (young / old / permanent) with fact extraction and supersession
- Session tracking with per-session concurrency control
- Context-health signals and a dashboard scaffold
- Configurable thresholds

**In scope — not built yet:**
- Reclaiming context rather than relocating it (sweeping the old generation; injecting permanent-generation facts back into the prompt)
- Summary compression that actually compresses — the default compactor is a no-op placeholder
- Transparent proxy for Claude Code (zero-config)
- Async pre-computation between turns
- Knowledge-state view: what the model still knows, at what fidelity
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

Once the real compactor lands, LLM-GC uses the *same mechanism* provider compaction
uses. It is not an alternative to compaction; it is a governance and transparency layer
around it.

## Measured results

These numbers come from profiling **real Claude Code sessions**. They measure the
problem and a naive baseline. They do **not** measure LLM-GC, which isn't finished.

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

### Targets — not yet measured

Stated so they can be checked later, and so nobody mistakes them for results:

| | Without LLM-GC | Target with LLM-GC |
|---|---|---|
| Decisions recoverable at turn 50 | ~10% | 90%+ |
| Context growth | linear | bounded |
| Token reduction | none | 40–60%, as a byproduct |

Two honesty notes on that last row. First, LLM-GC doesn't reclaim context yet, so the
figure today is **worse than zero** — see the status banner. Second, when it is measured
it will be measured against **cached-token pricing**, not raw token counts: rewriting the
prompt prefix invalidates the provider's prompt cache, so a real reduction in tokens can
still be an increase in cost. A savings number that ignores that isn't an honest number,
and this project would rather publish an awkward one.

## Configuration

```yaml
# ~/.llm-gc/config.yaml
gc:
  strategy: generational     # the only strategy in v1.0 (see "GC strategies" above)
  aggressiveness: 0.6        # 0.0 (keep all) to 1.0 (aggressive trim)
  young_gen_size: 8          # keep last N turns in full

scoring:
  recency: 0.30
  similarity: 0.30
  density: 0.15
  decision: 0.15
  reference: 0.10

storage:
  session: sqlite            # memory | sqlite | redis
  permanent: sqlite          # memory | sqlite | postgres

models:
  embedding: local           # local | openai | ollama
  summarizer: api            # api | ollama | heuristic

visualizer:
  enabled: true              # enable/disable the dashboard
  port: 9901                 # web dashboard port
  realtime: true             # WebSocket live updates
  post_session_report: true  # generate analysis when session ends
  # signals only — an aggregated risk display ships only if the learned
  # predictor (Phases 11-12) can ground it in real hallucination labels
```

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
- [System Spec](doc/end-product/spec.md) — the v1.0 system in detail: components, interfaces, configuration
- [Scoring](doc/scoring/overview.md) — the five relevance signals and how they combine
- [Sweeping & composition](doc/sweeper-and-composition/sweeping-mechanism.md) — three-tier classification and context assembly
- [Generational memory](doc/generational-memory/overview.md) — young / old / permanent, and supersession
- [Visibility](doc/visibility/overview.md) — transparency vs. observability, and why there is no aggregate risk score
- [Events](doc/events/overview.md) — the pub/sub bus every component instruments through
- [Engineering practices](doc/engineering-practices.md) — the internal disciplines followed while building this
- [Tech stack](doc/LLM-GC-TechStack.md) — dependencies and why each was chosen
- [ML research](doc/LLM-GC-ML-Research.md) — learned scoring, summary quality, fact extraction

## How this project was built

LLM-GC is written with AI coding assistance (Claude), used deliberately and throughout.
The division of labour is worth stating plainly, because it affects how to read this
repository:

- **The architecture and the design decisions are mine.** Every significant choice —
  three-tier sweeping, one-directional aging, signals over an aggregated risk score, the
  per-session lock contract — was made by me. The design docs above record the
  alternatives considered and rejected for each one, which is the part worth reviewing.
- **The implementation was written by an AI assistant under my direction**, reviewed by
  me, and verified by the test suite and CI.
- **The design documents are mine**, drafted collaboratively and edited by me.

Commits carry a `Co-Authored-By` trailer for the assistant. That trailer is a convention
meant for human collaborators and a tool isn't an author, so it is imprecise — it is used
anyway because it is the one form of attribution every reader and every tool already
recognises, and because a record written as the work happened is worth more than a claim
added afterwards.

## License

Apache 2.0
