# LLM-GC: Garbage Collection for LLM Context

> **Status:** Work in progress — currently in early development (Phase 0: project setup). Nothing is installable yet.

If you've spent 50+ turns in a Claude Code session, you've probably noticed the model starts forgetting things. The database you agreed on, the auth approach you picked, the thing you explicitly said *not* to do — it's all still in the conversation, but buried under 130K tokens of old file reads, stale command outputs, and resolved debugging context. The model either hallucinates a wrong answer or tells you "I don't have that context."

The primary root cause is: every API call resends the entire conversation history from scratch. Nothing gets cleaned up. Old context piles up, pushes important decisions out of the model's attention, and the quality of responses degrades. Measured across 7 real Claude Code sessions, 84% of input tokens at peak were stale context — noise that actively hurts the model's ability to recall what matters. And since 99% of token spend is input (the resent history), you're paying to re-send context that's working against you.

This problem isn't new. It's the same thing that happens when a program runs out of heap memory — except in that world, it was solved decades ago with garbage collection. Java, Go, C# — they all manage memory by identifying what's still needed, discarding what's not, and compacting what remains.

LLM-GC applies that same idea to LLM context windows. It scores each conversation turn for relevance, removes what's no longer useful, compresses what's aging, and extracts key decisions into a permanent store that survives indefinitely. The goal is to preserve what the model needs to remember — not to save tokens. Token savings are a natural byproduct of removing waste, not the point.

But fixing context is only half the problem. The other half is **visibility**. Right now, developers have zero insight into what's happening inside their LLM session. There's no signal telling you "the model has forgotten your database decision," no indicator that context quality is degrading, no warning before the hallucination happens. The model answers with the same confidence whether its context is clean or 84% noise. LLM-GC's Context Visualizer exposes what's happening inside your session — the AI provides transparency, the human provides judgment.

## The Problem

In a typical Claude Code session, every API call resends the **entire conversation history** from scratch. By turn 50, that's 130K+ tokens per call — mostly old file reads, stale command outputs, and resolved debugging context. Two things go wrong:

**Context Loss** — The model forgets what was decided earlier. "What database are we using?" gets a hallucinated answer because the decision from turn 5 is buried under 100K tokens of noise. In benchmarks on real sessions, a simple sliding window (keep last 10 turns, drop the rest) loses **nearly all early decisions**.

**Context Waste** — 99% of token spend is input tokens (the resent history), not output (the model's response). Measured across 7 real Claude Code sessions: 130M cumulative input tokens, 84% estimated waste at peak. The cost adds up, but worse, the noise actively degrades response quality.

This is the equivalent of a program crashing because it ran out of memory. In Java, this was solved decades ago with garbage collection. In LLMs, we can adopt the similar idea.

## How It Works

LLM-GC intercepts LLM API calls, optimizes the conversation context, and forwards the cleaned version. Three operations, inspired by classic GC algorithms:

1. **Mark** — Score every conversation turn for relevance to the current query
2. **Sweep** — Remove dead turns, compress stale ones, extract key facts before discarding
3. **Compact** — Assemble clean context: system prompt + extracted memories + compressed summaries + recent turns

### Generational Memory

Context is managed in three generations, inspired by generational garbage collection:

| Generation | What it holds | Analogy |
|---|---|---|
| **Young** | Recent turns in full detail | Java's young/eden space |
| **Old** | Older turns compressed into summaries | Java's tenured generation |
| **Permanent** | Extracted facts and decisions, never discarded | Java's permanent generation |

As turns age, they move through generations: kept in full (young) → compressed (old) → facts extracted (permanent) → original discarded. Key decisions survive indefinitely in the permanent generation and are retrieved when relevant.

### Context Visualizer — Epistemic Transparency

AI models don't know what they don't know. The research community calls this the problem of **epistemic vigilance** — how do you know when to trust a source? For AI, that's an unsolved research problem. But for humans, it's a natural skill — **if they have the data**.

Right now, developers have no data:
- You can't see that the model has effectively "forgotten" a decision you made 40 turns ago
- You can't see that the context is so bloated the model is statistically more likely to hallucinate
- You can't see which parts of the conversation are still "alive" in the model's attention vs. buried noise
- You have no signal telling you "this is the point where you should worry about answer quality"

LLM-GC's Context Visualizer solves this. Every GC component already computes rich internal state — relevance scores, generation classifications, waste percentages, decision tracking. Instead of using that data only internally, the visualizer **exposes it to the developer** as a live dashboard:

- **Context Health Score** — Overall 0-100 session health, updated every turn
- **Hallucination Risk Indicator** — Computed from measurable signals: context waste, decision burial depth, proximity to context limit. Not a guess — every factor is visible and explainable
- **Decision Survival Map** — Every key decision tracked: alive (young gen), compressed (old gen), preserved (permanent gen), or lost
- **Turn Relevance Heatmap** — Every message color-coded by current relevance
- **Generation State View** — Visual of what's in each generation and token usage
- **Post-Session Analysis** — After a session ends: full timeline of when quality peaked, when it degraded, which decisions survived, where the danger zones were

The hallucination risk starts as a heuristic (v1, computed from GC signals) and evolves into a trained ML predictor (v2, Phase 11-12). Most tools optimize context silently. LLM-GC shows you what's happening and lets YOU decide if the AI's output is trustworthy right now.

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
| GC pause | Latency overhead (<25ms warm) |
| Concurrent GC | Async pre-computation between turns |
| GC tuning flags | Configurable thresholds and strategies |

### Beyond the GC Analogy

| Concept | What LLM-GC Adds |
|---|---|
| Epistemic transparency | Developer sees context health, hallucination risk, decision survival — AI provides data, human provides judgment |
| Observability from day one | Every GC component emits structured events — visualizer, logger, benchmarks all consume the same metrics bus |
| Hallucination risk heuristic | Composite score from waste %, decision burial depth, token proximity to limit — not a black box |

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

### Everyday commands

```bash
uv sync --extra dev   # re-sync .venv after pulling changes to pyproject.toml
pytest                # run the test suite
ruff check .          # lint
```

## Planned Usage

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

### SDK Wrapper (one import change)

```python
from llm_gc import Anthropic  # drop-in replacement
client = Anthropic()
# Everything else identical. GC is automatic.
```

## What LLM-GC Does and Doesn't Do

**Handles:**
- Preserving key decisions and facts from early in the conversation
- Context garbage collection (mark, sweep, compact)
- Generational memory (young/old/permanent)
- Relevance scoring per conversation turn
- Summary compression of old turns
- Fact extraction and retrieval (decisions survive indefinitely)
- Async pre-computation between turns
- Session tracking
- Configurable strategies and thresholds
- Transparent proxy for Claude Code (zero-config)
- **Real-time context health dashboard** (hallucination risk, decision survival, waste metrics)
- **Post-session analysis** (session timeline, degradation curve, GC effectiveness report)

**Does not handle (not our problem):**
- Model routing (use LiteLLM or your proxy)
- Semantic caching (separate concern)
- Rate limiting / budget management
- Authentication / API key management
- Prompt engineering or fine-tuning
- Cross-session memory (use Mem0/Zep for that)

LLM-GC does one thing: keep the context window clean so the model remembers what matters. Token savings are a natural byproduct of removing what doesn't. And unlike every other context tool, it shows you exactly what it's doing — so you can judge for yourself when to trust the output.

## Measured Results (from real Claude Code sessions)

```
STAGE 0 — THE PROBLEM (7 real sessions, 1,742 API calls)
  Avg input tokens at peak:        132,579
  99% of token spend is input (resent history)
  84% of input is stale context at peak

STAGE 1 — DUMB GC BASELINE (sliding window, keep last 10 turns)
  Token savings:                   46-95% per session
  Decisions preserved:             NEARLY NONE (7.6 out of 8 lost)

  This is why dumb truncation isn't good enough.
  The savings are easy. Preserving decisions is the hard part.

WITHOUT LLM-GC:                        WITH LLM-GC (goal):
  Decisions remembered at turn 50: ~10%   Decisions remembered at turn 50: ~90%+
  Token cost grows linearly               Token cost stays flat
  Context quality degrades                Context quality stable
  Token reduction: none                   Token reduction: 40-60% (byproduct)
```

## Configuration

```yaml
# ~/.llm-gc/config.yaml
gc:
  strategy: generational     # sliding_window | summary | generational
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
  hallucination_risk: true   # compute and display risk heuristic
```

## GC Strategies

| Strategy | Description | Savings | Best for |
|---|---|---|---|
| `sliding_window` | Keep last N turns, drop the rest | 30-50% | Simple chatbots |
| `summary` | Keep recent in full, summarize older turns | 50-70% | Most use cases |
| `generational` | Full mark-sweep-compact with three generations | 60-85% | Long sessions, complex work |

## Documentation

- [Overview](doc/overview.md) — Why this project exists, what it does, how it is structured
- [System Spec](doc/end-product/spec.md) — The v1.0 system in detail: components, interfaces, configuration
- [Architecture Decision Records](doc/adr/) — Why specific design choices were made
- [Engineering Practices](doc/engineering-practices.md) — Internal disciplines we follow when building LLM-GC
- [Tech Stack](doc/LLM-GC-TechStack.md) — Dependencies and why each was chosen
- [Testing](doc/LLM-GC-Testing.md) — Benchmark methodology, quality metrics, test suite
- [ML Research](doc/LLM-GC-ML-Research.md) — Learned scoring, summary quality, fact extraction

## License

Apache 2.0
