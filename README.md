# LLM-GC: Garbage Collection for LLM Context

**See what your model remembers, what it has forgotten, and when to worry.**

If you've spent 50+ turns in a Claude Code session, you've probably watched the model
start forgetting things: the database you agreed on, the approach you picked, the thing
you said *not* to do. It's all still in the conversation, buried under 130K tokens of old
file reads, stale command output and settled debates. Every request re-sends all of it,
so you pay for the noise on every call, and the model answers in the same confident tone
whether the decision is still in view or not.

LLM-GC sits between your app and the provider and manages the conversation the way a
generational garbage collector manages memory. Recent turns stay word for word. Older
exchanges are shown at a level of detail — quoted whole, summarised, or archived to their
opening line and the facts they established — chosen by how much each matters now, chosen
again at every collection, and raised back when the conversation returns to them. The
next request carries one compact recap instead of the whole history. **And it never does this
silently:** a live dashboard shows, for every turn, exactly what the model still
receives, so you can see when its grounding has gone.

After the [setup](#setup), with `ANTHROPIC_API_KEY` set (or add `--stub-upstream` to run
offline, with canned replies):

```bash
python scripts/run_proxy.py --serve --dashboard   # the proxy on :9900, the dashboard on :9901
python scripts/chat.py                            # talk through it, and watch http://localhost:9901
ANTHROPIC_BASE_URL=http://127.0.0.1:9900 claude   # or run Claude Code through it
```

> **Status: work in progress, not yet pip-installable.** The engine, an
> Anthropic-compatible proxy (streaming included) and the dashboard are built and tested,
> and Claude Code runs through the proxy. A
> [decision-survival check](evals/decision_survival/README.md) measures what the model can
> still answer once LLM-GC has condensed a conversation: on its latest run all 37 answers
> survived, with prompts 12–17% smaller on the short conversations and 42–43% smaller on
> the long ones — one run, on scripted conversations. Not done yet: an `llm-gc` command
> with layered configuration, benchmarks on real sessions with a post-session report,
> recall (restoring a condensed exchange on the request that asks about it), and the
> OpenAI-compatible endpoint.

## The problem

Two distinct things go wrong, and only one of them is about money.

**Context loss.** The decision from turn 5 is still technically present at turn 50 — it's just one short message inside 100K tokens of less relevant material. Liu et al. (2023), [*Lost in the Middle*](https://arxiv.org/abs/2307.03172), showed that information buried in the middle of a long context is recalled less reliably than information at either end. So the decision can be *in* the window and still not usable. And the naive fix makes it worse: in an early profiling of 7 of the author's own Claude Code sessions, a sliding window that kept the last 10 turns lost **7.6 of 8 early decisions**.

**Context waste.** The same sessions averaged 132,579 input tokens at their peak, re-sent on every call, and about 84% of that was stale material: old file reads, command output, resolved debugging. You pay, on every call, to resend material that crowds out the material you need. *(That profiling predates this repository's benchmarks; its method is not yet published here, so treat these figures as motivation, not measurement.)*

This is recognisably a memory-management problem, and memory management has decades of prior art. Java, Go and C# all identify what is still needed, discard what is not, and compact what remains.

LLM-GC applies that shape to the context window. It scores each exchange by how much it matters now, condenses what has gone quiet without deleting anything, and files decisions in a permanent store along with whatever later replaced them. **The goal is preserving what the model needs to remember — not saving tokens.** Savings are a byproduct of removing waste.

But cleaning context is only half of it. The other half is **transparency**. Today you have no signal that the model has lost your database decision, no indicator that context quality is degrading, and no warning before the wrong answer arrives. The model answers in exactly the same confident tone whether its context is clean or 84% noise. LLM-GC surfaces what is happening to your context so that *you* can judge when to trust the output — the system provides the data, the human provides the judgment.

## How it works

LLM-GC sits between your app and the provider as a proxy. Every request is forwarded at once; after each reply, in the background, it runs a check, a GC pass. Most checks find too little to collect and change nothing. When enough older turns have gone quiet, the check makes a **collection**:

1. **Group** — the turns that have aged out of the recent tail become **spans**, one per exchange: a user's message and everything up to their next one
2. **Score** — every span, new and old, is scored by how much it matters *now*: recent, or about what the user is discussing, lifted by the decisions and specifics it states
3. **Plan** — each span gets a **detail level** from its score: quoted whole, summarised, or archived to its opening line and facts
4. **Condense** — each new span is summarised (one model call each), and one more call per collection reads the facts its turns settled, filed under the topic they change; the next request carries one **recap** of every span at its level, then the recent turns word for word

```
 what the client sends:     t0  t1  t2  …  t39  t40 … t57
 what the model receives:   [RECAP]  t40 … t57
                             └ t0–t39 as spans, each at its level, in one message
```

An agent's tool calls and their results are part of the conversation: they are sized,
scored, summarised and quoted like words, so old file reads and command output are
condensed too. The client's own instructions are not: its system prompt, its tools, and
the instructions it writes into messages (Claude Code's CLAUDE.md, memory and notes) reach
the model word for word with every request.

A collection decides whether it is worth making before it pays for a single summary. Collecting in batches means the start of the prompt changes rarely, so the provider's prompt cache keeps working, and none of it runs while a request waits.

### Generational memory

Context is managed in three generations, inspired by generational garbage collection:

| Generation | What it holds | Analogy |
|---|---|---|
| **Young** | Recent turns, sent word for word | Java's young/eden space |
| **Old** | Older exchanges as spans, each keeping its original turns and shown at a detail level | Java's tenured generation |
| **Permanent** | Facts read from every span, and what later replaced them; never discarded | Java's permanent generation |

Every span in the old generation is shown at one of four levels, and each level shows everything the one below it shows:

```
 level         what the model reads in the recap                          
 ───────────   ──────────────────────────────────────────────────────    
 kept          [Quoted exactly]  every word of the exchange               ▲ more detail
 summarised    [Summarised]      its opening line · its facts, a summary  │
 archived      [Archived]        its opening line · its facts             │
 out of view   [Out of view]     one line for a run of the oldest spans   ▼ fewer tokens
```

**The score decides the level; one budget caps the total.** A span scoring 0.7 or more is kept, 0.3 or more summarised, below that archived — re-decided at every collection, so a span the conversation returns to rises again from its own words. The recap may spend 30% of the model's window; over that, the coldest spans lose detail first. **Out of view** is the floor under all of that, not a judgement of the score: only when even one archived line per exchange no longer fits — around two thousand exchanges at the default budget — are the oldest folded behind one line saying they exist, and each still keeps its words. When a later turn replaces a decision ("we moved from Google Cloud to AWS"), the older span carries a note saying so, so the model does not read the old decision as current.

### Context Visualizer — epistemic transparency

AI models have no reliable internal sense of what they know and don't know. Getting a model to report its own uncertainty accurately is an open research problem (calibration), so you cannot lean on the model to tell you when its grounding is weak.

Humans, by contrast, do this naturally — **when they have the data**. Evaluating whether a source should be trusted is a cognitive skill the research literature calls **epistemic vigilance** (Sperber et al., 2010). LLM-GC's premise is to stop trying to automate the judgment and instead give the human what their own vigilance needs to work on.

Right now, developers have no data:
- You can't see that the model has effectively "forgotten" a decision you made 40 turns ago
- You can't see that the context is so bloated the model is statistically more likely to hallucinate
- You can't see which parts of the conversation are still "alive" in the model's attention vs. buried noise
- You have no signal telling you "this is the point where you should worry about answer quality"

LLM-GC's Context Visualizer solves this. Every GC component already computes rich internal state — span scores, detail levels, which generation each turn is in, what each collection did. Instead of using that data only internally, the visualizer **exposes it to the developer** as a live dashboard:

- **Named signals, no aggregate score** — token budget pressure, and what the last GC pass actually did (including when it was bypassed or failed — an unhealthy pass is visible, never hidden). Each signal is named for what it literally measures; nothing looks more authoritative than its math supports. The *transformation ratio* is computed but deliberately **not** displayed: it is honest arithmetic that would still be misread as a risk score if it sat on the page as the one number to glance at
- **What the model is told, fact by fact** — every fact on record under who stated it, you or the assistant, with the message it came from and how it reaches the model: on its exchange's line in the recap, in its words, or not sent. Every value replaced shows old and new, with both messages linked; the headline says how far the record reaches, and which messages' facts could not be read. A fact that is stored but never sent is the case this view exists to expose
- **What the model has of each message** — the whole conversation as a map, widths to scale: which messages go word for word, which are kept, compacted, archived or out of view, and what one request costs against the conversation as written
- **What the model knows, exchange by exchange** — every exchange with exactly what the model reads for it: kept (quoted word for word), compacted (its summary), archived (its opening line and facts), or out of view (one shared line saying it exists) — beside what was said. The newest request and its reply are shown as they happen, marked as not in the conversation yet; tool calls and results are labelled as such, never as your words; the client's own instructions show as a tag with their size. A second view shows the prompt exactly as forwarded, and a search answers "does the model still receive these words?"
- **What moved, and what fell short** — where every turn lives now (young / old / permanent), what each collection moved and when, a collection in progress, and every part of one that could not be done, with what the model gets instead
- **The next collection, and why** — the messages it would take and the level each would get, the earlier exchanges it would show at another level, and every exchange's score against the keep and archive lines, with the reason it sits where it does
- **Post-Session Analysis** *(planned)* — After a session ends: full timeline of when quality peaked, when it degraded, which decisions survived, where the danger zones were

There is deliberately **no single "health score" or "hallucination risk" number**. A number labelled "risk" claims an accuracy no heuristic can back without ground-truth hallucination labels — and a dishonest number is a new black box, which is the opposite of the point. If a learned predictor can be defensibly grounded in real labels, an aggregate ships *alongside* the signals; until then, the signals do the part that can be done honestly, and you do the part that can't yet be automated honestly. **Aggregation is earned, not assumed.** Most tools optimize context silently. LLM-GC shows you what's happening and lets YOU decide if the AI's output is trustworthy right now.

## The GC Analogy

LLM-GC borrows principles from garbage collection — a concept pioneered in Lisp (1959) and refined across decades of language runtime design in Java, Go, C#, and others. The generational model, mark-sweep-compact algorithm, and concurrent pre-computation are adapted from these well-established systems and applied to a new domain: LLM context management.

| GC Concept | LLM-GC Equivalent |
|---|---|
| Heap memory | Context window |
| Objects | Conversation turns, grouped into exchanges (spans) |
| Live objects | Relevant context (still needed) |
| Dead objects | Stale context (safe to condense) |
| GC roots | The next question, which has not been asked yet (see below) |
| Reachability analysis | Liveness: recent, or reached by what the user is discussing now |
| Mark phase | Score every span by how much it matters now |
| Sweep phase | Plan every span's detail level: kept, summarised, archived |
| Compaction | Rewrite the start of the prompt as one recap of every span at its level |
| Minor collection | A collection: a batch of older turns leaves the young generation as spans |
| Eden filling up | The turns a collection would condense reaching `min_cut_tokens` |
| Young generation | Recent turns (full detail) |
| Old generation | Older exchanges, each shown at a detail level |
| Permanent generation | Facts, and what replaced them (never discarded) |
| GC pause | None on the request: the check runs after each reply, in about 10 ms, and a collection's summaries are written then too |
| Concurrent GC | Collections run between turns, while the user reads the reply |
| GC tuning flags | Level lines, recap budget, batch size and newest turns kept |

### Where the analogy breaks — and why that matters more than where it holds

A runtime garbage collector is **sound**. Reachability is decidable and exact: trace from the roots, and an object is either reachable or it is not. The collector never frees a live object. That guarantee is precisely why Java's GC can run silently — it has nothing to report, because it cannot be wrong.

Relevance is not decidable. The root set for a conversation is the *next question*, and it hasn't been asked yet. A turn that looks dead at turn 10 can be the most important thing in the conversation at turn 12, and no analysis at turn 10 can determine that. Every relevance decision is a bet against a future that doesn't exist yet.

Three things follow, and they shape the whole design:

- **Cleanup cannot be silent.** A system making unsound, unverifiable decisions about what the model is allowed to see — and hiding them from the person who gets handed the answer — has replaced one opaque actor with two. Transparency here isn't a feature chosen because it's appealing. It's the only honest way to ship an operation that can't be proven correct.
- **Removal has to be recoverable.** Because a bet can be wrong, nothing is deleted: every span keeps its original turns, so a span the conversation returns to is shown in more detail again, and its decisions survive as facts whatever its level. A wrong bet should cost fidelity for a while, not knowledge. The [decision-survival check](evals/decision_survival/README.md) measures whether that holds.
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

# 2. Install dependencies (creates .venv automatically, installs the project in
#    editable mode, and the dev extras: pytest, ruff, mypy, hypothesis, the Anthropic SDK)
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

The **?** button explains the page. The **Conversation** tab shows every turn as it
was said beside what the model still has of it — word for word, kept, compacted,
archived, or out of view — and each fact exactly as the model is sent it. The
**Engine** tab shows the three generations, how each turn is scored, every check
and collection, and the proxy's counters. **Run GC now** collects the selected
conversation without waiting for the minimum. `--min-cut-tokens 300` makes
collections happen within a short conversation.

To look around without an API key or a conversation of your own, rehearse the
decision-survival check with its dashboard. It plays five scripted conversations
through the same engine, offline, each as its own session:

```bash
python -m evals.decision_survival.run --rehearse --dashboard
# open http://localhost:9911: it fills in as the check runs, and stays open after its report until Ctrl-C
```

Rehearsal shows the page's shape, not a real model's work: its stand-in summaries
state nothing, and its facts are read by simple patterns from your messages only, so
the assistant's column stays empty.

### Everyday commands

```bash
uv sync --extra dev   # re-sync .venv after pulling changes to pyproject.toml
pytest                # run the test suite
ruff check .          # lint
mypy                  # type-check the library
python -m evals.decision_survival.run --rehearse   # the decision-survival check, offline
```

## Using it

### With any Anthropic client

Point the client at the proxy and change nothing else. The client's own API key is
forwarded, never stored:

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:9900")
```

`python scripts/chat.py` is exactly that, as a terminal chat.

### With Claude Code

```bash
python scripts/run_proxy.py --serve --dashboard   # with ANTHROPIC_API_KEY set
ANTHROPIC_BASE_URL=http://127.0.0.1:9900 claude   # in the project you are working on
```

If your `~/.claude/settings.json` already sets `ANTHROPIC_BASE_URL`, because your
company routes Claude through its own proxy, that setting wins over the command line:
follow [Behind your organisation's proxy](#behind-your-organisations-proxy) instead.

Claude Code's own credentials are forwarded to Anthropic, never stored; the proxy's
`ANTHROPIC_API_KEY` pays only for summaries and fact reads. Without it, the proxy runs
with canned summaries, which is only useful for trying the dashboard. With an agent's conversation, LLM-GC:

- reads tool calls and results as text, so they are summarised and quoted like words;
- sends Claude Code's instructions (its system prompt, its tools, and the
  `<system-reminder>` blocks that carry CLAUDE.md and memory) word for word with every
  request, never condensed;
- passes its other calls, such as token counting, through untouched;
- shows its side requests (after each reply it asks the model to suggest your next
  prompt) as the *newest request*, never as something you said.

Sessions live in memory: restarting the proxy starts each conversation again from what
Claude Code sends, at the cost of one request sent whole and one collection. The first
collection of a long conversation can take a couple of minutes, and requests meanwhile
go whole — never worse than without LLM-GC.

### Behind your organisation's proxy

Use this if your company sends Claude's traffic through its own proxy: your
`~/.claude/settings.json` has an `env` block with `ANTHROPIC_BASE_URL` (the company
proxy's address) and a token or key. LLM-GC goes in the middle:

```
Claude Code  ──►  LLM-GC  ──►  your company's proxy  ──►  Anthropic
```

LLM-GC needs two things from your settings file:

- **the company proxy's URL**, the `ANTHROPIC_BASE_URL`, to forward Claude's requests to;
- **a credential it accepts**, the `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`), for
  LLM-GC's own summary calls, which go through the company proxy too.

#### 1. Start LLM-GC

In a terminal, from the LLM-GC folder:

```bash
settings=~/.claude/settings.json
export ANTHROPIC_AUTH_TOKEN="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["env"]["ANTHROPIC_AUTH_TOKEN"])' "$settings")"
python scripts/run_proxy.py --serve --dashboard \
  --upstream "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["env"]["ANTHROPIC_BASE_URL"])' "$settings")"
```

This reads both values from your settings, so you type neither and the token is never
shown. If your settings have `ANTHROPIC_API_KEY` rather than `ANTHROPIC_AUTH_TOKEN`,
change the name in both places on the `export` line.

In what it prints, `upstream:` should show your company proxy's URL. A warning that no
credential was found means the `export` didn't work. Leave this terminal open; the
dashboard is at http://127.0.0.1:9901.

#### 2. Point Claude Code at LLM-GC

Claude Code takes `ANTHROPIC_BASE_URL` from its settings files, and a value typed before
`claude` on the command line is overridden by them. So set LLM-GC's address in a settings
file. In the project you want to use, put this in `.claude/settings.local.json`:

```json
{ "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:9900" } }
```

If that file already exists (Claude Code saves your permission choices there), add the
`env` entry to it rather than replacing the file. Then start Claude in that project as
usual: `claude`.

A project's settings take priority over `~/.claude/settings.json`, so only this project
goes through LLM-GC; your other projects still use the company proxy directly. To stop,
remove the entry.

> **Every project instead?** Set `ANTHROPIC_BASE_URL` to `http://127.0.0.1:9900` in
> `~/.claude/settings.json` itself. Write down the company proxy's URL first: step 1 can
> no longer read it from that file, so pass it yourself, as `--upstream https://…`.

#### 3. Check that it works

Send Claude a message. LLM-GC's terminal prints a line for every request, and the
dashboard shows the conversation from its second message. Once the conversation has
grown, the dashboard says *Last collection …* and the prompt it sends gets smaller.

| What you see | What to do |
|---|---|
| Nothing new in LLM-GC's terminal | Claude isn't using LLM-GC: check step 2, and that you started `claude` in that project |
| Claude shows an error from the company proxy | `--upstream` is wrong: it must be exactly the company proxy's URL from your settings |
| Engine tab: *Summary request failed*, with a 401 or 403 | The company proxy refused LLM-GC's own calls: check the credential exported in step 1 |
| Nothing is ever compacted | Same cause as the row above: when every summary fails, nothing can shrink |

If the credential is right and summaries still fail, the error on the Engine tab says
why: the company proxy may block the model LLM-GC summarises with (`claude-sonnet-5`)
or need headers LLM-GC doesn't send yet.

**Other clients** work the same way: point them at `http://127.0.0.1:9900` with the
credential they already use for the company proxy. With the SDK, that is
`anthropic.Anthropic(base_url="http://127.0.0.1:9900")`; for `scripts/chat.py`, export
the same token in its terminal.

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
- Span scoring: each exchange scored by liveness (recent, or about what is discussed now) and by the decisions and specifics it states
- Detail levels — kept, summarised, archived, out of view — re-planned at every collection under one recap budget, so a span the conversation returns to rises again
- Generational memory (young / old / permanent): every span keeps its original turns; facts are filed with supersession, and a span whose decision was later replaced says so
- A summarising compactor that refuses any summary that does not shrink what it replaces
- Facts read by a model once per collection, filed under the topic they change; only the user changes a decision the user made
- Collections: the prompt changes only at its start, the recap, in batches, so the provider's prompt cache and the model's thinking blocks stay valid
- An Anthropic Messages proxy, streaming included, that finds each conversation by its content, passes the client's other calls through, and never makes a reply wait on GC
- Agent conversations: tool rounds condensed like words, the client's instructions never condensed — verified with Claude Code
- Per-session state that reads never wait for
- A live dashboard: what the model has of every message and fact, the newest request and reply, and what LLM-GC did
- Configurable level lines, margin, recap budget and batch size
- A decision-survival check: scripted conversations asked about their own facts, through LLM-GC and sent whole

**In scope — not built yet:**
- An `llm-gc` command, and layered configuration (defaults, a file, the environment, flags), with the context window taken from the model being called
- Benchmarks on real sessions: savings against cached-token pricing, latency, and decision survival
- Post-session analysis (session timeline, decision survival, GC effectiveness report)
- Tuning the defaults for larger savings, checked on the decision-survival check
- Faster first collections: summaries written at the same time, not one after another
- Recall: restoring a condensed exchange on the request that asks about it, rather than at the next collection
- The OpenAI-compatible endpoint, and retries for summary and fact-reading calls

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
their own compaction**, summarising with a frontier model, as LLM-GC's compactor does.

So LLM-GC doesn't claim a better summarizer. What it does differently is **govern and
report** the operation rather than perform it silently:

- **Per exchange and selective**, not summarize-everything-at-a-threshold
- **Recoverable** — every exchange keeps its original, so a condensed one can be shown whole again; provider compaction is one-way
- **Auditable** — a record of what happened to each turn and why
- **Under your policy** — level lines, recap budget and batch size are yours, not the provider's
- **Cross-client** — a proxy works for anything speaking the provider's API

LLM-GC's compactor uses the *same mechanism* provider compaction uses: a model
summarises. It is not an alternative to compaction; it is a governance and transparency
layer around it.

## Measured results

Benchmarks on real sessions are not published yet; results from them will come with their
dataset and method. What exists is the
[decision-survival check](evals/decision_survival/README.md): five scripted conversations,
37 questions about facts they established, asked once through LLM-GC and once with the
whole conversation sent, and graded. Its latest committed run:

| | Short conversations (18 questions) | Long conversations (19 questions) |
|---|---|---|
| Through LLM-GC | 18/18, prompts 12–17% smaller | 19/19, prompts 42–43% smaller |
| Whole conversation | 18/18 | 19/19 |

One run, on scripted conversations, with `claude-sonnet-5-5` answering; the
[evidence file](evals/decision_survival/evidence/facts-room-2026-10-06.json) holds every
answer. It grades one of them wrong: asked for the shard count, the model answered
"9 primaries", the user's value in words the grader did not accept until then. The
stored answers regrade to 37/37. The check's README lists the earlier runs and what each
change won back.

Savings will also be judged against **cached-token pricing**, not raw token counts:
rewriting the start of the prompt invalidates the provider's prompt cache, so a real
reduction in tokens can still be an increase in cost. A savings number that ignores that
isn't an honest number.

## Configuration

The library takes a `GCConfig`; the proxy script exposes the settings worth turning while
you watch the dashboard:

| `GCConfig` | `run_proxy.py` flag | Default | What it decides |
|---|---|---|---|
| `min_cut_tokens` | `--min-cut-tokens` | 1000 | tokens a collection must summarise or archive before it runs |
| `last_n_turns_to_keep` | `--keep-last` | 5 | newest messages always sent word for word |
| `context_window` | `--context-window` | 128,000 (proxy: 200,000) | the model's window; the recap budget is a share of it |
| `keep_threshold` | | 0.7 | score at or above which a span is quoted whole |
| `archive_threshold` | | 0.3 | score below which a span is archived; between the two, summarised |
| `level_margin` | | 0.05 | how far past a line a score must move before a span changes level, so a score near a line does not flip it back and forth |
| `max_recap_ratio` | | 0.3 | share of the window the recap may spend; over it, the coldest spans lose detail, then the oldest fold out of view |

An unknown setting is an error, so a misspelt one never looks configured.

The proxy also takes where to forward, and LLM-GC's own credential:

| Setting | Given as | What it decides |
|---|---|---|
| `--upstream` | flag | where requests are forwarded, and where LLM-GC's own summary and fact calls go; `https://api.anthropic.com` unless set |
| `ANTHROPIC_API_KEY` | environment | LLM-GC's own key, sent as `x-api-key` |
| `ANTHROPIC_AUTH_TOKEN` | environment | LLM-GC's own bearer token, sent as `Authorization`; either or both |

Credentials come only from the environment, never from flags, so they never land in
shell history. Without either, summaries are canned: useful for trying the dashboard,
not for real work.

## GC strategies

LLM-GC has **one** strategy, generational: exchanges scored by how much they matter now,
shown at detail levels re-planned at every collection, with facts and their replacements
carried in the prompt. One well-specified strategy beats three half-specified ones.

Two others were considered and rejected, for different reasons:

- **`sliding_window`** (drop everything older than N turns) — rejected outright. It
  ignores content entirely, which is what lost 7.6 of 8 decisions in the early profiling
  described under [The problem](#the-problem). It remains useful as the baseline to beat, not as an option to offer.
- **`summary`** (keep recent turns, roll everything older into one summary) — rejected
  on different grounds: it condenses what matters now as hard as what does not, cannot
  restore anything, and loses the per-exchange record that makes the transparency
  possible.

## Documentation

The design docs are the substantive part of this repository. Each one covers how a
subsystem works, why it exists, and **why it has that shape — including the credible
alternatives that were rejected and the reason.**

- [Overview](doc/overview.md) — why this project exists, the core principles, how the pieces fit
- [Scoring](doc/design/scoring/overview.md) — how much each span matters now: liveness, and the decisions and specifics it states
- [Generational memory](doc/design/generational-memory/overview.md) — young / old / permanent, spans, and supersession
- [Detail levels](doc/design/generational-memory/detail-levels.md) — kept, summarised, archived, out of view, and how the planner picks them
- [The cut](doc/design/generational-memory/the-cut.md) — when a collection happens, and how it changes the prompt
- [Compaction](doc/design/generational-memory/compaction.md) — how a span gets its summary, and refusing one that does not help
- [The proxy](doc/design/proxy/overview.md) — reconciling with the client's history, finding the conversation, forwarding, and what it never condenses
- [LLM providers](doc/design/llm-provider.md) — the model calls behind summaries and facts
- [Session management](doc/design/session/overview.md) — per-conversation state and its concurrency contract
- [Visibility](doc/design/visibility/overview.md) — transparency vs. observability, and why there is no aggregate risk score
- [Events](doc/design/events/overview.md) — the pub/sub bus every component instruments through
- [Storage](doc/design/storage.md) *(planned)* — schema versioning, so an upgrade never destroys retained knowledge
- [Failure handling](doc/design/failure-handling.md) *(planned)* — the circuit breaker, and keeping degradation visible

A full index is in [doc/README.md](doc/README.md).

## License

Apache 2.0
