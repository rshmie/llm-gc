# Roadmap

What is left before LLM-GC's first release, and what comes after it. How each part works today is in [`doc/design/`](doc/design/); this file lists only what remains.

## First release

The first release has to back one claim with numbers anyone can rerun: in a long conversation, LLM-GC keeps what the model needs, costs less than sending everything, and shows you what it did. It also has to install with `pip`.

The work runs in three steps. Measure first, because some fixes should be chosen by what the numbers show rather than by guessing. Ship last, because the README must not state a number that has not been measured.

```
  1. Measure    ──►   2. Fix          ──►   3. Ship
     the claim           what the              the package
                         numbers show
        ▲                    │
        └────── rerun ───────┘
```

### 1. Measure the claim

Decide these before building, because a published number cannot be quietly redefined later:

- **What "savings" counts.** Plain tokens, or what the provider bills: cache reads and writes, with LLM-GC's own model calls subtracted.
- **Where the test conversations come from.** Scripted ones are reproducible and shareable. Recorded real sessions are more credible, but they are user data and must not leave the machine unredacted.
- **What "without quality loss" means as a number**, and the threshold that fails a run.

Then build:

- [ ] **Rerun the decision-survival check** on a clean commit of the current code ([how to run it](evals/decision_survival/README.md)).
- [ ] **A benchmark harness** that runs conversations through the proxy with and without LLM-GC (`--compactor noop` is the baseline) and records each run.
- [ ] **Savings counted as the provider bills**, including what LLM-GC's own summary and fact calls cost. A smaller prompt can cost more when its start changes, and LLM-GC's own calls can cost more than the cache reads they save.
- [ ] **Quality by counterfactual replay:** ask the same question against the full conversation and against LLM-GC's version, and compare the answers.
- [ ] **Grounding failures in live replies:** the model asking for something the session already settled, or calling it unknown. It is the one live signal that can show quality was lost.
- [ ] **Latency:** what LLM-GC adds to each request (p50 and p95), how long a collection takes, and how long a pass blocks the proxy, with one and with four conversations at once.
- [ ] **Tune the defaults** against the benchmark, rerunning the decision-survival check after each change.
- [ ] **A post-session report** on the dashboard: what was kept, condensed and archived, which decisions survived, and what the session cost against no LLM-GC. It is the page people share, so it must make sense to someone who did not build LLM-GC.

### 2. Fix what the numbers show

- [ ] **Anything the benchmark shows losing quality.** One cause is already known: relevance matches words, so "datastore" never finds "database". Embedding relevance behind the existing `Relevance` seam fixes it without training anything.
- [ ] **Fewer of LLM-GC's own model calls.** Find where they go and cut the ones that buy nothing: on one live run, 25 of 65 summary calls went to a conversation that never sent a second request.
- [ ] **A way to run without paid API credit.** Someone on a Claude Pro or Max plan has no API key, and their conversation costs them nothing per token, so LLM-GC's summary calls are a new bill that saves them no money. Try open models served on their own machine through the existing OpenAI-compatible provider (such as Ollama, LM Studio or vLLM), and document the best one that passes the decision-survival check.
- [ ] **Faster first collections**, if the latency numbers confirm that writing summaries one after another is the cost (132 s on one cold start).
- [ ] **Small fixes:**
  - The dashboard groups conversations into sessions in two places, its script and the server, so the two can disagree. Keep the server's.
  - `--context-window`'s help still calls it the proxied model's window. It is the fallback used when the Models API does not know the model.
  - An exchange with no prose is named by its raw tool JSON. Name it by its tool and the first line of its result.
- [ ] **Rerun the benchmark** on the fixed code.

### 3. Ship

- [ ] **Integration tests** for the proxy over HTTP against a stand-in upstream, with coverage reported.
- [ ] **The benchmark's regression check in CI**, so a change that loses quality or savings fails its pull request.
- [ ] **PyPI publishing** through trusted publishing, with no long-lived token.
- [ ] **Repository settings**, when it goes public: CI must pass before a merge, and private vulnerability reporting is on (`SECURITY.md` points to it).
- [ ] **The README's results** replaced by the benchmark's numbers, with a link to how they were produced.

## After the first release

**Limits the first release ships with**, stated in its docs and fixed after it:

- **Sessions live in memory.** A restart costs each conversation one whole request and one collection.
- **A very large tool result is sent whole** while it is among the newest messages, and condensed only once it is older.
- **LLM-GC's own model calls are not retried.** A failed summary keeps its exchange whole, which costs tokens but no detail, and is shown as a degradation.
- **A relayed subagent report is not tied to the subagent that wrote it.**
- **Without a model, facts are read by patterns** that catch few phrasings.

**Next features:**

- **An OpenAI-compatible endpoint** (`/v1/chat/completions`), for clients that speak OpenAI's format. It waits so that the first release's claims are measured on one client, Claude Code, instead of shipping a second one unmeasured.
- **A Docker image** that starts a working proxy, published to localhost only by default, and shuts down cleanly.
- **Sessions that survive a restart.**

**Learned models.** Each is trained and judged on data the benchmark produces, so none can start before it, and each replaces today's method only if it beats it there:

- **Our own summary and fact model**, only if no open model passes the decision-survival check.
- **A learned relevance scorer.** Today 40% of a span's score comes from counting decision phrases and specifics with patterns. It needs a labelled set of what a person would keep, condense and archive.
- **A hallucination-risk predictor, evaluated first.** It ships only if it can be grounded in real, labelled grounding failures. Otherwise the separate signals stay, and the negative result is written up.
- **A summary-quality evaluator** that scores each summary directly.
