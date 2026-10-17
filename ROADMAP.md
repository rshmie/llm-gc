# Roadmap

What is left between today and LLM-GC's first version, in the order it should be done, and why each item is needed. How each part of the system works is in [`doc/design/`](doc/design/); this file says only what remains.

## The order, and why

```
  1. Verify           2. Measure             3. Fix                 4. Ship
  what exists   ──►   the claims      ──►    what blocks     ──►    the package
                                             a release
  live runs,          benchmark, quality,    event loop,            license, CI,
  the checks          report, tuning         retries, rerun         PyPI, Docker, docs
                            ▲                     │
                            └──── rerun ──────────┘
```

- **Verify before measuring.** The benchmark reads its numbers through the same counts the dashboard shows. If those are wrong, every benchmark result is wrong with them.
- **Measure before fixing.** Some fixes (running summaries concurrently, tuning the defaults) should be chosen by what the measurement shows, not by guessing. The benchmark is rerun after the fixes, so the published numbers are the released code's.
- **Ship last.** Packaging changes no behaviour, and the README cannot state a number that has not been measured yet. Publishing a claim before measuring it would break the project's own premise: say what is known, and how it is known.

## 1. Verify what exists

The newest changes have been checked by tests, not yet on a real session.

- [ ] **The switch policy, on a real Claude Code session.** A prepared collection now replaces the prompt only when the provider's cache is cold, when the saving repays the cache rewrite, or when the stale-weight bound forces it ([when to collect](doc/design/generational-memory/when-to-collect.md)). The live run that led to it saved fewer tokens than it made the provider write to the cache again. Done when, on a session of similar length, the extra cache writes caused by switches are fewer than the tokens saved, and every switch that costs more says it was forced.
- [ ] **The dashboard's counts, on the same run.** The provider's prompt tokens divided by LLM-GC's count stays roughly constant from request to request; turn and session totals equal the sum of their rows; each subagent sits under the call that started it; a rate limit shows the pause.
- [ ] **The decision-survival check, rerun** ([how to run it](evals/decision_survival/README.md)). The facts prompt and the timing of collections have both changed since its last run.

## 2. Measure the claims

Today's evidence is a 37-question check on scripted conversations and a few live runs read by hand. A claim on the README needs a benchmark anyone can rerun.

**Decide before building**, because each is hard to change once results are published:
- What the savings figure counts: plain tokens, or the provider's tokens by kind with the cache counted.
- Where the benchmark's conversations come from: scripted (reproducible and shareable), or recorded real sessions (credible, but they are user data and must not leave the machine unredacted).
- What "without quality loss" means as a number, and the threshold that fails a run.
- Whether tuning may change the defaults LLM-GC ships with, or only report on them.

**Build:**
- [ ] **A benchmark harness** that runs conversations through the proxy with and without LLM-GC (`--compactor noop` is the baseline) and records each run's results.
- [ ] **Savings counted the way the provider bills**, with cache reads and writes, not raw token deltas: a smaller prompt can cost more when its start changes.
- [ ] **Quality by counterfactual replay:** ask the same question against the full context and against LLM-GC's, and compare the answers.
- [ ] **Grounding failures, read from live replies:** the model asking for something the session already established, or calling it unknown. It is the one live signal that can show quality was lost.
- [ ] **Classification correctness** against a small labelled set: did LLM-GC keep, compact and archive what a person would have?
- [ ] **Latency added on the request path** (p50 and p95), and how long a collection takes cold and warm.
- [ ] **Tune the draft defaults** against the benchmark, rerunning the decision-survival check after each change: `min_cut_tokens`, `last_n_turns_to_keep`, `keep_threshold`, `archive_threshold`, `max_recap_ratio`, `min_saving_share` and `max_stale_ratio`.
- [ ] **A post-session report** on the dashboard: what was kept, compacted and archived, which decisions survived, the grounding failures seen, and what the session cost against no LLM-GC. It is the page people share, so it must make sense to someone who did not build LLM-GC.

## 3. Fix what blocks a release

- [ ] **Whatever the benchmark shows losing quality.** If a lost answer traces back to how a span was scored, fix that signal before release. One miss is already known: relevance matches words, so "datastore" never finds "database", and a span the conversation returns to in other words stays at its low level. Embedding relevance behind the existing `Relevance` seam fixes it without training anything.
- [ ] **Scoring runs on the proxy's event loop**, and is O(n²) in messages per pass. While a pass scores a long conversation, every other request the proxy handles waits. Make it linear, and move it off the loop.
- [ ] **A collection's summaries run one after another** (132 s cold on one live run). Run them concurrently if the latency measurement confirms they are the cost.
- [ ] **A retry policy for LLM-GC's own model calls:** retryable errors only (timeouts, 429, 5xx), exponential backoff with jitter, a cap on attempts, and `Retry-After` honoured. Today a failed call is shown as a degradation and a rate limit pauses collections, but nothing retries.
- [ ] **The dashboard groups conversations into sessions twice**, in its own script and on the server. Use the server's grouping only, so the two cannot disagree.
- [ ] **Small fixes:**
  - [ ] `--context-window`'s help still calls it the proxied model's window. It is the fallback used when the Models API does not know the model.
  - [ ] The README still titles the exchanges view "What the model knows, exchange by exchange"; the dashboard calls it "The conversation".
  - [ ] A span with no prose is named by its raw tool JSON. Name it by its tool and the first line of its result.
- [ ] **Rerun the benchmark** on the fixed code.

## 4. Ship

- [ ] **Choose a license**: Apache-2.0 includes an explicit patent grant; MIT is the simplest. Then set the `license` field in `pyproject.toml`.
- [ ] **Format the code base once with `ruff format`** (128 files change today), in a commit with nothing else in it, listed in `.git-blame-ignore-revs` so `git blame` skips it. Then add **pre-commit hooks** for Ruff.
- [ ] **Finish CI.** Pull requests already run `ruff check`, `mypy` and `pytest` on Python 3.13. Still to add: `ruff format --check` (after the formatting commit), every supported Python version, the benchmark's regression check, and a rule that the checks must pass before a merge.
- [ ] **Optional extras.** The proxy, dashboard and CLI dependencies (`fastapi`, `uvicorn`, `click`, `pydantic-settings`) move to `llm-gc[proxy]`, so someone using only the library does not install a web server.
- [ ] **Integration tests:** the proxy tested end to end over HTTP against a stand-in upstream, async tests on `pytest-asyncio`, and coverage reported.
- [ ] **A Docker image** that starts a working proxy, with its port published to localhost only by default.
- [ ] **PyPI publishing through trusted publishing**, with no long-lived token.
- [ ] **Dependency scanning:** Dependabot for the Python dependencies (today it watches only the GitHub Actions), and `pip-audit` in CI.
- [ ] **Project files:** `SECURITY.md` (with a private channel for reports), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `CHANGELOG.md` in the Keep a Changelog format.
- [ ] **What data LLM-GC keeps, documented:** which conversation data is held, where, for how long, and how to delete it.
- [ ] **A README for a first-time reader:** the 30-second demo, installation, a quickstart, and the benchmark's numbers with a link to how they were produced.

## Known limits of the first release

Each is accepted for the first release, stated in the docs, and fixed after it.

- **Sessions live in memory.** A restart costs each conversation one whole request and one collection's model calls.
- **A very large tool result is sent whole** while it is among the newest turns kept, and is compacted only once it is older.
- **The proxy speaks only Anthropic's wire format.** Clients that speak OpenAI's cannot use it yet.
- **Without a model, facts are read by patterns** that catch few phrasings. This affects offline runs only.
- **A relayed subagent report is not tied to the subagent that wrote it.**
- **Facts that failed to be read are not read again** by a later collection; their span stays marked.

## After the first version release

- **An OpenAI-compatible endpoint** (`/v1/chat/completions`) on the same collection path. It waits because the release's claims would first be measured on Claude Code; a second wire format without its own measurement would ship unmeasured.
- **Sessions that survive a restart.**

The learned models below are trained and judged on labels the benchmark produces, so none of them can start before it. Each replaces today's method only if it beats it there.

- **A learned relevance scorer.** Today 40% of a span's score comes from counting decision phrases and specifics (names, numbers, code) with patterns.
- **A learned hallucination-risk predictor, evaluated first.** It needs labelled grounding failures, which only the benchmark and live runs produce. It ships only if it can be grounded in real labels; otherwise the individual signals stay the permanent design, and the negative result is written up.
- **A small local fact reader**, for runs without a model, or to make the facts call cheaper. With a model configured, facts are already read by the model.
- **A summary-quality evaluator** that scores each summary directly, beyond what the benchmark's replay shows.
