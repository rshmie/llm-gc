# Contributing to LLM-GC

Thank you for helping. This page says how to set up, what a change needs, and what to expect in review. By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

If you have found a security problem, do not open an issue. Follow [SECURITY.md](SECURITY.md) instead.

## Before you start

For anything bigger than a small fix, open an issue first and say what problem you want to solve and how. The design docs under [`doc/design/`](doc/design/) record why each part is shaped as it is, including the alternatives that were rejected. A change that goes against one of them needs that conversation before the code, not after.

## Set up

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.13 for you.

```bash
git clone https://github.com/rshmie/llm-gc.git
cd llm-gc
uv sync                    # LLM-GC, the proxy extra and the development tools
source .venv/bin/activate
pre-commit install         # runs Ruff on each commit
```

The tests need no network and no API key.

## The checks

CI runs these four on every pull request, and a change should pass all of them before you open one:

```bash
ruff check .            # lint
ruff format --check .   # formatting; `ruff format .` fixes it
mypy                    # types, strict, for the whole package
pytest                  # tests
```

CI also installs the library alone, with no extras, and checks that it imports and that the `llm-gc` command says what to install. And a weekly job audits the locked dependencies for known vulnerabilities.

## How the code is organised

LLM-GC is a library (`llm_gc`) and an application built on it.

- **The library** is everything under `src/llm_gc/` except the four packages below. Other programs import it, so it must work with only `pydantic`, `tiktoken` and `httpx2` installed.
- **The application** is `cli/`, `proxy/`, `visualizer/` and `web/`. It needs the `proxy` extra: the web server, the settings loader and `click`.
- **The application depends on the library, never the reverse.** A test imports every library module with the extra's packages made unavailable, so a library module that imports one fails it.
- **The public API is what `llm_gc/__init__.py` exports.** Everything else is internal and may change in any release.
- A new dependency goes in `pyproject.toml` with a floor (the version the lockfile tests) and a ceiling below the next major version. If only the application uses it, it belongs in the `proxy` extra.

## What a change needs

- **A test.** For a bug, a test that fails without the fix. Tests never call a real model: use `FakeProvider`, or `httpx2.MockTransport` for HTTP. Test the failure path as well as the good one.
- **The design doc, when the change alters how something works or why.** Design docs describe the system as it is now. Rewrite the sentence that is no longer true instead of adding history, and give a rejected alternative one line with its reason.
- **Visible degradation.** If a fallback is taken, the result and an event say so. Nothing in LLM-GC quietly does a worse job.
- **No conversation text and no credentials in logs or events.** Conversation text is the user's data. Log counts, ids and names.
- **Plain comments.** A comment or docstring is a full sentence that says what the code does and why. Longer reasoning belongs in the design doc.
- **Names and constants.** Functions are named verb-first, and a number or string with a meaning is a named constant with a comment.
- **You understand every line.** AI coding tools are welcome. What they write must be tested and verified, and you must be able to understand it and justify every decision behind it.

## Commit messages and pull requests

Start the subject with what kind of change it is: `fix:`, `feat:`, `refactor:`, `docs:`, `ci:` or `style:` [commit-convention](https://www.conventionalcommits.org/en/v1.0.0/), then say in plain words what is different afterwards. The body is a short paragraph on why the change was needed, then a list of the behaviours that changed. Do not list files. The diff already does.

Keep a formatting-only change in a commit of its own. Its hash is added to `.git-blame-ignore-revs`, which GitHub reads, so `git blame` skips it. To skip it locally too, run `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

A pull request describes the problem, what you changed, and how you checked it. After the first release, a change a user can see also adds a line under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md). Until then the first release's entry will describe everything.
