"""Run the decision-survival check and report which facts LLM-GC kept.

    python -m evals.decision_survival.run --rehearse     # offline: no key, no cost
    ANTHROPIC_API_KEY=... python -m evals.decision_survival.run

How to read the numbers: README.md beside this file.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx2
import uvicorn

from evals.decision_survival.dataset import load_conversations
from evals.decision_survival.grading import Presence, Verdict, grade, presence, split_answers
from evals.decision_survival.harness import Effort, Run, rehearsal_facts, rehearsal_model, rehearsal_summaries, run_conversation
from evals.decision_survival.progress import Progress, plural
from llm_gc.config import GCConfig
from llm_gc.engine import GcService, build_gc_service
from llm_gc.llm import AnthropicClient, LLMProvider
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.visualizer import TranscriptRecorder, create_dashboard_app

DEFAULT_MODEL = "claude-sonnet-5-5" #"claude-opus-5-5"
# As `run_proxy.py --serve` runs it; the memory budget is a share of this window.
CONTEXT_WINDOW = 200_000
RUNS_DIR = Path(__file__).with_name("runs")


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    presence: Presence
    with_gc: Verdict
    without_gc: Verdict
    answer_with_gc: str | None
    answer_without_gc: str | None


def grade_run(run: Run) -> list[ProbeResult]:
    probes = run.conversation.probes
    with_gc = split_answers(run.with_gc.text, len(probes))
    without_gc = split_answers(run.without_gc.text, len(probes))
    return [ProbeResult(probe=probe.id, presence=presence(probe, run.forwarded), with_gc=grade(ours, probe),
                        without_gc=grade(theirs, probe), answer_with_gc=ours, answer_without_gc=theirs)
            for probe, ours, theirs in zip(probes, with_gc, without_gc, strict=True)]


def totals(results: Sequence[ProbeResult]) -> dict[str, int]:
    return {
        "probes": len(results),
        "right_with_gc": sum(r.with_gc is Verdict.RIGHT for r in results),
        "right_without_gc": sum(r.without_gc is Verdict.RIGHT for r in results),
        # The headline: what the model knew from the whole conversation and not from ours.
        "lost_to_gc": sum(r.without_gc is Verdict.RIGHT and r.with_gc is not Verdict.RIGHT for r in results),
        "absent_from_prompt": sum(r.presence is Presence.ABSENT for r in results),
        # In the prompt and still answered wrong: a presentation problem, not a loss.
        "present_but_missed": sum(r.presence is not Presence.ABSENT and r.with_gc is not Verdict.RIGHT
                                  for r in results),
    }


async def check(args: argparse.Namespace, api_key: str | None, publish: Callable[[dict[str, Any]], None]) -> None:
    config = GCConfig(context_window=CONTEXT_WINDOW, min_cut_tokens=args.min_cut_tokens,
                      last_n_turns_to_keep=args.keep_last)
    conversations = [c for c in load_conversations() if not args.conversation or c.id in args.conversation]
    report: dict[str, Any] = {"settings": _settings(args), "conversations": []}
    every_probe: list[ProbeResult] = []
    summaries: LLMProvider = rehearsal_summaries() if api_key is None else AnthropicClient(api_key=api_key)
    # One engine for the run, each conversation in its own session, so one dashboard can show them all.
    service = build_gc_service(config, summary_provider=summaries,
                               fact_reader=rehearsal_facts() if api_key is None else None)
    transcripts = TranscriptRecorder() if args.dashboard else None
    dashboard = _serve_dashboard(service, config, transcripts, args.dashboard_port) if args.dashboard else None
    progress = Progress(total=len(conversations))
    print(f"{plural(len(conversations), 'conversation')} to check")
    try:
        for conversation in conversations:
            progress.begin(conversation.id, "starting")
            try:
                run = await run_conversation(
                    conversation, config=config, summary_provider=summaries, service=service,
                    on_client_history=transcripts.record if transcripts is not None else None,
                    on_step=progress.update,
                    model_transport=rehearsal_model(conversation) if api_key is None else _real_model,
                    api_key=api_key or "rehearsal", model=args.model, effort=cast(Effort, args.effort))
            except Exception as err:
                # A provider error or a dropped connection costs this conversation, not the
                # ones already paid for: it is reported, kept out of the totals, and the run fails.
                failure = f"{type(err).__name__}: {err}"
                report["conversations"].append({"conversation": conversation.id, "failed": failure})
                progress.end(True, f"FAILED, {failure[:160]},")
                continue
            results = grade_run(run)
            every_probe += results
            report["conversations"].append(_describe(run, results))
            counted = totals(results)
            progress.end(True, f"{plural(run.cuts, 'collection')}, right {counted['right_with_gc']}/{counted['probes']} "
                               f"with LLM-GC and {counted['right_without_gc']}/{counted['probes']} sent whole,")
        report["totals"] = totals(every_probe)
        publish(report)
        if dashboard is not None:
            print(f"The dashboard stays open at http://127.0.0.1:{args.dashboard_port} to look at each "
                  "conversation's sessions. Ctrl-C to stop.")
            await dashboard[1]
    finally:
        if dashboard is not None and not dashboard[1].done():
            # Stopped, not cancelled: a cancelled server logs a traceback that buries the real error.
            dashboard[0].should_exit = True
            await dashboard[1]
        if isinstance(summaries, AnthropicClient):
            await summaries.aclose()


def _serve_dashboard(service: GcService, config: GCConfig, transcripts: TranscriptRecorder | None,
                     port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    store = service.session_manager.store
    monitor = ContextHealthMonitor(
        event_bus=service.event_bus, gc_config=config,
        read_generations=lambda session_id: (state.snapshot if session_id is not None
                                             and (state := store.get(session_id)) is not None else None))
    app = create_dashboard_app(service, monitor, transcripts=transcripts)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    print(f"dashboard: http://127.0.0.1:{port} (each conversation is a session; it fills in as the check runs)")
    return server, asyncio.create_task(server.serve())


def print_report(report: dict[str, Any]) -> None:
    settings = report["settings"]
    print(f"\n{'REHEARSAL (offline stand-ins; checks the path, not the model)' if settings['rehearsal'] else ''}"
          f"\nmodel {settings['model']} (effort {settings['effort']}) · min_cut_tokens {settings['min_cut_tokens']}"
          f" · keep_last {settings['keep_last']} · code {settings['code_version']}")
    for item in report["conversations"]:
        if "failed" in item:
            print(f"\n{item['conversation']}: FAILED, so its questions are not in the totals\n  {item['failed']}")
            continue
        tokens = item["prompt_tokens"]
        saved = 1 - tokens["with_gc"] / tokens["without_gc"] if tokens["without_gc"] else 0.0
        print(f"\n{item['conversation']}: {item['cuts']} cuts · prompt {tokens['with_gc']:,} tokens "
              f"vs {tokens['without_gc']:,} sent whole ({saved:.0%} smaller)")
        for warning in item["warnings"]:
            print(f"  WARNING: {warning}")
        width = max(len("probe"), *(len(probe["probe"]) for probe in item["probes"]))
        print(f"  {'probe':<{width}}  {'in the prompt':<14} {'with LLM-GC':<12} sent whole")
        for probe in item["probes"]:
            print(f"  {probe['probe']:<{width}}  {probe['presence']:<14} {probe['with_gc']:<12} {probe['without_gc']}")
    t = report["totals"]
    failed = [item["conversation"] for item in report["conversations"] if "failed" in item]
    if failed:
        print(f"\nINCOMPLETE: {', '.join(failed)} failed; the totals cover the other conversations only.")
    print(f"\nright with LLM-GC {t['right_with_gc']}/{t['probes']} · sent whole {t['right_without_gc']}/{t['probes']}"
          f" · lost to LLM-GC {t['lost_to_gc']} · absent from the prompt {t['absent_from_prompt']}"
          f" · present but missed {t['present_but_missed']}\n")


def _describe(run: Run, results: Sequence[ProbeResult]) -> dict[str, Any]:
    by_reason = Counter(degradation.split(":", 1)[0] for degradation in run.degradations)
    warnings = [f"a clean-up degraded {count}x with {reason}, e.g. "
                f"{next(d for d in run.degradations if d.startswith(reason)).split(': ', 1)[1]}"
                for reason, count in by_reason.items()]
    if run.bypassed:
        warnings.append(f"{run.bypassed} request(s) bypassed GC, so less GC was measured than configured")
    warnings += [f"the {arm} reply stopped with {answer.stop_reason!r}" for arm, answer
                 in (("LLM-GC", run.with_gc), ("whole", run.without_gc)) if answer.stop_reason != "end_turn"]
    return {
        "conversation": run.conversation.id, "cuts": run.cuts, "warnings": warnings,
        "degradations": run.degradations,
        "prompt_tokens": {"with_gc": run.with_gc.input_tokens, "without_gc": run.without_gc.input_tokens},
        "probes": [asdict(result) for result in results],
        "replies": {"with_gc": run.with_gc.text, "without_gc": run.without_gc.text},
        "forwarded": run.forwarded,
    }


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {"rehearsal": args.rehearse, "model": args.model, "effort": args.effort,
            "min_cut_tokens": args.min_cut_tokens, "keep_last": args.keep_last, "context_window": CONTEXT_WINDOW,
            "code_version": _code_version(), "started_at": datetime.now(UTC).isoformat(timespec="seconds")}


def _code_version() -> str | None:
    """The commit measured, marked dirty when uncommitted changes were part of the run."""
    try:
        done = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).parent)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def _real_model() -> httpx2.AsyncBaseTransport:
    return httpx2.AsyncHTTPTransport()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rehearse", action="store_true", help="offline stand-ins for every model call")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the model asked the questions")
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--min-cut-tokens", type=int, default=300)
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--conversation", action="append", help="run only this conversation id (repeatable)")
    parser.add_argument("--out", type=Path, help="results file (default: a new file in runs/)")
    parser.add_argument("--dashboard", action="store_true", help="serve a live dashboard of the check's own engine")
    parser.add_argument("--dashboard-port", type=int, default=9911, help="apart from the proxy's own dashboard")
    args = parser.parse_args()

    api_key = None if args.rehearse else os.environ.get("ANTHROPIC_API_KEY")
    if not args.rehearse and not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Set it, or pass --rehearse to run offline.")

    published: list[dict[str, Any]] = []

    def publish(report: dict[str, Any]) -> None:
        published.extend(report["conversations"])
        print_report(report)
        out = args.out or RUNS_DIR / f"{datetime.now(UTC):%Y%m%d-%H%M%S}{'-rehearsal' if args.rehearse else ''}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"results: {out}")

    try:
        asyncio.run(check(args, api_key, publish))
    except KeyboardInterrupt:
        # uvicorn re-raises the Ctrl-C after it has stopped the dashboard.
        print("\nstopped")
    if any("failed" in item for item in published):
        sys.exit(1)


if __name__ == "__main__":
    main()
