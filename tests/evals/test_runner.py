"""The decision-survival runner, driven offline: the real proxy and engine, stand-in models."""

import asyncio
import json

import httpx2
import pytest

from evals.decision_survival import run as cli
from evals.decision_survival.dataset import load_conversations
from evals.decision_survival.grading import Presence, Verdict
from evals.decision_survival.harness import (
    Answer,
    Run,
    rehearsal_answer,
    rehearsal_model,
    rehearsal_summaries,
    run_conversation,
)
from evals.decision_survival.run import ProbeResult, grade_run, totals
from llm_gc.config import GCConfig
from llm_gc.engine.generations.front import FRONT_HEADER
from llm_gc.llm import FakeProvider, LLMResponse, StopReason

CONVERSATION = load_conversations()[0]
CONFIG = GCConfig(context_window=200_000, min_cut_tokens=300, last_n_turns_to_keep=5)


def _run(summary_provider=None, model_transport=None) -> Run:
    return asyncio.run(run_conversation(
        CONVERSATION, config=CONFIG, summary_provider=summary_provider or rehearsal_summaries(),
        model_transport=model_transport or rehearsal_model(CONVERSATION),
        api_key="rehearsal", model="claude-opus-5-5", effort="medium"))


class TestARehearsedRun:
    def test_the_scripted_turns_never_reach_the_model(self):
        """Only the questions may cost a model call; a leak would bill every scripted turn."""
        calls = []
        answer = rehearsal_answer(CONVERSATION)

        def counting(request: httpx2.Request) -> httpx2.Response:
            calls.append(json.loads(request.content)["messages"])
            return answer(request)

        _run(model_transport=lambda: httpx2.MockTransport(counting))

        assert len(calls) == 2   # once through LLM-GC, once sent whole
        assert all(messages[-1]["content"].startswith("Answer each question") for messages in calls)

    def test_it_cuts_and_sends_the_model_less_than_the_whole_conversation(self):
        run = _run()

        assert run.cuts >= 1
        assert run.bypassed == 0
        assert run.degradations == []
        assert run.forwarded[0]["content"].startswith(FRONT_HEADER)
        assert run.with_gc.input_tokens < run.without_gc.input_tokens

    def test_the_control_arm_sees_every_answer(self):
        """The rehearsal model answers from what it was sent, so the whole conversation scores full marks."""
        results = grade_run(_run())

        assert all(result.without_gc is Verdict.RIGHT for result in results)

    def test_a_refused_summary_is_reported_not_hidden(self):
        truncated = LLMResponse(text="partial", stop_reason=StopReason.TRUNCATED, input_tokens=0, output_tokens=1,
                                model="m", provider="p")
        run = _run(summary_provider=FakeProvider(responses=[truncated] * 50))

        assert any(degradation.startswith("compaction_refused: ") for degradation in run.degradations)
        assert any("compaction_refused" in warning for warning in cli._describe(run, grade_run(run))["warnings"])


class TestTotals:
    def test_right_only_with_gc_is_not_a_loss(self):
        results = [_result(with_gc=Verdict.RIGHT, without_gc=Verdict.WRONG),
                   _result(with_gc=Verdict.UNKNOWN, without_gc=Verdict.RIGHT, presence=Presence.ABSENT)]

        counted = totals(results)

        assert counted["lost_to_gc"] == 1
        assert counted["right_with_gc"] == counted["right_without_gc"] == 1

    def test_present_but_missed_counts_only_answers_that_were_in_the_prompt(self):
        results = [_result(with_gc=Verdict.WRONG, presence=Presence.CONDENSED),
                   _result(with_gc=Verdict.UNKNOWN, presence=Presence.ABSENT)]

        counted = totals(results)

        assert counted["present_but_missed"] == 1
        assert counted["absent_from_prompt"] == 1


def test_a_question_left_unanswered_is_graded_missing():
    run = _run()
    reply = "\n".join(run.with_gc.text.splitlines()[:-1])   # drop the last numbered line
    truncated = Run(**{**run.__dict__, "with_gc": Answer(text=reply, input_tokens=0, stop_reason="max_tokens")})

    assert grade_run(truncated)[-1].with_gc is Verdict.MISSING


def test_a_live_run_without_a_key_refuses_to_start(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["run"])

    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.main()


def _result(*, with_gc: Verdict, without_gc: Verdict = Verdict.RIGHT,
            presence: Presence = Presence.VERBATIM) -> ProbeResult:
    return ProbeResult(probe="p", presence=presence, with_gc=with_gc, without_gc=without_gc,
                       answer_with_gc=None, answer_without_gc=None)
