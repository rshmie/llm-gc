"""The decision-survival runner, driven offline: the real proxy and engine, stand-in models."""

import argparse
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
from llm_gc.engine import build_gc_service
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


class TestOneEngineForARun:
    def test_each_conversation_keeps_its_own_session_and_degradations(self):
        """One engine serves a run so a dashboard can show it; its bus is shared, so a
        conversation's warnings must not leak into another's, already returned, report."""
        truncated = LLMResponse(text="partial", stop_reason=StopReason.TRUNCATED, input_tokens=0, output_tokens=1,
                                model="m", provider="p")
        provider = FakeProvider(responses=[truncated] * 200)
        service = build_gc_service(CONFIG, summary_provider=provider)
        first, second = load_conversations()[:2]

        def run(conversation):
            return asyncio.run(run_conversation(
                conversation, config=CONFIG, summary_provider=provider, service=service,
                model_transport=rehearsal_model(conversation), api_key="rehearsal",
                model="claude-opus-5-5", effort="medium"))

        earlier = run(first)
        before = list(earlier.degradations)
        later = run(second)

        assert earlier.degradations == before, "the second conversation added to the first's report"
        assert later.degradations
        assert service.session_manager.store.get(first.id) and service.session_manager.store.get(second.id)


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


def test_a_run_says_what_it_is_doing_at_each_step():
    """A long conversation plays for minutes; without steps the run looks stuck."""
    steps = []
    asyncio.run(run_conversation(
        CONVERSATION, config=CONFIG, summary_provider=rehearsal_summaries(),
        model_transport=rehearsal_model(CONVERSATION), api_key="rehearsal", model="claude-opus-5-5",
        effort="medium", on_step=steps.append))

    assert steps[0] == f"turn 1/{len(CONVERSATION.turns)} · 0 collections so far"
    assert steps[-2:] == ["asking the questions through LLM-GC", "asking them with the whole conversation"]
    assert any(not step.endswith(" 0 collections so far") for step in steps if step.startswith("turn")), \
        "the collection count moves as the conversation plays"


def test_a_failed_conversation_is_reported_and_the_others_still_count(monkeypatch):
    """One provider error used to end the run and discard every conversation already paid for."""
    rejecting = httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error",
                                                                      "message": "rejected for the test"}})
    real = cli.rehearsal_model
    monkeypatch.setattr(cli, "rehearsal_model", lambda conversation: (
        (lambda: httpx2.MockTransport(lambda request: rejecting)) if conversation.id == "clinic-booking"
        else real(conversation)))
    args = argparse.Namespace(rehearse=True, model="claude-opus-5-5", effort="medium", min_cut_tokens=300,
                              keep_last=5, conversation=["ingest-service", "clinic-booking"], dashboard=False,
                              dashboard_port=0, out=None)
    published = []

    asyncio.run(cli.check(args, None, published.append))

    ingest, clinic = published[0]["conversations"]
    assert "failed" not in ingest and "rejected for the test" in clinic["failed"]
    assert published[0]["totals"]["probes"] == len(ingest["probes"]), "the failure is kept out of the totals"
