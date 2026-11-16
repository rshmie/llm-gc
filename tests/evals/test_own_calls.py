"""The benchmark's record of LLM-GC's own model calls."""

import asyncio

import pytest

from evals.benchmark.own_calls import OwnCallRecorder
from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.exceptions import LLMRateLimitError
from llm_gc.llm import FakeProvider, LLMResponse, ModelUsage, StopReason
from llm_gc.llm.fake_provider import FAKE_MODEL_NAME
from llm_gc.models import Message
from tests.engine.support import collect, turn


def _response(model: str, input_tokens: int = 40, output_tokens: int = 7) -> LLMResponse:
    return LLMResponse(
        text="a summary",
        stop_reason=StopReason.COMPLETE,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        provider="fake",
    )


def _call(provider: OwnCallRecorder):
    return provider.generate(
        messages=[Message(role="user", content="summarise", turn_index=0)], max_output_tokens=50, timeout_s=5.0
    )


def test_calls_are_kept_under_the_model_that_answered_them():
    """A gateway can answer with a model other than the one named, and the bill is the answering model's."""
    recorder = OwnCallRecorder(
        FakeProvider(responses=[_response("haiku"), _response("haiku", 100, 20), _response("sonnet", 500, 50)])
    )

    async def scenario():
        for _ in range(3):
            await _call(recorder)

    asyncio.run(scenario())
    assert recorder.take() == {
        "haiku": ModelUsage(calls=2, input_tokens=140, output_tokens=27),
        "sonnet": ModelUsage(calls=1, input_tokens=500, output_tokens=50),
    }


def test_a_take_returns_only_what_was_spent_since_the_last_one():
    recorder = OwnCallRecorder(FakeProvider(responses=[_response("m")] * 3))

    async def scenario():
        await _call(recorder)
        first = recorder.take()
        await _call(recorder)
        await _call(recorder)
        return first, recorder.take(), recorder.take()

    first, second, third = asyncio.run(scenario())
    assert first == {"m": ModelUsage(calls=1, input_tokens=40, output_tokens=7)}
    assert second == {"m": ModelUsage(calls=2, input_tokens=80, output_tokens=14)}
    assert third == {}


def test_a_call_still_running_at_a_take_is_billed_to_the_next_one():
    """Late but never lost. A recorder that chose its count before awaiting the call would add it to one already taken."""

    async def scenario():
        answer = asyncio.Event()

        class SlowProvider:
            name = "slow"

            async def generate(self, **_: object) -> LLMResponse:
                await answer.wait()
                return _response("m")

        recorder = OwnCallRecorder(SlowProvider())
        call = asyncio.create_task(_call(recorder))
        await asyncio.sleep(0)  # the call is now waiting on the provider
        during = recorder.take()
        answer.set()
        await call
        return during, recorder.take()

    during, after = asyncio.run(scenario())
    assert during == {}
    assert after == {"m": ModelUsage(calls=1, input_tokens=40, output_tokens=7)}


def test_a_call_that_raises_is_passed_on_and_adds_nothing():
    recorder = OwnCallRecorder(FakeProvider(responses=[LLMRateLimitError("slow down")]))

    async def scenario():
        with pytest.raises(LLMRateLimitError):
            await _call(recorder)

    asyncio.run(scenario())
    assert recorder.take() == {}


def test_handed_to_the_default_engine_it_records_what_the_engine_counts_for_its_collections():
    """The two counts are kept independently, so agreeing is a check on both."""
    recorder = OwnCallRecorder(FakeProvider())
    service = build_gc_service(GCConfig(min_cut_tokens=0, last_n_messages_to_keep=1), summary_provider=recorder)

    async def scenario():
        results = []
        for index in range(8):
            state = service.session_manager.get_or_create("s")
            state.snapshot = state.snapshot.with_turn(turn(index, role="user" if index % 2 == 0 else "assistant"))
            results.append(await collect(service, "s", ignore_batch_minimum=True))
        return results

    results = asyncio.run(scenario())
    engine_count = ModelUsage(
        calls=sum(result.summary_calls for result in results),
        input_tokens=sum(result.summary_input_tokens for result in results),
        output_tokens=sum(result.summary_output_tokens for result in results),
    )
    assert engine_count.calls > 0
    assert recorder.take() == {FAKE_MODEL_NAME: engine_count}
