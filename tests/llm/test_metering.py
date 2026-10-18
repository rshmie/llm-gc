import asyncio

import pytest

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.gc_status import GCStatus
from llm_gc.events import EventBus
from llm_gc.exceptions import LLMRateLimitError
from llm_gc.llm import FakeProvider, LLMResponse, MeteredProvider, StopReason, measure_model_usage
from llm_gc.models import Message
from tests.engine.support import COMPACTED_SCORE, collect, make_service, turn

RESPONSE = LLMResponse(
    text="a summary", stop_reason=StopReason.COMPLETE, input_tokens=40, output_tokens=7, model="m", provider="fake"
)


def _call(provider: MeteredProvider):
    return provider.generate(
        messages=[Message(role="user", content="summarise", turn_index=0)], max_output_tokens=50, timeout_s=5.0
    )


def test_a_call_outside_any_measurement_is_answered_and_counted_nowhere():
    provider = MeteredProvider(FakeProvider(responses=[RESPONSE]))

    assert asyncio.run(_call(provider)) == RESPONSE


def test_calls_made_by_tasks_started_inside_a_measurement_land_in_it():
    provider = MeteredProvider(FakeProvider(responses=[RESPONSE] * 3))

    async def scenario():
        with measure_model_usage() as spent:
            await asyncio.gather(*(_call(provider) for _ in range(3)))
        return spent

    spent = asyncio.run(scenario())
    assert (spent.calls, spent.input_tokens, spent.output_tokens) == (3, 120, 21)


def test_two_measurements_running_at_once_never_mix():
    """Two conversations collected concurrently: a process-wide counter would credit each with the other's calls."""
    provider = MeteredProvider(FakeProvider(responses=[RESPONSE] * 5))

    async def collection(calls: int):
        with measure_model_usage() as spent:
            for _ in range(calls):
                await _call(provider)
                await asyncio.sleep(0)  # let the other collection run in between
        return spent.calls

    async def scenario():
        return await asyncio.gather(collection(2), collection(3))

    assert asyncio.run(scenario()) == [2, 3]


def test_a_call_that_raises_adds_nothing():
    provider = MeteredProvider(FakeProvider(responses=[LLMRateLimitError("slow down")]))

    async def scenario():
        with measure_model_usage() as spent:
            with pytest.raises(LLMRateLimitError):
                await _call(provider)
        return spent

    assert asyncio.run(scenario()).calls == 0


def test_a_collection_reports_what_its_own_model_calls_spent():
    summary = turn(0).token_count
    provider = FakeProvider(responses=[RESPONSE] * 20)
    service, _, _ = make_service(
        GCConfig(min_cut_tokens=summary, last_n_messages_to_keep=1),
        compactor=LLMCompactor(event_bus=EventBus(), provider=MeteredProvider(provider)),
        default_score=COMPACTED_SCORE,
    )

    async def scenario():
        results = []
        for index in range(3):
            state = service.session_manager.get_or_create("s")
            state.snapshot = state.snapshot.with_turn(turn(index))
            results.append(await collect(service, "s"))
        return results

    results = asyncio.run(scenario())
    completed = [result for result in results if result.status is GCStatus.COMPLETED]
    assert completed, [result.status for result in results]
    assert sum(result.summary_calls for result in results) == len(provider.calls) > 0
    assert sum(result.summary_output_tokens for result in results) == 7 * len(provider.calls)
    assert all(result.summary_calls == 0 for result in results if result.status is GCStatus.BYPASSED_BATCH_TOO_SMALL)
