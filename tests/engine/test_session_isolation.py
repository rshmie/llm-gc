"""One conversation's memory never reaches another conversation's prompt.

Every other engine test drives a single session, which is how a shared old and
permanent generation went unnoticed until the proxy served two users at once.
"""

import asyncio

from llm_gc.engine.compaction import LLMCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.events import EventBus
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import Message
from tests.engine.support import ARCHIVED_SCORE, COMPACTED_SCORE, make_service, turn

ALICE_SECRET = "The patient records database lives in frankfurt."

_SUMMARY_WITH_THE_SECRET = LLMResponse(
    text="The records database lives in frankfurt; retention was discussed.",
    stop_reason=StopReason.COMPLETE,
    input_tokens=0,
    output_tokens=12,
    model="scripted",
    provider="offline",
)


def _service(score: float) -> GcService:
    """Every session's turn 0 is condensed at `score`'s level as soon as it ages out."""
    compactor = LLMCompactor(event_bus=EventBus(), provider=FakeProvider(responses=[_SUMMARY_WITH_THE_SECRET] * 20))
    service, _, _ = make_service(scores={0: score}, compactor=compactor)
    return service


async def _alice_condenses_her_first_exchange(service: GcService) -> None:
    """Her question and its answer: a cut never separates them, so her next question
    is what lets the first exchange be condensed."""
    await service.gc_update("alice", turn(0, opening=ALICE_SECRET))
    await service.gc_update("alice", turn(1, role="assistant"))
    await service.gc_update("alice", turn(2))
    assert service.session_manager.store.get("alice").snapshot.old, "precondition: turns 0-1 were condensed"


def _text(messages: list[Message]) -> str:
    return "\n".join(m.content for m in messages)


def test_an_archived_span_stays_in_its_own_session():
    service = _service(ARCHIVED_SCORE)

    async def scenario():
        await _alice_condenses_her_first_exchange(service)
        await service.gc_update("bob", turn(0, opening="Hello, I am planning a party."))
        return await service.gc_collect("alice"), await service.gc_collect("bob")

    alice, bob = asyncio.run(scenario())
    assert "frankfurt" in alice[0].content, "precondition: Alice's archived line is in her front"
    assert "frankfurt" not in _text(bob), "Alice's archived span reached Bob's prompt"


def test_a_summary_stays_in_its_own_session():
    service = _service(COMPACTED_SCORE)

    async def scenario():
        await _alice_condenses_her_first_exchange(service)
        await service.gc_update("bob", turn(0, opening="Hello, I am planning a party."))
        return await service.gc_collect("alice"), await service.gc_collect("bob")

    alice, bob = asyncio.run(scenario())
    assert "frankfurt" in alice[0].content, "precondition: Alice's summary is in her front"
    assert "frankfurt" not in _text(bob), "Alice's summary reached Bob's prompt"


def test_a_removed_session_takes_its_memory_with_it():
    """Deleting a conversation must delete everything derived from it. A new
    conversation that happens to reuse the id starts empty."""
    service = _service(ARCHIVED_SCORE)

    async def scenario():
        await _alice_condenses_her_first_exchange(service)
        service.session_manager.store.remove("alice")
        await service.gc_update("alice", turn(0, opening="A brand new conversation."))
        return await service.gc_collect("alice")

    assert "frankfurt" not in _text(asyncio.run(scenario()))
