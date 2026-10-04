"""One conversation's memory never reaches another conversation's prompt.

Every other engine test drives a single session, which is how a shared old and
permanent generation went unnoticed until the proxy served two users at once.
"""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import NoOpCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.sweep import SweepClassification, SweepEntry, SweepResult
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import Message
from llm_gc.scoring.relevance_scorer_result import RelevanceScorerResult
from llm_gc.session import SessionManager

ALICE_SECRET = "the patient records database lives in frankfurt"


class _KeepScorer:
    def score(self, message, conversation):
        return RelevanceScorerResult(message=message, combined_score=1.0, scorer_results=[])


class _SweepFirstTurn:
    """Ages turn 0 of every session with the given classification, keeps the rest."""

    def __init__(self, classification: SweepClassification) -> None:
        self.classification = classification

    def sweep(self, conversation, relevance_scores):
        entries = [
            SweepEntry(message=m, relevance_score=0.5, turn_index=m.turn_index,
                       classification=self.classification if m.turn_index == 0 else SweepClassification.KEEP)
            for m in conversation
        ]
        counts = {c: sum(1 for e in entries if e.classification == c) for c in SweepClassification}
        return SweepResult(sweep_entries=entries, classification_counts=counts, processing_time_ms=0.0,
                           total_keep_tokens=0, total_compact_tokens=0, total_archive_tokens=0)


def _service(classification: SweepClassification) -> GcService:
    bus = EventBus()
    # gc_threshold=0.0: always under pressure, so every update ages.
    config = GCConfig(context_window=1000, max_memory_ratio=0.5, min_compactable_tokens=1,
                      min_cut_tokens=0, last_n_turns_to_keep=1)
    return GcService(
        session_manager=SessionManager(),
        generational_memory=GenerationalMemory(
            knowledge_extractor=KnowledgeExtractor(event_bus=bus),
            compactor=NoOpCompactor(event_bus=bus),
        ),
        relevance_scorer=_KeepScorer(),
        sweeper=_SweepFirstTurn(classification),
        event_bus=bus,
        gc_config=config,
    )


def _turn(index: int, content: str) -> Message:
    return Message(role="user" if index % 2 == 0 else "assistant", content=content,
                   token_count=len(content.split()), turn_index=index)


async def _alice_ages_her_first_turn(service: GcService) -> None:
    await service.gc_update("alice", _turn(0, ALICE_SECRET))
    await service.gc_update("alice", _turn(1, "noted, frankfurt it is"))


def _text(messages: list[Message]) -> str:
    return "\n".join(m.content for m in messages)


def test_archived_facts_stay_in_their_own_session():
    service = _service(SweepClassification.ARCHIVE)

    async def scenario():
        await _alice_ages_her_first_turn(service)
        await service.gc_update("bob", _turn(0, "hello, I am planning a party"))
        alice = await service.gc_collect("alice")
        bob = await service.gc_collect("bob")
        return alice, bob

    alice, bob = asyncio.run(scenario())
    assert "frankfurt" in _text(alice), "precondition: Alice's own memory is injected for her"
    assert "frankfurt" not in _text(bob), "Alice's archived fact reached Bob's prompt"


def test_old_gen_summaries_stay_in_their_own_session():
    service = _service(SweepClassification.COMPACT)

    async def scenario():
        await _alice_ages_her_first_turn(service)
        await service.gc_update("bob", _turn(0, "hello, I am planning a party"))
        return await service.gc_collect("alice"), await service.gc_collect("bob")

    alice, bob = asyncio.run(scenario())
    assert "frankfurt" in _text(alice), "precondition: Alice's summary is in her context"
    assert "frankfurt" not in _text(bob), "Alice's old-gen summary reached Bob's context"


def test_a_removed_session_takes_its_memory_with_it():
    """Deleting a conversation must delete everything derived from it. A new
    conversation that happens to reuse the id starts empty."""
    service = _service(SweepClassification.ARCHIVE)

    async def scenario():
        await _alice_ages_her_first_turn(service)
        service.session_manager.store.remove("alice")
        await service.gc_update("alice", _turn(0, "a brand new conversation"))
        return await service.gc_collect("alice")

    assert "frankfurt" not in _text(asyncio.run(scenario()))


def test_the_stateless_composer_keeps_conversations_apart():
    """The library path had the same leak: `ContextComposer` archived into a shared
    store, so one composer serving two conversations mixed their facts."""
    from llm_gc.engine.context_composer import ContextComposer

    bus = EventBus()
    composer = ContextComposer(
        event_bus=bus,
        generational_memory=GenerationalMemory(knowledge_extractor=KnowledgeExtractor(event_bus=bus),
                                               compactor=NoOpCompactor(event_bus=bus)),
        compactor=NoOpCompactor(event_bus=bus),
    )
    sweep = _SweepFirstTurn(SweepClassification.ARCHIVE)

    def compose(turns: list[Message]):
        return asyncio.run(composer.compose(sweep.sweep(turns, [])))

    alice = compose([_turn(0, ALICE_SECRET), _turn(1, "noted")])
    bob = compose([_turn(0, "hello, I am planning a party"), _turn(1, "fun")])

    assert any("frankfurt" in e.content for e in alice.archived_knowledge), "precondition"
    assert not any("frankfurt" in e.content for e in bob.archived_knowledge)
