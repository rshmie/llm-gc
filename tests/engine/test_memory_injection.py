"""Archived turns must reach the model, not just the dashboard.

An archived turn whose facts are stored but never sent is a deletion with a
receipt: visible on the dashboard, unusable by the model - the same loss as the
sliding window this project exists to beat, and worse, because the transparency
layer would report the fact as retained. Facts reach the model inside the front,
chosen when a cut is made, so the front is identical on every request between cuts.
"""

import asyncio

import pytest
from pydantic import ValidationError

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import NoOpCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory, LexicalFactRetriever
from llm_gc.engine.generations.front import FRONT_HEADER
from llm_gc.engine.generations.memory_block import MEMORY_BLOCK_HEADER
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus, EventType
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import AgingDelta, apply_aging
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.scoring import RecencyScorer, RelevanceScorer, RelevanceType
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens

SESSION = "s"


def _config(**overrides) -> GCConfig:
    base = {
        "context_window": 200,
        "keep_threshold": 0.55,
        "archive_threshold": 0.3,
        "min_compactable_tokens": 10,
        "last_n_turns_to_keep": 2,
        "min_cut_tokens": 1000,     # no cut during these tests: facts are placed directly
        "max_memory_ratio": 0.6,
    }
    base.update(overrides)
    return GCConfig(**base)


def _build(config: GCConfig, retriever=None) -> GcService:
    bus = EventBus()
    return GcService(
        session_manager=SessionManager(),
        generational_memory=GenerationalMemory(knowledge_extractor=KnowledgeExtractor(event_bus=bus),
                                               compactor=NoOpCompactor(event_bus=bus)),
        relevance_scorer=RelevanceScorer(
            scorers=[RecencyScorer(decay_rate=0.3)], weights={RelevanceType.RECENCY: 1.0}, event_bus=bus
        ),
        sweeper=Sweeper(sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus),
        event_bus=bus,
        gc_config=config,
        fact_retriever=retriever,
    )


def _store(service: GcService, topic: str, content: str, turn: int,
           kind: KnowledgeType = KnowledgeType.FACT) -> None:
    """Put a fact straight into the session's permanent generation.

    Driving a whole conversation through the sweeper to produce one archived turn
    makes a test about *injection* depend on every threshold in the config - which
    is how a previous draft of these tests silently measured nothing. Facts are
    placed directly; the aging path has its own tests.
    """
    entry = KnowledgeEntry(message_turn=turn, content=content, topic_label=topic, knowledge_type=kind)
    state = service.session_manager.get_or_create(SESSION)
    state.snapshot = apply_aging(state.snapshot, AgingDelta(new_knowledge=(entry,))).snapshot


def _young(service: GcService, texts: list[str], start: int = 50) -> None:
    state = service.session_manager.get_or_create(SESSION)
    for offset, text in enumerate(texts):
        state.snapshot = state.snapshot.with_turn(
            Message(role="user", content=text, token_count=count_tokens(text), turn_index=start + offset)
        )


def _collect(service: GcService) -> list[Message]:
    return asyncio.run(service.gc_collect(SESSION))


def _front(context: list[Message]) -> Message | None:
    return context[0] if context and context[0].turn_index == -1 else None


class TestAnArchivedFactReachesThePrompt:
    def test_a_stored_fact_is_in_the_front(self):
        service = _build(_config())
        _store(service, "database", "postgres, partitioned by day", turn=2)
        _young(service, ["and what about the deployment checklist"])

        front = _front(_collect(service))
        assert front is not None
        assert "postgres, partitioned by day" in front.content

    def test_it_is_announced_rather_than_disguised_as_a_turn(self):
        service = _build(_config())
        _store(service, "database", "postgres", turn=2)

        front = _front(_collect(service))
        assert front is not None
        assert front.content.startswith(FRONT_HEADER)
        assert MEMORY_BLOCK_HEADER in front.content

    def test_the_front_opens_the_conversation_as_a_user_turn(self):
        """The provider requires the first message to be a user turn; facts are the
        oldest information in the context, and a conversation reads chronologically."""
        service = _build(_config())
        _store(service, "database", "postgres", turn=2)
        _young(service, ["first surviving turn", "second surviving turn"])

        context = _collect(service)
        assert context[0].role == "user"
        assert [m.turn_index for m in context] == [-1, 50, 51]

    def test_with_nothing_stored_there_is_no_front(self):
        service = _build(_config())
        _young(service, ["just a turn"])
        assert _front(_collect(service)) is None


class TestTheFrontIsStableBetweenCuts:
    def test_the_front_does_not_change_with_the_question(self):
        """A front that changed with each question would change the prompt on every
        request: no cache hit, and every thinking block invalidated."""
        service = _build(_config())
        _store(service, "database", "postgres", turn=2)
        _store(service, "standup", "ten o'clock", turn=4)

        _young(service, ["which database did we pick?"], start=50)
        first = _front(_collect(service))
        _young(service, ["what time is the standup?"], start=51)
        second = _front(_collect(service))

        assert first is not None and second is not None
        assert first.content == second.content


class TestSupersededFactsNeverReachThePrompt:
    def test_only_the_surviving_value_is_in_the_front(self):
        """The single most important assertion here. Without it the model reads
        "postgres" and "mysql" as equal claims, and every supersession the system
        detected was recorded for nothing."""
        service = _build(_config())
        _store(service, "database", "the database is postgres", turn=2, kind=KnowledgeType.DECISION)
        _store(service, "database", "the database is mysql", turn=9, kind=KnowledgeType.DECISION)

        front = _front(_collect(service))
        assert front is not None
        assert "mysql" in front.content
        assert "postgres" not in front.content


class TestTheBudgetIsEnforcedOnTheRealPrompt:
    def test_the_facts_respect_max_memory_tokens(self):
        service = _build(_config(context_window=200, max_memory_ratio=0.25))
        for index in range(12):
            _store(service, f"topic{index}", f"a reasonably long fact number {index} about the system", turn=index)

        front = _front(_collect(service))
        assert front is not None
        facts_section = front.content.split(MEMORY_BLOCK_HEADER, 1)[1]
        assert count_tokens(MEMORY_BLOCK_HEADER + facts_section) <= service.gc_config.max_memory_tokens

    def test_by_default_only_the_token_budget_limits_the_facts(self):
        """Every fact that fits the budget is sent: no count leaves out facts there is room for."""
        service = _build(_config(context_window=2000))
        for index in range(12):
            _store(service, f"topic{index}", f"fact {index}", turn=index)

        front = _front(_collect(service))
        assert front is not None
        assert front.content.count("\n- (turn") == 12

    def test_front_tokens_count_toward_the_reported_context(self):
        """The front is real tokens in the real prompt. A context size that omitted
        it would under-report pressure in the reassuring direction."""
        service = _build(_config())
        _store(service, "database", "postgres", turn=2)
        _young(service, ["a surviving turn"])

        context = _collect(service)
        assert sum(m.token_count for m in context) == context[0].token_count + context[1].token_count


class TestFactsCanBeTurnedOff:
    def test_a_zero_memory_ratio_leaves_facts_out(self):
        """How a deployment opts out, and how a benchmark measures what facts are
        worth by running the same conversation both ways."""
        service = _build(_config(max_memory_ratio=0.0))
        _store(service, "database", "postgres", turn=2)
        assert _front(_collect(service)) is None

    def test_the_retired_count_setting_is_refused_rather_than_ignored(self):
        """Ignored, a caller passing it would believe facts were capped or off when they were not."""
        with pytest.raises(ValidationError, match="max_memory_facts"):
            GCConfig(max_memory_facts=0)


class TestTheRetrieverIsSwappable:
    def test_an_injected_retriever_chooses_the_facts(self):
        """Same dependency-injection seam as the compactor and the provider: a
        deployment with its own ranking supplies one, and nothing in the engine
        changes. It is asked without a query, because facts are chosen at the cut."""
        queries = []

        class _OnlyTheOldest:
            name = "oldest"

            def retrieve(self, entries, query, limit):
                queries.append(query)
                return sorted(entries, key=lambda entry: entry.message_turn)[:1]

        service = _build(_config(), retriever=_OnlyTheOldest())
        _store(service, "database", "postgres", turn=2)
        _store(service, "queue", "rabbitmq", turn=30)

        front = _front(_collect(service))
        assert front is not None
        assert "postgres" in front.content and "rabbitmq" not in front.content
        assert queries == [None]

    def test_the_default_retriever_is_lexical(self):
        assert isinstance(_build(_config()).fact_retriever, LexicalFactRetriever)


class TestUpdateAndCollectStillAgree:
    def test_both_paths_see_the_same_front(self):
        """`_assemble` is shared so the number the dashboard shows and the payload
        the proxy forwards cannot drift."""
        service = _build(_config())
        _store(service, "database", "postgres", turn=2)
        reported = []
        service.event_bus.subscribe(EventType.GC_FINISHED,
                                    lambda event: reported.append(event.data["gc_result"].final_messages))

        async def scenario():
            text = "a new turn mentioning the database again"
            await service.gc_update(SESSION, Message(role="user", content=text, token_count=count_tokens(text),
                                                     turn_index=60))
            return await service.gc_collect(SESSION)

        context = asyncio.run(scenario())
        assert [m.content for m in reported[-1]] == [m.content for m in context]


class TestTheFrontKeepsWhatMattersMost:
    def test_a_decision_outranks_newer_raw_excerpts(self):
        """Found by running the proxy demo: ranked by recency alone, the front kept
        two later acknowledgements ("Understood. I have noted that.") and dropped
        the turn-2 decision the whole demo exists to show surviving."""
        # A budget with room for about two entries, so ranking decides which survive.
        service = _build(_config(max_memory_ratio=0.22))
        _store(service, "database", "postgres, partitioned by day", turn=2, kind=KnowledgeType.DECISION)
        for turn in (10, 11, 12):
            _store(service, "__raw__", f"Understood, noted ({turn}).", turn=turn, kind=KnowledgeType.RAW)

        front = _front(_collect(service))
        assert front is not None
        assert "postgres, partitioned by day" in front.content
