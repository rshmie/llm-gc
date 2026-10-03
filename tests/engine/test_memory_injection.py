"""Archived turns must reach the model, not just the dashboard.

Until this landed, `gc_collect` assembled old gen + young gen and stopped, so an
archived turn was a deletion with a receipt: the fact stored, visible on the
dashboard, and unusable by the model. That is the same information loss as the
sliding window this project exists to beat — and worse, because the transparency
layer reported the fact as retained.
"""

import asyncio

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import NoOpCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory, LexicalFactRetriever
from llm_gc.engine.generations.memory_block import MEMORY_BLOCK_HEADER
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeType
from llm_gc.scoring import RecencyScorer, RelevanceScorer, RelevanceType
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens

SESSION = "s"


def _config(**overrides) -> GCConfig:
    base = {
        "context_window": 200,
        "gc_threshold": 0.5,
        "keep_threshold": 0.55,
        "archive_threshold": 0.3,
        "min_compactable_tokens": 10,
        "last_n_turns_to_keep": 2,
        "max_memory_tokens": 120,
        "max_memory_facts": 6,
    }
    base.update(overrides)
    return GCConfig(**base)


def _build(config: GCConfig, retriever=None):
    bus = EventBus()
    memory = GenerationalMemory(
        event_bus=bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        permanent_generation=PermanentGeneration(event_bus=bus),
        compactor=NoOpCompactor(event_bus=bus),
    )
    service = GcService(
        session_manager=SessionManager(),
        generational_memory=memory,
        relevance_scorer=RelevanceScorer(
            scorers=[RecencyScorer(decay_rate=0.3)], weights={RelevanceType.RECENCY: 1.0}, event_bus=bus
        ),
        sweeper=Sweeper(
            sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus
        ),
        event_bus=bus,
        gc_config=config,
        fact_retriever=retriever,
    )
    return service, memory


def _store(memory: GenerationalMemory, topic: str, content: str, turn: int) -> None:
    """Put a fact straight into permanent generation.

    Driving a whole conversation through the sweeper to produce one archived turn
    makes a test about *injection* depend on every threshold in the config — which
    is how a previous draft of these tests silently measured nothing. Facts are
    placed directly; the aging path has its own tests.
    """
    memory.permanent_generation.add_knowledge_entry(
        KnowledgeEntry(message_turn=turn, content=content, topic_label=topic, knowledge_type=KnowledgeType.FACT)
    )


def _young(service: GcService, texts: list[str], start: int = 50) -> None:
    async def scenario():
        async with service.session_manager.acquire(SESSION) as state:
            for offset, text in enumerate(texts):
                state.messages.append(
                    Message(
                        role="user",
                        content=text,
                        token_count=count_tokens(text),
                        turn_index=start + offset,
                    )
                )

    asyncio.run(scenario())


def _collect(service: GcService, query: str | None = None) -> list[Message]:
    return asyncio.run(service.gc_collect(SESSION, query=query))


def _block(context: list[Message]) -> Message | None:
    return context[0] if context and context[0].turn_index == -1 else None


class TestAnArchivedFactReachesThePrompt:
    def test_the_relevant_fact_is_injected(self):
        service, memory = _build(_config())
        _store(memory, "database", "postgres, partitioned by day", turn=2)
        _young(service, ["and what about the deployment checklist"])

        context = _collect(service, query="Which database are we actually targeting?")
        block = _block(context)

        assert block is not None
        assert "database: postgres, partitioned by day" in block.content

    def test_it_is_announced_rather_than_disguised_as_a_turn(self):
        service, memory = _build(_config())
        _store(memory, "database", "postgres", turn=2)

        block = _block(_collect(service, query="which database?"))

        assert block is not None
        assert block.content.startswith(MEMORY_BLOCK_HEADER)

    def test_it_comes_before_every_surviving_turn(self):
        """These facts are the oldest information in the context, and a conversation
        should read chronologically."""
        service, memory = _build(_config())
        _store(memory, "database", "postgres", turn=2)
        _young(service, ["a later turn", "the latest turn"])

        context = _collect(service, query="which database?")

        assert context[0].turn_index == -1
        assert [message.turn_index for message in context[1:]] == [50, 51]

    def test_an_irrelevant_fact_is_not_injected(self):
        """Memory is retrieved, not dumped. A prompt full of unrelated "memory"
        spends tokens and invites the model to use something off-topic."""
        service, memory = _build(_config())
        _store(memory, "database", "postgres", turn=2)

        assert _block(_collect(service, query="what time is the standup?")) is None

    def test_with_nothing_stored_nothing_is_injected(self):
        service, _ = _build(_config())
        _young(service, ["just one turn"])

        context = _collect(service)
        assert _block(context) is None
        assert len(context) == 1


class TestSupersededFactsNeverReachThePrompt:
    def test_only_the_surviving_value_is_injected(self):
        """The single most important assertion here. Without it the model reads
        "the database is postgres" and "the database is mysql" as equal claims, and
        every supersession the system detected was recorded for nothing."""
        service, memory = _build(_config())
        _store(memory, "database", "postgres", turn=2)
        _store(memory, "database", "mysql", turn=40)

        block = _block(_collect(service, query="which database are we targeting?"))

        assert block is not None
        assert "mysql" in block.content
        assert "postgres" not in block.content


class TestTheBudgetIsEnforcedOnTheRealPrompt:
    def test_the_block_respects_max_memory_tokens(self):
        service, memory = _build(_config(max_memory_tokens=60, max_memory_facts=20))
        for index in range(20):
            _store(memory, f"topic{index}", f"some value number {index} with trailing prose", turn=index)

        block = _block(_collect(service, query="topic0 topic1 topic2 topic3 topic4 value"))

        assert block is not None
        assert block.token_count <= 60

    def test_max_memory_facts_caps_retrieval(self):
        service, memory = _build(_config(max_memory_facts=2, max_memory_tokens=10_000))
        for index in range(10):
            _store(memory, f"topic{index}", f"value {index}", turn=index)

        block = _block(_collect(service, query=" ".join(f"topic{i}" for i in range(10))))

        assert block is not None
        assert block.content.count("\n") == 2, "header plus exactly two facts"

    def test_injected_tokens_count_toward_the_reported_context(self):
        """The block is real tokens in the real prompt. A context size that omitted
        it would under-report pressure in the reassuring direction - the same error
        the token meter already refuses to make about old-gen summaries."""
        service, memory = _build(_config())
        _store(memory, "database", "postgres, partitioned by day", turn=2)
        _young(service, ["a turn about the database"])

        with_memory = _collect(service, query="which database?")
        without_memory = _collect(service, query="what time is the standup?")

        assert sum(m.token_count for m in with_memory) > sum(m.token_count for m in without_memory)


class TestInjectionCanBeTurnedOff:
    def test_zero_memory_tokens_disables_it(self):
        """How a deployment opts out, and how a benchmark measures what injection is
        worth by running the same conversation both ways."""
        service, memory = _build(_config(max_memory_tokens=0))
        _store(memory, "database", "postgres", turn=2)

        assert _block(_collect(service, query="which database?")) is None

    def test_zero_memory_facts_disables_it(self):
        service, memory = _build(_config(max_memory_facts=0))
        _store(memory, "database", "postgres", turn=2)

        assert _block(_collect(service, query="which database?")) is None


class TestWithoutAQuery:
    def test_recent_facts_are_still_injected(self):
        """A caller with no query has not asked for no memory. The dashboard reads
        the context without one."""
        service, memory = _build(_config())
        _store(memory, "database", "postgres", turn=2)
        _store(memory, "queue", "rabbitmq", turn=30)

        block = _block(_collect(service, query=None))

        assert block is not None
        assert "rabbitmq" in block.content


class TestTheRetrieverIsSwappable:
    def test_an_injected_retriever_is_used(self):
        """Same dependency-injection seam as the compactor and the provider: a
        deployment with its own vector store supplies one, and nothing in the
        engine changes."""

        class _OnlyTheOldest:
            name = "oldest"

            def retrieve(self, entries, query, limit):
                return sorted(entries, key=lambda entry: entry.message_turn)[:limit]

        service, memory = _build(_config(), retriever=_OnlyTheOldest())
        _store(memory, "database", "postgres", turn=2)
        _store(memory, "queue", "rabbitmq", turn=30)

        block = _block(_collect(service, query="what time is the standup?"))

        # The lexical retriever would have returned nothing for this query; this one
        # ignores the query entirely, which is how we know it ran.
        assert block is not None
        assert "postgres" in block.content

    def test_the_default_retriever_is_lexical(self):
        service, _ = _build(_config())
        assert isinstance(service.fact_retriever, LexicalFactRetriever)


class TestUpdateAndCollectStillAgree:
    def test_both_paths_see_the_same_memory(self):
        """`_assemble` is shared so the number the dashboard shows and the payload
        the proxy forwards cannot drift. Injection must not break that."""
        service, memory = _build(_config())
        _store(memory, "database", "postgres", turn=2)

        async def scenario():
            text = "a new turn mentioning the database again"
            await service.gc_update(
                SESSION, Message(role="user", content=text, token_count=count_tokens(text), turn_index=60)
            )
            return await service.gc_collect(SESSION, query=None)

        context = asyncio.run(scenario())
        assert _block(context) is not None
