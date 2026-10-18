"""The default engine: which scorers, extractor and compactor a `GcService` runs with.

One definition, so the proxy and anything that measures it run the same engine.
Why these components and weights: doc/design/scoring/overview.md.
"""

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import BaseCompactor, LLMCompactor, NoOpCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.events import EventBus
from llm_gc.extraction import FactReader, LLMFactReader, PatternFactReader
from llm_gc.llm import LLMProvider, MeteredProvider
from llm_gc.scoring import DecisionScorer, DensityScorer, SpanScorer
from llm_gc.session import SessionManager


def build_gc_service(
    config: GCConfig,
    *,
    summary_provider: LLMProvider | None,
    session_manager: SessionManager | None = None,
    fact_reader: FactReader | None = None,
) -> GcService:
    """A `GcService` wired with the default components.

    Args:
        config: Thresholds and budgets for the conversations it will manage.
        summary_provider: The model LLM-GC calls to summarise spans and read their
            facts. None wires `NoOpCompactor`, whose "summary" is the turns
            themselves and never shorter, so spans are kept or archived but never
            compacted, and facts are read by patterns from the user's own turns.
        session_manager: Where sessions live. When omitted, a fresh one that announces
            each close on the service's bus. One passed in announces on the bus it was
            given, if any.
        fact_reader: Overrides how facts are read, for a run that must make no
            model call for them, such as an offline rehearsal.
    """
    bus = EventBus()
    # Metered, so each collection can report what its own model calls spent.
    summary_provider = MeteredProvider(summary_provider) if summary_provider is not None else None
    compactor: BaseCompactor = (
        NoOpCompactor(event_bus=bus)
        if summary_provider is None
        else LLMCompactor(event_bus=bus, provider=summary_provider)
    )
    return GcService(
        session_manager=session_manager if session_manager is not None else SessionManager(event_bus=bus),
        generational_memory=GenerationalMemory(
            fact_reader=fact_reader
            if fact_reader is not None
            else PatternFactReader()
            if summary_provider is None
            else LLMFactReader(summary_provider),
            compactor=compactor,
        ),
        span_scorer=SpanScorer([DecisionScorer(), DensityScorer()]),
        event_bus=bus,
        gc_config=config,
    )
