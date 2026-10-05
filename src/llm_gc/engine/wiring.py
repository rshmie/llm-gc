"""The default engine: which scorers, weights, sweeper and compactor a `GcService` runs with.

One definition, so the proxy and anything that measures it run the same engine.
Why these components and weights: doc/design/scoring/overview.md.
"""

from llm_gc.config import GCConfig
from llm_gc.engine.compaction import BaseCompactor, LLMCompactor, NoOpCompactor
from llm_gc.engine.gc_service import GcService
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.llm import LLMProvider
from llm_gc.scoring import (
    DecisionScorer,
    DensityScorer,
    RecencyScorer,
    ReferenceScorer,
    RelevanceScorer,
    RelevanceType,
)
from llm_gc.session import SessionManager

# Per-turn decay of the recency score: a turn ten turns old scores about a fifth of the newest.
DEFAULT_RECENCY_DECAY_RATE = 0.15

# Recency dominates; the content scorers lift what recency alone would let go.
# They sum to 1 so the combined score stays on the thresholds' 0-1 scale.
DEFAULT_SCORER_WEIGHTS: dict[RelevanceType, float] = {
    RelevanceType.RECENCY: 0.6,
    RelevanceType.DENSITY: 0.2,
    RelevanceType.DECISION: 0.1,
    RelevanceType.REFERENCE: 0.1,
}


def build_gc_service(config: GCConfig, *, summary_provider: LLMProvider | None,
                     session_manager: SessionManager | None = None) -> GcService:
    """A `GcService` wired with the default components.

    Args:
        config: Thresholds and budgets for the conversations it will manage.
        summary_provider: The model LLM-GC calls to summarise turns at a cut. None
            wires `NoOpCompactor`, which keeps turns whole: nothing is summarised,
            so cuts save only what archiving saves.
        session_manager: Where sessions live; a fresh one when omitted.
    """
    bus = EventBus()
    compactor: BaseCompactor = (NoOpCompactor(event_bus=bus) if summary_provider is None
                                else LLMCompactor(event_bus=bus, provider=summary_provider))
    relevance_scorer = RelevanceScorer(
        scorers=[RecencyScorer(decay_rate=DEFAULT_RECENCY_DECAY_RATE), DensityScorer(), DecisionScorer(),
                 ReferenceScorer()],
        weights=dict(DEFAULT_SCORER_WEIGHTS),
        event_bus=bus,
    )
    return GcService(
        session_manager=session_manager if session_manager is not None else SessionManager(),
        generational_memory=GenerationalMemory(knowledge_extractor=KnowledgeExtractor(event_bus=bus),
                                               compactor=compactor),
        relevance_scorer=relevance_scorer,
        sweeper=Sweeper(sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus),
        event_bus=bus,
        gc_config=config,
    )
