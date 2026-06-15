from abc import ABC, abstractmethod

from llm_gc.engine.sweep.sweep_entry import SweepEntry
from llm_gc.models import Message
from llm_gc.scoring import RelevanceScorerResult

class BaseSweepStrategy(ABC):
    """Abstract base class for sweep classification strategies.

    Defines the interface that all sweep strategies must implement.
    The Sweeper delegates classification logic to whichever concrete
    strategy is injected, enabling pluggable algorithms.
    """
    def __init__(self, sweep_strategy: str):
        self.sweep_strategy = sweep_strategy

    @abstractmethod
    def sweep(self, conversation: list[Message], relevance_scores: list[RelevanceScorerResult]) -> list[SweepEntry]:
        """Classify each message and return the list of SweepEntries."""