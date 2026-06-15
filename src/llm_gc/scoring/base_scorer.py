from abc import ABC, abstractmethod

from llm_gc.models.message import Message
from llm_gc.scoring.relevance_type import RelevanceType
from llm_gc.scoring.scorer_result import ScorerResult

class BaseScorer(ABC):
    def __init__(self, scorer_name: RelevanceType):
        self.scorer_name = scorer_name

    @abstractmethod
    def score(self, message: Message, conversation : list[Message]) -> ScorerResult:
        """Score a single message's relevance within the conversation context."""



