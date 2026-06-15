import math

from llm_gc.models import Message
from llm_gc.scoring import ScorerResult, RelevanceType
from llm_gc.scoring.base_scorer import BaseScorer

class RecencyScorer(BaseScorer):
    """A scorer that evaluates message relevance based on recency using an exponential decay function.

    score = e^(-decay_rate * age)

    - Age is the message's distance from the end of the conversation (0 = newest).
    - Higher decay_rate means older messages lose relevance faster.
    - Default decay_rate of 0.1 gives 1 at age 0, ~0.37 at age 10 and ~0.007 at age 50.
    """

    def __init__(self, decay_rate: float = 0.1):
        super().__init__(scorer_name=RelevanceType.RECENCY)
        self.decay_rate = decay_rate

    def score(self, message: Message, conversation: list[Message]) -> ScorerResult:
        age = len(conversation) - conversation.index(message) - 1
        raw_score = math.exp(-self.decay_rate * age)
        return ScorerResult(
            score=raw_score,
            scorer_name=self.scorer_name,
            reason=f"Message is {age} turns old with decay rate {self.decay_rate}, score {raw_score:.2f}",
            signals={"age": age, "decay_rate": self.decay_rate, "raw_score": raw_score}
        )
