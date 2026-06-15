import re

from llm_gc.config.constants import DECISION_SCORER_PATTERN
from llm_gc.models import Message
from llm_gc.scoring import BaseScorer, ScorerResult, RelevanceType

class DecisionScorer(BaseScorer):
    """Score messages by detecting decision-bearing messages. It scores messages by how many decision-indicator phrases they contain,
      normalized against a configurable threshold (max_decision): decision_signal_count / max_decision

      Currently, detects decision-carrying signals via regex: @DECISION_SCORER_PATTERN
      This is to be enhanced in later phases with ML based models and tuning.
      max_decision indicates a threshold count for number of decision phrases.
    """
    def __init__(self, max_decision: int = 3):
        super().__init__(scorer_name=RelevanceType.DECISION)
        self.max_decision = max_decision

    def score(self, message: Message, conversation: list[Message]) -> ScorerResult:
        decision_signal_count = len(re.findall(DECISION_SCORER_PATTERN, message.content, flags=re.IGNORECASE))
        decision_score = min(decision_signal_count / self.max_decision, 1.0)
        return ScorerResult(score=decision_score, scorer_name=self.scorer_name,
                            reason=f"Message has {decision_signal_count} decision signals, score {decision_score:.2f}",
                            signals={"decision_signal_count": decision_signal_count, "decision_score": decision_score})


