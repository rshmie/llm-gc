import re

from llm_gc.config.constants import REFERENCE_SCORER_PATTERN
from llm_gc.models import Message
from llm_gc.scoring import BaseScorer, ScorerResult, RelevanceType

class ReferenceScorer(BaseScorer):
    """ Scores a message by detecting if it gets referenced by later messages in the conversation.
    The rationale is that if a message is referenced by later messages, it likely contains important information worth keeping.
     - It counts how many later messages reference the message (via regex pattern matching).
     - The score is then calculated as reference_count / max_references, capped at 1.0.
     - max_references is a tunable parameter that controls the normalization ceiling — how many references are needed to score 1.0.
     - This is a heuristic approach and can be further improved with more sophisticated NLP techniques for reference detection in future iterations.
    """
    def __init__(self, max_references: int = 3):
        super().__init__(scorer_name=RelevanceType.REFERENCE)
        self.max_references = max_references

    def score(self, message: Message, conversation: list[Message]) -> ScorerResult:
        later_messages = conversation[conversation.index(message) + 1:]
        reference_count = 0
        for later_message in later_messages:
            if re.search(REFERENCE_SCORER_PATTERN, later_message.content, flags=re.IGNORECASE):
                reference_count += 1

        reference_score = min(reference_count / self.max_references, 1)
        return ScorerResult(score=reference_score, scorer_name=self.scorer_name,
                            reason=f"{reference_count} message is being referenced by later messages, score {reference_score:.2f}",
                            signals={"reference_count": reference_count, "reference_score": reference_score})