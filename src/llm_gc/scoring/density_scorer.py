import re

from llm_gc.config.constants import DENSITY_SCORER_PATTERN
from llm_gc.models import Message
from llm_gc.scoring import RelevanceType, BaseScorer, ScorerResult
from llm_gc.utils.token_counter import count_tokens

class DensityScorer(BaseScorer):
    """Score messages by information density: signal_count / token_count, normalized.

    Currently, detects information-carrying signals via regex: capitalized words, numbers,
    code fragments (backticks), and URLs (to be improved at later stage via NLP techniques).
    Higher density = more worth keeping.
    max_density controls the normalization ceiling — ratio at or above it scores 1.0.
    """

    def __init__(self, max_density: float = 0.5):
        super().__init__(scorer_name=RelevanceType.DENSITY)
        self.max_density = max_density

    def score(self, message: Message, conversation: list[Message]) -> ScorerResult:
        token_counts = count_tokens(message.content)
        signal_count = len(re.findall(DENSITY_SCORER_PATTERN, message.content))
        density_ratio = 0 if token_counts == 0 else  signal_count / token_counts
        density_score = min(density_ratio / self.max_density, 1.0)

        return ScorerResult(
            score=density_score,
            scorer_name=self.scorer_name,
            reason=f"Message has {signal_count} signals in {token_counts} tokens, density ratio {density_ratio:.4f}, score {density_score:.2f}",
            signals={"signal_count": signal_count, "token_counts": token_counts, "density_ratio": density_ratio,
                     "density_score": density_score}
        )
