from llm_gc.events import EventType, Event
from llm_gc.events.event_bus import EventBus
from llm_gc.models import Message
from llm_gc.scoring.base_scorer import BaseScorer
from llm_gc.scoring.relevance_type import RelevanceType
from llm_gc.scoring.relevance_scorer_result import RelevanceScorerResult

class RelevanceScorer:
    """Orchestrates all individual scorers to produce a single combined relevance score per message.

    Accepts scorers and weights via DI, applies weighted combination, and normalizes the result.
    The Sweeper calls this once per message to get a keep/compress/remove signal.
    """

    def __init__(self, scorers: list[BaseScorer], weights: dict[RelevanceType, float], event_bus: EventBus | None = None):
        self.scorers = scorers
        self.weights = weights
        self.event_bus = event_bus

    def score(self, message: Message, conversation: list[Message]) -> RelevanceScorerResult:
        weighted_score = 0
        scorer_result_list = []
        for scorer in self.scorers:
            scorer_result = scorer.score(message, conversation)
            weight = self.weights.get(scorer.scorer_name, 0.0)
            weighted_score += scorer_result.score * weight
            scorer_result_list.append(scorer_result)

        weighted_score = weighted_score / sum(self.weights.values())
        if self.event_bus is not None:
            self.event_bus.emit(Event(event_type=EventType.MESSAGE_RELEVANCE_SCORED,
                                      data={"message": message, "scorer_results": scorer_result_list, "combined_score": weighted_score}))
        return RelevanceScorerResult(message=message, combined_score=weighted_score, scorer_results=scorer_result_list)
