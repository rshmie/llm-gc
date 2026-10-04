from llm_gc.config import GCConfig
from llm_gc.engine.sweep import BaseSweepStrategy, SweepClassification, SweepEntry
from llm_gc.models import Message
from llm_gc.scoring import RelevanceScorerResult

TOO_SHORT_TO_SUMMARISE = "Forced KEEP: too short to be worth summarising (min_compactable_tokens)"


class ThresholdSweepStrategy(BaseSweepStrategy):
    """Threshold-based sweep strategy that classifies messages using score and token count.

    Classification logic:
      - Score >= keep_threshold OR token_count <= min_compactable_tokens -> KEEP
      - archive_threshold <= score < keep_threshold -> COMPACT
      - Score < archive_threshold -> ARCHIVE

    A message kept only because it is too short to be worth summarising - its score
    alone would have compacted or archived it - is recorded as an override, so a
    reader can tell a size rule from a high score. The universal rules (last N
    turns, system messages) are applied afterwards by the Sweeper.
    """
    def __init__(self, gc_config: GCConfig):
        super().__init__(sweep_strategy="threshold")
        self.gc_config = gc_config

    def sweep(self, conversation: list[Message], relevance_scores: list[RelevanceScorerResult]) -> list[SweepEntry]:
        """Mark each of the message with appropriate sweep classification type and return the list of Sweep entries."""
        sweep_entries = []
        for message, score_result in zip(conversation, relevance_scores):
            combined_score: float = score_result.combined_score
            too_short = (combined_score < self.gc_config.keep_threshold
                         and message.token_count <= self.gc_config.min_compactable_tokens)
            if combined_score >= self.gc_config.keep_threshold or too_short:
                sweep_classification = SweepClassification.KEEP
            elif self.gc_config.archive_threshold <= combined_score < self.gc_config.keep_threshold:
                sweep_classification = SweepClassification.COMPACT
            else:
                sweep_classification = SweepClassification.ARCHIVE
            sweep_entries.append(SweepEntry(message=message, classification=sweep_classification,
                                        relevance_score=score_result.combined_score, turn_index=message.turn_index,
                                        override_applied=too_short,
                                        override_reason=TOO_SHORT_TO_SUMMARISE if too_short else None))
        return sweep_entries
