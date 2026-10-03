import logging
import uuid
from time import perf_counter

from llm_gc.config.gc_config import GCConfig
from llm_gc.engine.context_composer import ContextComposer
from llm_gc.engine.gc_result import GCResult
from llm_gc.engine.gc_status import GCStatus
from llm_gc.engine.sweep import Sweeper, SweepClassification
from llm_gc.events import EventBus, Event, EventType
from llm_gc.models import Message
from llm_gc.scoring import RelevanceScorer

logger = logging.getLogger(__name__)
class GarbageCollector:
    """ Garbage collector is the component that owns the execution contract of the GC pipeline - it owns the order in which
    steps run, what happens when step fails, what the caller is allowed to see, and what gets emitted on the events bus around the
    whole run. It does not own any single step's logic, but it owns the shape of the run as a whole and centralizes the wiring logic.
    """
    def __init__(self,
                 event_bus: EventBus,
                 gc_config: GCConfig,
                 relevance_scorer: RelevanceScorer,
                 sweeper: Sweeper,
                 context_composer: ContextComposer
                 ) -> None:
        self.event_bus = event_bus
        self.gc_config = gc_config
        self.relevance_scorer = relevance_scorer
        self.sweeper = sweeper
        self.context_composer = context_composer

    async def collect(self, messages: list[Message]) -> GCResult:
        start = perf_counter()
        original_token_count = sum(m.token_count for m in messages)
        gc_run_id = str(uuid.uuid4())

        # Threshold guard
        threshold_tokens = int(self.gc_config.context_window * self.gc_config.gc_threshold)
        if original_token_count < threshold_tokens:
            # By pass GC
            gc_result = GCResult(final_messages=messages, gc_run_id=gc_run_id, status=GCStatus.BYPASSED_BELOW_THRESHOLD,
                            tokens_before=original_token_count, tokens_in_final=original_token_count,
                            kept_count=len(messages), compacted_count=0, archived_count=0,
                            sweep_result=None, duration_ms=(perf_counter() - start) * 1000)
            logger.info("GC bypassed successfully!", extra={"gc_run_id": gc_run_id, "reason": "below threshold"})
            self.event_bus.emit(Event(event_type=EventType.GC_FINISHED, data={"gc_run_id": gc_run_id, "gc_result": gc_result}))
            return gc_result

        try:
            current_stage = "score"
            scores = []
            for message in messages:
                scores.append(self.relevance_scorer.score(message, messages))
            current_stage = "sweep"
            sweep_result = self.sweeper.sweep(messages, scores)
            current_stage = "compose"
            final_messages = await self.context_composer.compose(sweep_result)
            gc_result = GCResult(final_messages=final_messages, gc_run_id=gc_run_id, status=GCStatus.COMPLETED,
                                 tokens_before=original_token_count, tokens_in_final=sum(m.token_count for m in final_messages),
                                 kept_count=sweep_result.classification_counts[SweepClassification.KEEP], compacted_count=sweep_result.classification_counts[SweepClassification.COMPACT],
                                 archived_count=sweep_result.classification_counts[SweepClassification.ARCHIVE],
                                 sweep_result=sweep_result, duration_ms=(perf_counter() - start) * 1000)
            logger.info("GC completed successfully!", extra={"gc_run_id": gc_run_id, "duration_ms": (perf_counter() - start) * 1000})
            self.event_bus.emit(Event(event_type=EventType.GC_FINISHED, data={"gc_run_id": gc_run_id, "gc_result": gc_result}))
            return gc_result
        except Exception as e:
            logger.exception("GC completion failed!", extra={"gc_run_id": gc_run_id, "stage": current_stage})
            gc_result = GCResult(final_messages=messages, gc_run_id=gc_run_id, status=GCStatus.BYPASSED_ON_ERROR,
                            failure_reason=str(e), failure_stage=current_stage,
                            tokens_before=original_token_count, tokens_in_final=sum(m.token_count for m in messages),
                            kept_count=len(messages), compacted_count=0, archived_count=0,
                            sweep_result=None, duration_ms=(perf_counter() - start) * 1000)
            self.event_bus.emit(Event(event_type=EventType.GC_FINISHED, data={"gc_run_id": gc_run_id,
                                                                              "stage": current_stage, "error": str(e), "exception_type": type(e).__name__,
                                                                              "gc_result": gc_result}))
            return gc_result





