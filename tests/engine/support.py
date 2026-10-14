"""Fakes shared by tests that drive a whole `GcService`.

A turn here is sized like a real one: a short first sentence, which becomes its
span's index line, then a long body. That makes a quote cost more than a
summary, and a summary more than an archived line - the order every detail
level depends on. With a one-word turn, keeping it would be cheapest and no span
could ever be compacted or archived.
"""

from collections.abc import Mapping

from llm_gc.config import GCConfig
from llm_gc.engine import GCResult, GcService
from llm_gc.engine.compaction import BaseCompactor, LLMCompactor
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.switch_policy import CacheState
from llm_gc.events import EventBus
from llm_gc.extraction import FactReader, PatternFactReader
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.models import Message
from llm_gc.scoring import SpanScore
from llm_gc.session import SessionManager
from llm_gc.utils import count_tokens

KEPT_SCORE, COMPACTED_SCORE, ARCHIVED_SCORE = 0.9, 0.5, 0.1

SUMMARY = LLMResponse(text="The turn weighed storage options, replication and failover in brief.",
                      stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=14, model="scripted",
                      provider="offline")

_SENTENCE = ("We compared the storage layer options at length, weighing replication, failover, retention "
             "and the cost of each replica under the expected load. ")


def turn(index: int, role: str = "user", opening: str | None = None, length: int = 4) -> Message:
    """A turn with its own one-sentence opening and a body `length` sentences long."""
    text = f"{opening or f'Turn {index} is about storage.'} {_SENTENCE * length}"
    return Message(role=role, content=text, token_count=count_tokens(text), turn_index=index)


class ScriptedSpanScorer:
    """Scores each span by its first turn from a map the test controls; unlisted spans score as kept.

    The test can change the map between collections to move a span up or down.
    """

    def __init__(self, scores: Mapping[int, float] | None = None, default: float = KEPT_SCORE) -> None:
        self.scores = dict(scores or {})
        self.default = default

    def score(self, spans, now, newest_turn):
        values = [self.scores.get(span[0].turn_index, self.default) for span in spans]
        return [SpanScore(combined=value, liveness=value, parts={}) for value in values]


def make_service(config: GCConfig | None = None, scores: Mapping[int, float] | None = None, *,
                 compactor: BaseCompactor | None = None, fact_reader: FactReader | None = None,
                 default_score: float = KEPT_SCORE) -> tuple[GcService, ScriptedSpanScorer, FakeProvider]:
    """A service on a scripted scorer and a scripted summariser; returns both for the test to steer."""
    bus = EventBus()
    provider = FakeProvider(responses=[SUMMARY] * 500)
    scorer = ScriptedSpanScorer(scores, default_score)
    service = GcService(
        session_manager=SessionManager(),
        generational_memory=GenerationalMemory(
            fact_reader=fact_reader if fact_reader is not None else PatternFactReader(),
            compactor=compactor if compactor is not None else LLMCompactor(event_bus=bus, provider=provider)),
        span_scorer=scorer,  # type: ignore[arg-type]  # duck-typed: only `score` is called
        event_bus=bus,
        gc_config=config if config is not None else GCConfig(min_cut_tokens=0, last_n_turns_to_keep=1),
    )
    return service, scorer, provider


async def collect(service: GcService, session_id: str, *, ignore_batch_minimum: bool = False) -> GCResult:
    """Prepare a cut and switch it in at once, as on a cold cache: a whole collection in one call.

    Returns the collection when one was switched in, with what its preparation spent; otherwise
    what the pass reported. The switch carries what preparing fell short of itself.
    """
    prepared = await service.gc_age(session_id, ignore_batch_minimum=ignore_batch_minimum)
    switched = service.gc_switch(session_id, CacheState.COLD)
    if switched is None:
        return prepared
    return switched.model_copy(update={"summary_calls": prepared.summary_calls,
                                       "summary_input_tokens": prepared.summary_input_tokens,
                                       "summary_output_tokens": prepared.summary_output_tokens})
