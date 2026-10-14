"""What the model knows about one conversation, span by span, turn by turn and fact by fact.

The engine's records answer "what did the collector do". This projection answers
the question a developer actually has: where is turn 17 now, how much of it does
the model still receive, and is the fact read from it in front of the model at all.
It derives everything from one `SessionInspection`, so every part describes the
same moment. See doc/design/visibility/overview.md.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from llm_gc.engine.inspection import SessionInspection
from llm_gc.engine.switch_policy import CacheReading, CacheState
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import DetailLevel, Span
from llm_gc.models.knowledge_entry import KnowledgeStatus
from llm_gc.models.message import tool_result_ids, tool_use_ids
from llm_gc.utils import count_tokens


class TurnFidelity(Enum):
    """How much of a turn the model receives: word for word in the tail, or through its span at its level.

    Named after the detail levels, so the dashboard, the code and the front's
    markers use the same words.
    """

    VERBATIM = "verbatim"          # in the tail, exactly as the client sent it
    KEPT = "kept"                  # its span's turns, word for word
    COMPACTED = "compacted"        # its span's summary, under its opening line and facts
    ARCHIVED = "archived"          # its span's opening line and facts
    OUT_OF_VIEW = "out_of_view"    # one shared line saying its span exists, and nothing of what was said


def _fidelity(level: DetailLevel) -> TurnFidelity:
    return TurnFidelity(level.value)


class TurnState(BaseModel):
    """One recorded turn: its words, how the model receives it now, and what the next collection would do."""

    turn_index: int
    role: str
    tokens: int
    text: str                              # as recorded: a span keeps its turns' originals
    fidelity: TurnFidelity
    span: int | None = None                # index into `spans`, once the turn has left the tail
    score: float | None = None             # its span's score now; a tail turn has one only if the next collection takes it
    becomes: TurnFidelity | None = None    # set when the next collection would change how the model receives it
    tool_calls: bool = False               # an assistant turn asking the client to run a tool
    tool_results: bool = False             # a user turn carrying what the client's tools returned, not words the user wrote
    relayed: bool = False                  # a user turn relaying another agent's report, not words the user wrote


class SpanState(BaseModel):
    """One span of the old generation, and exactly what the front carries for it."""

    turns: list[int]
    level: DetailLevel
    text: str                     # what the model reads for it, marker included; a folded span's is its run's shared line
    tokens: int | None            # what the front spends on its lines; None while folded, as its run shares one line
    summary: str | None           # its summary without the marker, when it has one
    score: float                  # how much it matters to what is discussed now
    becomes: DetailLevel | None   # set when the next collection would show it at another level
    facts_unread: bool = False    # its facts could not be read when it was made, so none are on record


class FactState(BaseModel):
    turn: int
    speaker: str | None                # "user" or "assistant": who stated it; None when the reader did not say
    topic: str
    content: str
    status: KnowledgeStatus
    sent: bool                         # on its span's line in the front, so the model reads it as a fact
    span_level: DetailLevel | None     # how the span it was read from is shown; None if no span holds its turn
    superseded_by_turn: int | None = None
    superseded_by: str | None = None   # what the fact that replaced it says
    superseded_by_speaker: str | None = None   # and who stated that


class NextCutState(BaseModel):
    turns: list[int]                      # the verbatim turns the cut would take
    spans: list[list[int]]                # those turns, as the spans the cut would make of them
    compactable_tokens: int
    min_cut_tokens: int
    batch_ready: bool                     # necessary, not sufficient - see `CutPreview`
    estimated_saving: int                 # what the cut would save at estimated summary sizes


class StagedCutState(BaseModel):
    """A prepared cut waiting to be switched in, and what decides when: the switch policy's own measures."""

    turns: list[int]                      # the verbatim turns it would take; the next pass may prepare more
    saving_tokens: int
    rewritten_tokens: int
    stale_tokens: int
    stale_bound_tokens: int
    min_saving_share: float
    switches: bool                        # whether the next request would switch it in
    reason: str                           # the policy's reason, as `SwitchReason` names it


class CacheReadingState(BaseModel):
    """The provider's cache for this conversation as the proxy reads it, and from what."""

    state: CacheState
    uses_cache: bool | None               # None before any request was answered
    idle_s: float | None
    lifetime_s: float | None


class KnowledgeState(BaseModel):
    """Everything the dashboard shows about one conversation's content."""

    session_id: str
    version: int
    recorded_turns: int
    cuts: int
    context_tokens: int
    front_text: str | None                # the front exactly as the model receives it
    front_tokens: int
    spans: list[SpanState]                # oldest first
    turns: list[TurnState]                # oldest first
    facts: list[FactState]
    next_cut: NextCutState
    aging_in_flight: bool
    last_activity: datetime
    context_window: int                   # the window of the model being called, as the recap is sized for it
    recap_budget: int
    collected: bool                       # False: recorded and shown, never collected
    staged: StagedCutState | None = None  # set while a prepared cut waits to be switched in
    cache: CacheReadingState | None = None   # None when no proxy reads the cache beside this dashboard


def project_knowledge_state(inspection: SessionInspection, cache: CacheReading | None = None) -> KnowledgeState:
    """Pivot one session's inspection from generations to spans, turns and facts.

    `cache` is the reading the inspection's verdict on a staged cut was made for, shown beside it.
    """
    snapshot = inspection.snapshot
    spans = _span_states(inspection)
    staged = inspection.staged
    return KnowledgeState(
        session_id=inspection.session_id,
        version=snapshot.version,
        recorded_turns=snapshot.recorded_turn_count,
        cuts=snapshot.cut_count,
        context_tokens=inspection.context_tokens,
        front_text=inspection.front.message.content if inspection.front is not None else None,
        front_tokens=inspection.front.message.token_count if inspection.front is not None else 0,
        spans=[state for _, state in spans],
        turns=_turn_states(inspection, spans),
        facts=_fact_states(snapshot.knowledge, inspection, spans),
        next_cut=NextCutState(turns=list(inspection.next_cut.turns),
                              spans=[list(turns) for turns in inspection.next_cut.spans],
                              compactable_tokens=inspection.next_cut.compactable_tokens,
                              min_cut_tokens=inspection.next_cut.min_cut_tokens,
                              batch_ready=inspection.next_cut.batch_ready,
                              estimated_saving=inspection.next_cut.estimated_saving),
        aging_in_flight=inspection.aging_in_flight,
        last_activity=inspection.last_activity,
        context_window=inspection.context_window,
        recap_budget=inspection.recap_budget,
        collected=inspection.collection_enabled,
        staged=StagedCutState(turns=sorted(snapshot.staged.delta.departed_turns) if snapshot.staged else [],
                              saving_tokens=staged.saving_tokens, rewritten_tokens=staged.rewritten_tokens,
                              stale_tokens=staged.stale_tokens, stale_bound_tokens=staged.stale_bound_tokens,
                              min_saving_share=staged.min_saving_share, switches=staged.decision.switch,
                              reason=staged.decision.reason.value) if staged is not None else None,
        cache=CacheReadingState(state=cache.state, uses_cache=cache.uses_cache, idle_s=cache.idle_s,
                                lifetime_s=cache.lifetime_s) if cache is not None else None,
    )


def _span_states(inspection: SessionInspection) -> list[tuple[Span, SpanState]]:
    """Every span of the old generation with its state, oldest first.

    The text is read from the rendered front rather than rendered again, so it is
    byte for byte what the model receives.
    """
    front = inspection.front
    lines: dict[int, str] = {}
    if front is not None:
        lines = {id(span): text for span, text in front.entries}
        lines |= {id(span): line for run, line in front.folded for span in run}
    preview = inspection.next_cut
    states = []
    for span, level, score in zip(inspection.snapshot.old, preview.span_levels, preview.span_scores, strict=True):
        folded = span.level is DetailLevel.OUT_OF_VIEW
        states.append((span, SpanState(
            turns=list(span.turn_indices), level=span.level, text=lines[id(span)],
            tokens=None if folded else count_tokens(lines[id(span)]),
            summary=_summary(span),
            score=score, becomes=level if level is not span.level else None, facts_unread=span.facts_unread)))
    return sorted(states, key=lambda pair: pair[0].first_turn)


def _summary(span: Span) -> str | None:
    if span.summary is None:
        return None
    return span.summary_text if span.summary_text is not None else span.summary.content


def _turn_states(inspection: SessionInspection, spans: list[tuple[Span, SpanState]]) -> list[TurnState]:
    """One state per turn the session holds, oldest first: each is in the tail or in exactly one span."""
    preview = inspection.next_cut
    states = []
    for message in inspection.tail:
        level = preview.levels.get(message.turn_index)
        states.append(_turn_state(message, TurnFidelity.VERBATIM, score=preview.scores.get(message.turn_index),
                                  becomes=_fidelity(level) if level is not None else None))
    for index, (span, state) in enumerate(spans):
        becomes = _fidelity(state.becomes) if state.becomes is not None else None
        states += [_turn_state(turn, _fidelity(span.level), span=index, score=state.score, becomes=becomes)
                   for turn in span.turns]
    return sorted(states, key=lambda state: state.turn_index)


def _turn_state(message: Message, fidelity: TurnFidelity, *, span: int | None = None, score: float | None = None,
                becomes: TurnFidelity | None = None) -> TurnState:
    return TurnState(turn_index=message.turn_index, role=message.role, tokens=message.token_count,
                     text=message.content, fidelity=fidelity, span=span, score=score, becomes=becomes,
                     tool_calls=bool(tool_use_ids(message)), tool_results=bool(tool_result_ids(message)),
                     relayed=message.relayed)


def _fact_states(knowledge: tuple[KnowledgeEntry, ...], inspection: SessionInspection,
                 spans: list[tuple[Span, SpanState]]) -> list[FactState]:
    """Every fact ever filed, superseded ones included, ordered by the turn it came from.

    A superseded fact names the fact that replaced it, and its turn: the next fact
    filed on the same topic that could supersede, which is the rule supersession
    follows.
    """
    sent = set(inspection.front.facts) if inspection.front is not None else set()
    level_of = {turn: span.level for span, _ in spans for turn in span.turn_indices}
    states = []
    for position, fact in enumerate(knowledge):
        replacement = None
        if fact.knowledge_status is KnowledgeStatus.SUPERSEDED:
            replacement = next((later for later in knowledge[position + 1:]
                                if later.topic_label == fact.topic_label and later.can_supersede), None)
        states.append(FactState(turn=fact.message_turn, speaker=fact.speaker, topic=fact.topic_label, content=fact.content,
                                status=fact.knowledge_status, sent=fact in sent,
                                span_level=level_of.get(fact.message_turn),
                                superseded_by_turn=replacement.message_turn if replacement is not None else None,
                                superseded_by=replacement.content if replacement is not None else None,
                                superseded_by_speaker=replacement.speaker if replacement is not None else None))
    return sorted(states, key=lambda state: state.turn)
