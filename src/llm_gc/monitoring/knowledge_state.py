"""What the model knows about one conversation, turn by turn and fact by fact.

The engine's records answer "what did the collector do". This projection answers
the question a developer actually has: where is turn 17 now, how faithfully does
the model still see it, and is the fact it produced in front of the model at all.
It derives everything from one `SessionInspection`, so every part describes the
same moment. See doc/design/visibility/overview.md.
"""

from collections import defaultdict
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from llm_gc.engine.generations.front import stated_text
from llm_gc.engine.inspection import SessionInspection
from llm_gc.models import KnowledgeEntry, SweepClassification
from llm_gc.models.generation_snapshot import DetailLevel, Span
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType
from llm_gc.utils import count_tokens


class TurnFidelity(Enum):
    """How much of a turn the model still receives, from most to least."""

    VERBATIM = "verbatim"        # in the tail, exactly as the client sent it
    QUOTED = "quoted"            # in the front, word for word
    SUMMARISED = "summarised"    # in the front, condensed together with its neighbours
    FACTS_ONLY = "facts_only"    # only the facts extracted from it remain
    SUPERSEDED = "superseded"    # only facts remain, and every one has since been contradicted


# How a turn reads once its span is shown at each level, and the dashboard's word for that level.
_FIDELITY_AT_LEVEL = {
    DetailLevel.KEPT: TurnFidelity.QUOTED,
    DetailLevel.COMPACTED: TurnFidelity.SUMMARISED,
    DetailLevel.ARCHIVED: TurnFidelity.FACTS_ONLY,
    DetailLevel.OUT_OF_VIEW: TurnFidelity.FACTS_ONLY,
}
_CLASSIFICATION_AT_LEVEL = {
    DetailLevel.KEPT: SweepClassification.KEEP,
    DetailLevel.COMPACTED: SweepClassification.COMPACT,
    DetailLevel.ARCHIVED: SweepClassification.ARCHIVE,
    DetailLevel.OUT_OF_VIEW: SweepClassification.ARCHIVE,
}


class TurnState(BaseModel):
    """One recorded turn: where it is now, and what the next cut would make of it.

    A turn's own text, size and score are known only while it is verbatim; after a
    cut, what remains is the front entry or the facts that stand in for it.
    """

    turn_index: int
    fidelity: TurnFidelity
    role: str | None = None
    tokens: int | None = None
    text: str | None = None
    front_entry: int | None = None        # index into `front_entries`, while quoted or summarised
    facts_stored: int = 0
    facts_sent: int = 0                   # of those, how many the front carries
    score: float | None = None            # current relevance, while verbatim
    classification: SweepClassification | None = None
    becomes: TurnFidelity | None = None   # set when the next cut would take this turn


class FrontEntryState(BaseModel):
    kind: DetailLevel                     # kept or compacted: the front carries no text for an archived span
    source_turns: list[int]
    text: str                             # as the model receives it, marker included
    stated_text: str                      # what it says, without the marker - for a person to read
    tokens: int


class FactState(BaseModel):
    turn: int
    source_turns: list[int]
    topic: str
    content: str
    knowledge_type: KnowledgeType
    status: KnowledgeStatus
    sent: bool                            # in the front, so the model sees it
    sent_content: str                     # the content as the front carries it, cut if over the line limit
    shortened: bool                       # the front carries only the start of `content`
    superseded_by_turn: int | None = None


class NextCutState(BaseModel):
    turns: list[int]                      # the verbatim turns the cut would take
    compactable_tokens: int
    min_cut_tokens: int
    batch_ready: bool                     # necessary, not sufficient - see `CutPreview`


class KnowledgeState(BaseModel):
    """Everything the dashboard shows about one conversation's content."""

    session_id: str
    version: int
    recorded_turns: int
    cuts: int
    context_tokens: int
    front_text: str | None                # the front exactly as the model receives it
    front_tokens: int
    front_entries: list[FrontEntryState]
    turns: list[TurnState]
    facts: list[FactState]
    next_cut: NextCutState
    aging_in_flight: bool
    last_activity: datetime


def project_knowledge_state(inspection: SessionInspection) -> KnowledgeState:
    """Pivot one session's inspection from generations to turns and facts."""
    snapshot = inspection.snapshot
    front = inspection.front
    entries = tuple((span, text) for span, text in (front.entries if front is not None else ())
                    if span.level in (DetailLevel.KEPT, DetailLevel.COMPACTED))
    sent = set(front.facts) if front is not None else set()

    facts_by_turn: dict[int, list[KnowledgeEntry]] = defaultdict(list)
    for fact in snapshot.knowledge:
        facts_by_turn[fact.message_turn].append(fact)

    return KnowledgeState(
        session_id=inspection.session_id,
        version=snapshot.version,
        recorded_turns=snapshot.recorded_turn_count,
        cuts=snapshot.cut_count,
        context_tokens=inspection.context_tokens,
        front_text=front.message.content if front is not None else None,
        front_tokens=front.message.token_count if front is not None else 0,
        front_entries=[_front_entry_state(span, text) for span, text in entries],
        turns=_turn_states(inspection, entries, facts_by_turn, sent),
        facts=_fact_states(snapshot.knowledge, sent),
        next_cut=NextCutState(turns=list(inspection.next_cut.turns),
                              compactable_tokens=inspection.next_cut.compactable_tokens,
                              min_cut_tokens=inspection.next_cut.min_cut_tokens,
                              batch_ready=inspection.next_cut.batch_ready),
        aging_in_flight=inspection.aging_in_flight,
        last_activity=inspection.last_activity,
    )


def _front_entry_state(span: Span, text: str) -> FrontEntryState:
    return FrontEntryState(kind=span.level, source_turns=list(span.turn_indices), text=text,
                           stated_text=stated_text(span), tokens=count_tokens(text))


def _turn_states(inspection: SessionInspection, entries: tuple[tuple[Span, str], ...],
                 facts_by_turn: dict[int, list[KnowledgeEntry]], sent: set[KnowledgeEntry]) -> list[TurnState]:
    """One state per recorded turn, oldest first.

    A turn that is neither verbatim nor covered by a front entry has been reduced
    to facts: every departure from the young generation files a front entry or at
    least one fact, and an archived entry passes its coverage on to its facts.
    """
    covering = {turn: index for index, (span, _) in enumerate(entries) for turn in span.turn_indices}
    verbatim = {message.turn_index: message for message in inspection.tail}
    preview = inspection.next_cut

    states: list[TurnState] = []
    for turn in range(inspection.snapshot.recorded_turn_count):
        facts = facts_by_turn.get(turn, [])
        state = TurnState(turn_index=turn, fidelity=TurnFidelity.FACTS_ONLY, facts_stored=len(facts),
                          facts_sent=sum(fact in sent for fact in facts))
        if turn in verbatim:
            message, level = verbatim[turn], preview.levels.get(turn)
            state.fidelity = TurnFidelity.VERBATIM
            state.role, state.tokens, state.text = message.role, message.token_count, message.content
            if level is not None:
                state.score, state.classification = preview.scores[turn], _CLASSIFICATION_AT_LEVEL[level]
                state.becomes = _FIDELITY_AT_LEVEL[level]
        elif turn in covering:
            state.front_entry = covering[turn]
            quoted = entries[covering[turn]][0].level is DetailLevel.KEPT
            state.fidelity = TurnFidelity.QUOTED if quoted else TurnFidelity.SUMMARISED
        elif facts and all(fact.knowledge_status is KnowledgeStatus.SUPERSEDED for fact in facts):
            state.fidelity = TurnFidelity.SUPERSEDED
        states.append(state)
    return states


def _fact_states(knowledge: tuple[KnowledgeEntry, ...], sent: set[KnowledgeEntry]) -> list[FactState]:
    """Every fact ever filed, superseded ones included, ordered by the turn it came from.

    A superseded fact names the turn of the fact that replaced it: the next fact
    filed on the same topic that could supersede, which is the rule supersession
    follows.
    """
    states = []
    for position, fact in enumerate(knowledge):
        replacement = None
        if fact.knowledge_status is KnowledgeStatus.SUPERSEDED:
            replacement = next((later.message_turn for later in knowledge[position + 1:]
                                if later.topic_label == fact.topic_label and later.can_supersede), None)
        states.append(FactState(turn=fact.message_turn, source_turns=[fact.message_turn],
                                topic=fact.topic_label, content=fact.content, knowledge_type=fact.knowledge_type,
                                status=fact.knowledge_status, sent=fact in sent, sent_content=fact.content,
                                shortened=False,
                                superseded_by_turn=replacement))
    return sorted(states, key=lambda state: state.turn)
