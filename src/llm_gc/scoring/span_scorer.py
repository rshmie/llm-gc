"""Scoring spans: how much each one matters to the conversation as it stands now.

Spans are scored again at every collection, and "now" moves with the
conversation, so a span it returns to scores higher than it did when it was
condensed. A span's score sets its detail level; see
doc/design/generational-memory/overview.md.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from llm_gc.exceptions import ConfigError
from llm_gc.models import Message
from llm_gc.models.message import person_text
from llm_gc.scoring.base_scorer import BaseScorer
from llm_gc.scoring.relevance_type import RelevanceType
from llm_gc.utils import TextCache, count_tokens_cached
from llm_gc.utils.text import significant_words, tokenize_words

# Liveness leads: a span that is recent, or that what is being discussed reaches,
# is part of the conversation as it stands. Importance lifts what states decisions
# and specifics, and alone can never take a span to the top band. Drafts, to be
# tuned on the decision-survival check.
DEFAULT_LIVENESS_WEIGHT = 0.6
DEFAULT_IMPORTANCE_WEIGHTS: Mapping[RelevanceType, float] = {RelevanceType.DECISION: 0.2, RelevanceType.DENSITY: 0.2}

# Per turn of distance from the newest turn: a span loses half its recency every
# fourteen turns. Slow on purpose: the score should archive only what has gone
# quiet, and leave bounding the front to the budget. At 0.15, the turn-level rate,
# measured spans were archived about seven turns after they were said.
DEFAULT_SPAN_RECENCY_DECAY = 0.05

# Spans whose importance one scorer remembers. One entry is a hash and two scores, about 400
# bytes, so this is about 8 MB. It has to hold every span of the conversations in use at once
# (a long conversation has a thousand or more); a span dropped is only scored again.
IMPORTANCE_REMEMBERED = 20_000


class Relevance(Protocol):
    """How closely each span matches what is being discussed now, each from 0 to 1."""

    def score(self, spans: Sequence[str], now: str) -> list[float]: ...


# Words saying that a message refers back to the conversation, not what it refers to:
# "as I mentioned earlier, what was the fee?" is about the fee. Counted as words asked
# about, they match no span and dilute the match with the span that is meant.
_BACK_REFERENCES = frozenset(
    {
        "mention",
        "mentioned",
        "earlier",
        "before",
        "previously",
        "discuss",
        "discussed",
        "say",
        "said",
        "tell",
        "told",
        "talk",
        "talked",
        "remember",
        "recall",
        "remind",
        "back",
        "above",
        "again",
        "conversation",
    }
)


class LexicalRelevance:
    """How much of what is being discussed now a span talks about, word for word.

    Each word asked about is weighted by how rare it is across this conversation's
    spans - a word every span uses says little about which one is meant - and a
    span earns that weight by repeating it, with diminishing credit: one mention
    earns half, ten earn nine-tenths. A span that discusses the subject outranks
    one that names it in passing, and a weak match stays weak. Words that only
    refer back ("as I mentioned earlier") are not counted. Misses paraphrase
    ("datastore" for "database"), which an embedding implementation fixes.
    """

    def score(self, spans: Sequence[str], now: str) -> list[float]:
        """See `Relevance.score`."""
        asked = significant_words(now) - _BACK_REFERENCES
        if not spans or not asked:
            return [0.0] * len(spans)
        mentions = [Counter(word for word in tokenize_words(text) if word in asked) for text in spans]
        spans_using = Counter(word for counts in mentions for word in counts)
        weight = {word: math.log(1 + len(spans) / (1 + spans_using[word])) for word in asked}
        total = sum(weight.values())
        return [
            sum(weight[word] * count / (count + 1) for word, count in counts.items()) / total for counts in mentions
        ]


@dataclass(frozen=True)
class SpanScore:
    """A span's combined score, and what it was combined from, for a reader to check.

    `liveness` is the larger of the span's recency and its relevance to now;
    `parts` holds every individual measure, recency and relevance included.
    """

    combined: float
    liveness: float
    parts: dict[RelevanceType, float]


class SpanScorer:
    """Scores every span at once, against the turns the caller calls "now".

    A span is **live** if it is recent or if what is being discussed now reaches
    it - the larger of the two, not their sum. Summed, recency would cap what an
    old span can score, and a span the conversation returns to could never be
    shown in full again. Importance is added on top.

    Batch rather than one span at a time: word rarity needs every span, and "now"
    is its own input rather than the end of one flat list, in which a span of
    twenty turns would count as one position.

    Every span is scored again at each pass, but only recency and relevance can have
    changed: both depend on where the conversation is now. Importance depends on the
    span's text alone, so it is measured once per text and remembered. An importance
    scorer must therefore read only the content of the message it is given.

    Raises:
        ConfigError: An importance scorer has no weight or a weight has no scorer,
            a weight is negative, or the weights add up to zero.
    """

    def __init__(
        self,
        importance: Sequence[BaseScorer] = (),
        *,
        relevance: Relevance | None = None,
        liveness_weight: float = DEFAULT_LIVENESS_WEIGHT,
        importance_weights: Mapping[RelevanceType, float] = DEFAULT_IMPORTANCE_WEIGHTS,
        recency_decay: float = DEFAULT_SPAN_RECENCY_DECAY,
    ) -> None:
        if {scorer.scorer_name for scorer in importance} != set(importance_weights):
            raise ConfigError(
                "every importance scorer needs a weight, and every weight a scorer: "
                f"scorers {sorted(s.scorer_name.value for s in importance)}, "
                f"weights {sorted(part.value for part in importance_weights)}"
            )
        weights = [liveness_weight, *importance_weights.values()]
        if min(weights) < 0 or sum(weights) <= 0:
            raise ConfigError("span weights must be non-negative and add up to more than zero")
        self.importance = tuple(importance)  # fixed once importance is remembered by text
        self.relevance = relevance if relevance is not None else LexicalRelevance()
        self.liveness_weight = liveness_weight
        self.importance_weights = dict(importance_weights)
        self.recency_decay = recency_decay
        self._total_weight = sum(weights)
        self._importance = TextCache(self._measure_importance, IMPORTANCE_REMEMBERED)

    @property
    def best_score_however_old(self) -> float:
        """The highest score a span of any age can reach.

        Every level must be reachable by an old span, or one the conversation
        returns to could never be restored to it. Recency falls to zero with age,
        but relevance to now does not, so liveness can still reach 1.
        """
        return (self.liveness_weight + sum(self.importance_weights.values())) / self._total_weight

    def score(self, spans: Sequence[Sequence[Message]], now: Sequence[Message], newest_turn: int) -> list[SpanScore]:
        """Score each span, in the order given.

        Args:
            spans: Each span's turns, in order.
            now: The turns that say what is being discussed: at a collection, what
                the user said in the tail; at a recall, the newest message. Only the
                words the person wrote are read, not what their client added.
            newest_turn: The index of the newest recorded turn, which recency is
                measured from.

        Raises:
            ValueError: The relevance implementation returned a score count that
                does not match the spans.
        """
        texts = ["\n".join(turn.content for turn in span) for span in spans]
        relevance = self.relevance.score(texts, "\n".join(person_text(turn) for turn in now))
        return [
            self._score_one(span, text, similarity, newest_turn)
            for span, text, similarity in zip(spans, texts, relevance, strict=True)
        ]

    def _score_one(self, span: Sequence[Message], text: str, similarity: float, newest_turn: int) -> SpanScore:
        recency = math.exp(-self.recency_decay * max(0, newest_turn - span[-1].turn_index))
        # A new dict each time: the remembered one is shared by every later pass.
        parts = {RelevanceType.RECENCY: recency, RelevanceType.SIMILARITY: similarity, **self._importance.get(text)}
        liveness = max(recency, similarity)
        importance = sum(weight * parts[part] for part, weight in self.importance_weights.items())
        return SpanScore(
            combined=(self.liveness_weight * liveness + importance) / self._total_weight, liveness=liveness, parts=parts
        )

    def _measure_importance(self, text: str) -> dict[RelevanceType, float]:
        """Each importance measure of a span's text.

        A span mixes speakers, so its text is given to the scorers as one message whose
        role and position mean nothing; they read its content.
        """
        joined = Message(role="user", content=text, token_count=count_tokens_cached(text))
        return {scorer.scorer_name: scorer.score(joined, [joined]).score for scorer in self.importance}
