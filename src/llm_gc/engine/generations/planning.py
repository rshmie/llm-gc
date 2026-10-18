"""Choosing each span's detail level: the score decides the level, the budget caps the total.

A pure function of scores, costs and the spans' current levels - no model, no
I/O - so a collection and a recall can both run it, and it can be checked against
any number of generated conversations. See doc/design/generational-memory/overview.md.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llm_gc.exceptions import ConfigError
from llm_gc.models.generation_snapshot import DetailLevel

_RANK = {DetailLevel.OUT_OF_VIEW: 0, DetailLevel.ARCHIVED: 1, DetailLevel.COMPACTED: 2, DetailLevel.KEPT: 3}
_RICHEST_FIRST = (DetailLevel.KEPT, DetailLevel.COMPACTED, DetailLevel.ARCHIVED)


@dataclass(frozen=True)
class Bands:
    """The score lines between levels, and the margin a score must clear to cross one.

    Raises:
        ConfigError: The keep line is not above the archive line, or the margin
            is negative or so wide that a score inside the bands could never move
            a span either way.
    """

    keep: float
    archive: float
    margin: float

    def __post_init__(self) -> None:
        if not self.archive < self.keep:
            raise ConfigError(f"the keep line ({self.keep}) must be above the archive line ({self.archive})")
        if not 0 <= self.margin < (self.keep - self.archive) / 2:
            raise ConfigError(
                f"the level margin ({self.margin}) must be at least 0 and under half the gap "
                f"between the lines ({self.keep - self.archive})"
            )


@dataclass(frozen=True)
class LevelOptions:
    """One span as the planner sees it: its score, where it stands, and what each level costs.

    `compacted_tokens` is None when the span has no summary, so it cannot be
    compacted. `archivable` is False when its facts could not be read: archived, it
    would be only its opening line. `current` is None for a span the collection has
    just made.
    """

    score: float
    current: DetailLevel | None
    kept_tokens: int
    compacted_tokens: int | None
    archived_tokens: int
    archivable: bool = True

    def cost(self, level: DetailLevel) -> int:
        """Tokens the front spends on this span at `level`; a folded span costs nothing of its own."""
        if level is DetailLevel.KEPT:
            return self.kept_tokens
        if level is DetailLevel.COMPACTED:
            if self.compacted_tokens is None:
                raise ValueError("a span without a summary has no compacted cost")
            return self.compacted_tokens
        return self.archived_tokens if level is DetailLevel.ARCHIVED else 0

    def worthwhile(self, level: DetailLevel) -> bool:
        """Whether the span can be shown at `level`, for less than at every richer level it can be shown at."""
        if level is DetailLevel.COMPACTED and self.compacted_tokens is None:
            return False
        if level is DetailLevel.ARCHIVED and not self.archivable:
            return False
        richer = [
            self.cost(other)
            for other in _RICHEST_FIRST
            if _RANK[other] > _RANK[level] and (other is not DetailLevel.COMPACTED or self.compacted_tokens is not None)
        ]
        return all(self.cost(level) < cost for cost in richer)


@dataclass(frozen=True)
class Plan:
    """A level for every span, in the order given, and how the budget shaped it.

    `lowered` names the spans shown below the level their score earned, to fit;
    `folded` counts the oldest spans folded out of view. `over_budget` is True
    only when folding everything still did not fit.
    """

    levels: tuple[DetailLevel, ...]
    tokens: int
    lowered: frozenset[int]
    folded: int
    over_budget: bool


def plan_levels(spans: Sequence[LevelOptions], bands: Bands, budget: int, fold_line_tokens: int) -> Plan:
    """Choose a level for each span, oldest first in `spans`.

    Each span gets the richest level its score earns - with hysteresis, so a score
    near a line does not move it back and forth - and never a level that is not
    cheaper than a richer one. While the total is over `budget`, the coldest span
    is lowered one level at a time, the older first on equal scores. Only when no
    span can go lower are the oldest folded out of view, together, behind one line
    costing `fold_line_tokens`. Reaching the budget never raises a level.
    """
    levels = [_settle(span, _band(span, bands)) for span in spans]
    tokens = sum(span.cost(level) for span, level in zip(spans, levels, strict=True))

    lowered: set[int] = set()
    while tokens > budget:
        options = [
            (span.score, index, lower)
            for index, span in enumerate(spans)
            if (lower := _next_lower(span, levels[index])) is not None
        ]
        if not options:
            break
        _, index, lower = min(options, key=lambda option: (option[0], option[1]))
        tokens -= spans[index].cost(levels[index]) - spans[index].cost(lower)
        levels[index] = lower
        lowered.add(index)

    folded = 0
    while tokens > budget and folded < len(spans):
        tokens += (fold_line_tokens if folded == 0 else 0) - spans[folded].cost(levels[folded])
        levels[folded] = DetailLevel.OUT_OF_VIEW
        folded += 1

    return Plan(
        levels=tuple(levels),
        tokens=tokens,
        lowered=frozenset(lowered - set(range(folded))),
        folded=folded,
        over_budget=tokens > budget,
    )


def _band(span: LevelOptions, bands: Bands) -> DetailLevel:
    """The level the score earns, moving a span off its current level only past the margin.

    A folded span is below every band, so it returns to whatever its score earns
    moving up: folding answers the budget, never the score.
    """
    plain = _level_for(span.score, bands.keep, bands.archive)
    if span.current is None:
        return plain
    current = span.current
    up = _level_for(span.score, bands.keep + bands.margin, bands.archive + bands.margin)
    down = _level_for(span.score, bands.keep - bands.margin, bands.archive - bands.margin)
    if _RANK[up] > _RANK[current]:
        return up
    if _RANK[down] < _RANK[current]:
        return down
    return current


def _level_for(score: float, keep: float, archive: float) -> DetailLevel:
    if score >= keep:
        return DetailLevel.KEPT
    return DetailLevel.COMPACTED if score >= archive else DetailLevel.ARCHIVED


def _settle(span: LevelOptions, level: DetailLevel) -> DetailLevel:
    """The level itself if it is worthwhile, otherwise the nearest richer one that is."""
    for candidate in reversed(_RICHEST_FIRST[: _RICHEST_FIRST.index(level) + 1]):
        if span.worthwhile(candidate):
            return candidate
    return DetailLevel.KEPT


def _next_lower(span: LevelOptions, level: DetailLevel) -> DetailLevel | None:
    """The next poorer level worth using, or None at the floor; folding is not a lowering."""
    if level is DetailLevel.OUT_OF_VIEW:
        return None
    for candidate in _RICHEST_FIRST[_RICHEST_FIRST.index(level) + 1 :]:
        if span.worthwhile(candidate) and span.cost(candidate) < span.cost(level):
            return candidate
    return None
