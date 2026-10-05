import pytest
from pydantic import ValidationError

from llm_gc.config.gc_config import GCConfig


def test_a_threshold_outside_zero_to_one_is_refused():
    with pytest.raises(ValidationError):
        GCConfig(keep_threshold=-0.1)
    with pytest.raises(ValidationError):
        GCConfig(archive_threshold=2.5)


def test_a_context_window_must_be_positive():
    with pytest.raises(ValidationError):
        GCConfig(context_window=0)
    with pytest.raises(ValidationError):
        GCConfig(context_window=-5)


def test_the_recap_budget_is_a_share_of_the_window():
    assert GCConfig(context_window=10_000, max_recap_ratio=0.3).max_recap_tokens == 3_000


def test_the_keep_line_must_be_above_the_archive_line():
    with pytest.raises(ValidationError, match="must be above"):
        GCConfig(keep_threshold=0.3, archive_threshold=0.3)


def test_a_margin_wider_than_half_the_band_is_refused():
    """A score between the lines could then never move a span either way."""
    with pytest.raises(ValidationError, match="level_margin"):
        GCConfig(keep_threshold=0.7, archive_threshold=0.3, level_margin=0.2)


def test_an_unknown_setting_is_refused_rather_than_ignored():
    """Ignored, a misspelt setting looks configured when it is not."""
    with pytest.raises(ValidationError, match="min_cut_token"):
        GCConfig(min_cut_token=300)
