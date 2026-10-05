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


def test_the_memory_budget_is_a_share_of_the_window():
    assert GCConfig(context_window=10_000, max_memory_ratio=0.1).max_memory_tokens == 1_000


def test_an_unknown_setting_is_refused_rather_than_ignored():
    """Ignored, a misspelt setting looks configured when it is not."""
    with pytest.raises(ValidationError, match="min_cut_token"):
        GCConfig(min_cut_token=300)
