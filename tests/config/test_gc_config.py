from llm_gc.config.gc_config import GCConfig
import pytest
from pydantic import ValidationError


def test_invalid_threshold_raises_error():
    with pytest.raises(ValidationError):
        GCConfig(gc_threshold=-0.1)
    with pytest.raises(ValidationError):
        GCConfig(gc_threshold=2.5)

def test_invalid_context_window_raises_error():
    with pytest.raises(ValidationError):
        GCConfig(context_window=0)
    with pytest.raises(ValidationError):
        GCConfig(context_window=-5)

def test_valid_threshold_and_context_window():
    config = GCConfig(gc_threshold=0.7, context_window=10000)
    assert config.gc_threshold == 0.7
    assert config.context_window == 10000

