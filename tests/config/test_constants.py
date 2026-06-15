from llm_gc.config.constants import get_default_config
from llm_gc import __version__

def test_get_default_config_returns_dict():
    default_config = get_default_config()
    assert isinstance(default_config, dict)

def test_get_default_config_has_required_key():
    default_config = get_default_config()
    assert "app_name" in default_config
    assert "app_version" in default_config
    assert "model" in default_config
    assert "context_window" in default_config
    assert "gc_threshold" in default_config

def test_default_config_version_matches_package():
    assert __version__ == get_default_config()["app_version"]

def test_gc_threshold_in_valid_range():
    gc_threshold = get_default_config()["gc_threshold"]
    assert 0.0 <= gc_threshold <= 1.0