"""The provider factory: names to adapters, with each vendor's own defaults."""

import pytest

from llm_gc.exceptions import ConfigError
from llm_gc.llm import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL
from llm_gc.llm.factory import create_provider


def test_anthropic_is_built_with_its_own_defaults():
    provider = create_provider("anthropic", api_key="k")
    assert provider.name == "anthropic"
    assert provider._model == DEFAULT_ANTHROPIC_MODEL


def test_openai_is_built_with_its_own_defaults():
    provider = create_provider("openai", api_key="k")
    assert provider.name == "openai"
    assert provider._model == DEFAULT_OPENAI_MODEL


def test_a_named_model_overrides_the_default():
    assert create_provider("openai", api_key="k", model="gpt-x")._model == "gpt-x"


def test_an_unknown_provider_names_the_choices():
    with pytest.raises(ConfigError, match="anthropic, openai"):
        create_provider("cohere", api_key="k")


def test_openai_refuses_a_bearer_token_that_would_be_silently_dropped():
    with pytest.raises(ConfigError, match="anthropic only"):
        create_provider("openai", api_key="k", auth_token="t")


def test_a_missing_credential_fails_at_construction():
    with pytest.raises(ConfigError):
        create_provider("openai")
