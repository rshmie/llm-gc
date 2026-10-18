"""Build a provider client from a name: the one place that maps a vendor to its adapter.

Used by the application's settings, and available to library users who pick a vendor by
name. A new vendor is one entry here plus its adapter; nothing else in the engine learns
its name.
"""

from llm_gc.exceptions import ConfigError
from llm_gc.llm.anthropic_client import ANTHROPIC_BASE_URL, DEFAULT_ANTHROPIC_MODEL, AnthropicClient
from llm_gc.llm.openai_client import DEFAULT_OPENAI_MODEL, OPENAI_BASE_URL, OpenAIClient
from llm_gc.llm.provider import LLMProvider

PROVIDER_NAMES = ("anthropic", "openai")
# The model each vendor's client uses when none is named.
DEFAULT_MODELS = {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL}


def create_provider(
    name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    auth_token: str | None = None,
) -> LLMProvider:
    """A client for the named vendor, with that vendor's own defaults for what is left out.

    Args:
        name: One of `PROVIDER_NAMES`.
        model: The model, as the vendor (or a gateway in front of it) names it.
        base_url: The vendor's API root, overridable for a gateway.
        api_key: The vendor's API key.
        auth_token: A bearer token, for a gateway that takes one instead of a key.
            Anthropic only: OpenAI's key is itself a bearer token.

    Raises:
        ConfigError: An unknown vendor, a credential the vendor cannot take, or a missing one.
    """
    if name == "anthropic":
        return AnthropicClient(
            api_key=api_key,
            auth_token=auth_token,
            model=model or DEFAULT_MODELS["anthropic"],
            base_url=base_url or ANTHROPIC_BASE_URL,
        )
    if name == "openai":
        if auth_token:
            raise ConfigError(
                "openai authenticates with an API key, which it sends as a bearer token; "
                "auth_token applies to anthropic only"
            )
        return OpenAIClient(
            api_key or "", model=model or DEFAULT_MODELS["openai"], base_url=base_url or OPENAI_BASE_URL
        )
    raise ConfigError(f"unknown provider {name!r}: choose one of {', '.join(PROVIDER_NAMES)}")
