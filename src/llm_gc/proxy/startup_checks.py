"""What the proxy checks before it serves: settings that would otherwise fail at every collection."""

from llm_gc.exceptions import ConfigError, LLMAuthError, LLMError, LLMRequestRefusedError, LLMResponseError
from llm_gc.llm import LLMProvider
from llm_gc.models import Message
from llm_gc.utils import count_tokens

# One token is enough to show the upstream accepts the model and the credential, at almost no cost.
CHECK_OUTPUT_TOKENS = 1
CHECK_TIMEOUT_S = 30.0
_CHECK_PROMPT = "Reply with OK."


async def check_summariser(provider: LLMProvider, *, timeout_s: float = CHECK_TIMEOUT_S) -> str | None:
    """Ask the summariser for one token, so a model or credential the upstream refuses stops the proxy at startup.

    Unchecked, a refused summariser fails at every collection: each one is dropped as
    no smaller, nothing is ever compacted, and the reason shows only on the dashboard.

    Returns:
        None when the upstream answered. Otherwise a warning: it could not be asked,
        being unreachable, busy or rate-limited, which may pass, so the proxy starts anyway.

    Raises:
        ConfigError: The upstream refused the credential or the request itself, which no
            retry fixes: most often a model it does not offer, or an account out of credit.
            Chained to the provider's error.
    """
    message = Message(role="user", content=_CHECK_PROMPT, token_count=count_tokens(_CHECK_PROMPT), turn_index=0)
    try:
        await provider.generate(messages=[message], max_output_tokens=CHECK_OUTPUT_TOKENS, timeout_s=timeout_s)
    except LLMAuthError as err:
        raise ConfigError(f"the upstream refused LLM-GC's credential: {err}") from err
    except LLMRequestRefusedError as err:
        raise ConfigError(f"the upstream refused LLM-GC's summary request: {err}") from err
    except LLMResponseError as err:
        if err.retryable:
            return f"the upstream could not answer the summariser check, so it is unchecked: {err}"
        if err.status_code is not None and err.status_code != 200:
            raise ConfigError(f"the upstream refused LLM-GC's summary request: {err}") from err
        return f"the upstream answered the summariser check in a form LLM-GC could not read: {err}"
    except LLMError as err:
        return f"the upstream could not be reached to check the summariser, so it is unchecked: {err}"
    return None
