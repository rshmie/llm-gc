"""
Context-aware garbage collection for LLMs with built-in epistemic transparency.

The names in `__all__` are the library's public API. Anything imported from a
subpackage instead is internal and may change in any release.
"""

import logging

from llm_gc.config import GCConfig
from llm_gc.config.constants import APP_VERSION
from llm_gc.engine import GCResult, GCStatus, GcService, build_gc_service
from llm_gc.exceptions import (
    ConfigError,
    LLMAuthError,
    LLMConnectionError,
    LLMContextOverflowError,
    LLMError,
    LLMGCError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from llm_gc.llm import AnthropicClient, LLMProvider, LLMResponse, OpenAIClient, StopReason
from llm_gc.models import Message
from llm_gc.utils import count_tokens

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = APP_VERSION

__all__ = [
    # Configure and run the engine
    "GCConfig",
    "GCResult",
    "GCStatus",
    "GcService",
    "Message",
    "build_gc_service",
    "count_tokens",
    # The model LLM-GC calls: a built-in client, or your own behind `LLMProvider`
    "AnthropicClient",
    "LLMProvider",
    "LLMResponse",
    "OpenAIClient",
    "StopReason",
    # Errors
    "ConfigError",
    "LLMAuthError",
    "LLMConnectionError",
    "LLMContextOverflowError",
    "LLMError",
    "LLMGCError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTimeoutError",
    "__version__",
]
