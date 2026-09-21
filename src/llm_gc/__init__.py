"""
Context-aware garbage collection for LLMs with built-in epistemic transparency.
"""

import logging

from llm_gc.exceptions import (
    ConfigError,
    LLMAuthError,
    LLMContextOverflowError,
    LLMError,
    LLMGCError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    SessionLockTimeout,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "LLMAuthError",
    "LLMContextOverflowError",
    "LLMError",
    "LLMGCError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTimeoutError",
    "SessionLockTimeout",
    "__version__",
]
