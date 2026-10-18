"""How large each model's context window is, as the provider reports it. Application code.

Each conversation names its model on every request, and a session can run several - a
main conversation on one model, its subagents on another. So the window is looked up per
model, asked of the upstream once, off the request path, and remembered. Until the answer
arrives, or when there is none - a gateway that does not list models, a model it names
its own way, no credential of LLM-GC's own - the configured window is used, and said to be
assumed.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Distinct model names asked about. A client cannot make LLM-GC call the upstream without end.
MAX_MODELS_LOOKED_UP = 64

WindowLookup = Callable[[str], Awaitable[int | None]]


@dataclass(frozen=True, slots=True)
class ModelWindow:
    tokens: int
    reported: bool  # True when the provider reported it for this model; False for the configured window


class ModelWindows:
    """Each model's window: the provider's answer once it has one, the configured window until then."""

    def __init__(
        self, configured: int, lookup: WindowLookup | None, schedule: Callable[[Awaitable[None]], None]
    ) -> None:
        self._configured = ModelWindow(configured, reported=False)
        self._lookup = lookup
        self._schedule = schedule
        self._known: dict[str, int | None] = {}  # None: asked, and no window came back
        self._asking: set[str] = set()

    def window_for(self, model: str | None) -> ModelWindow:
        """The window to size a recap for. Never waits: an unknown model is asked about in the background."""
        if model is None or self._lookup is None:
            return self._configured
        if model in self._known:
            tokens = self._known[model]
            return ModelWindow(tokens, reported=True) if tokens is not None else self._configured
        if model not in self._asking and len(self._known) + len(self._asking) < MAX_MODELS_LOOKED_UP:
            self._asking.add(model)
            self._schedule(self._ask(model, self._lookup))
        return self._configured

    async def _ask(self, model: str, lookup: WindowLookup) -> None:
        try:
            tokens = await lookup(model)
        except Exception as err:
            # The reason's type only: an upstream's error text can echo the request.
            logger.warning(
                "could not ask the upstream for %s's context window (%s); using the configured %s",
                model,
                type(err).__name__,
                f"{self._configured.tokens:,}",
                extra={"model": model},
            )
            tokens = None
        else:
            if tokens is None:
                logger.warning(
                    "the upstream reports no context window for %s; using the configured %s",
                    model,
                    f"{self._configured.tokens:,}",
                    extra={"model": model},
                )
            else:
                logger.info(
                    "%s has a %s-token context window, as the upstream reports it",
                    model,
                    f"{tokens:,}",
                    extra={"model": model, "context_window": tokens},
                )
        finally:
            self._asking.discard(model)
        self._known[model] = tokens
