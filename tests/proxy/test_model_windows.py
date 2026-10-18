"""Each model's window: asked once, off the request path, and the configured one meanwhile."""

import asyncio
import logging

from llm_gc.proxy.model_windows import MAX_MODELS_LOOKED_UP, ModelWindow, ModelWindows

CONFIGURED = ModelWindow(200_000, reported=False)


class _Upstream:
    def __init__(self, windows: dict[str, int | None] | None = None, fails: bool = False) -> None:
        self.windows = windows or {}
        self.fails = fails
        self.asked: list[str] = []

    async def __call__(self, model: str) -> int | None:
        self.asked.append(model)
        if self.fails:
            raise ConnectionError("gateway down")
        return self.windows.get(model)


def _windows(upstream, pending: list | None = None) -> ModelWindows:
    """Lookups are held in `pending` and run when the test says, as a background task would run later."""
    queue = pending if pending is not None else []
    return ModelWindows(200_000, upstream, queue.append)


def _run(pending: list) -> None:
    while pending:
        asyncio.run(pending.pop(0))


def test_a_model_is_sized_by_the_configured_window_until_the_upstream_answers():
    upstream, pending = _Upstream({"claude-opus-5-5": 1_000_000}), []
    windows = _windows(upstream, pending)

    assert windows.window_for("claude-opus-5-5") == CONFIGURED
    assert windows.window_for("claude-opus-5-5") == CONFIGURED  # still asking: never asked twice
    _run(pending)

    assert windows.window_for("claude-opus-5-5") == ModelWindow(1_000_000, reported=True)
    assert upstream.asked == ["claude-opus-5-5"]


def test_each_model_has_its_own_window():
    upstream, pending = _Upstream({"claude-opus-5-5": 1_000_000, "claude-haiku-4-5": 200_000}), []
    windows = _windows(upstream, pending)
    windows.window_for("claude-opus-5-5")
    windows.window_for("claude-haiku-4-5")
    _run(pending)

    assert windows.window_for("claude-haiku-4-5") == ModelWindow(200_000, reported=True)
    assert windows.window_for("claude-opus-5-5").tokens == 1_000_000


def test_a_model_the_upstream_does_not_know_stays_on_the_configured_window_and_is_not_asked_again(caplog):
    caplog.set_level(logging.WARNING, logger="llm_gc.proxy.model_windows")
    upstream, pending = _Upstream(), []
    windows = _windows(upstream, pending)
    windows.window_for("renamed-by-a-gateway")
    _run(pending)

    assert windows.window_for("renamed-by-a-gateway") == CONFIGURED
    assert upstream.asked == ["renamed-by-a-gateway"]
    assert "reports no context window for renamed-by-a-gateway" in caplog.text


def test_a_failed_lookup_is_said_once_and_falls_back(caplog):
    caplog.set_level(logging.WARNING, logger="llm_gc.proxy.model_windows")
    upstream, pending = _Upstream(fails=True), []
    windows = _windows(upstream, pending)
    windows.window_for("claude-opus-5-5")
    _run(pending)

    assert windows.window_for("claude-opus-5-5") == CONFIGURED
    assert "ConnectionError" in caplog.text and "gateway down" not in caplog.text


def test_without_a_lookup_or_a_model_name_nothing_is_asked():
    pending: list = []
    assert ModelWindows(200_000, None, pending.append).window_for("claude-opus-5-5") == CONFIGURED
    assert _windows(_Upstream(), pending).window_for(None) == CONFIGURED
    assert pending == []


def test_a_client_naming_endless_models_cannot_make_endless_lookups():
    upstream, pending = _Upstream(), []
    windows = _windows(upstream, pending)
    for index in range(MAX_MODELS_LOOKED_UP + 10):
        windows.window_for(f"model-{index}")
    _run(pending)

    assert len(upstream.asked) == MAX_MODELS_LOOKED_UP
