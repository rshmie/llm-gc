"""Asking a running proxy for its counters, and showing them without claiming more than they say."""

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_gc.exceptions import ProxyUnavailableError

STATS_TIMEOUT_S = 5.0


class ProxyStats(BaseModel):
    """The proxy's `/stats` answer. Validated, because whatever is on that port may not be an LLM-GC proxy."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    requests: int
    gc_applied: int
    gc_bypassed: int
    tokens_in_client_history: int
    tokens_forwarded: int
    tokens_saved: int
    history_rewinds: int
    history_restarts: int
    sessions_seeded: int
    ratio: float | None


def fetch_proxy_stats(url: str, *, transport: httpx2.BaseTransport | None = None) -> ProxyStats:
    """GET the proxy's stats. A `transport` replaces the network, for tests.

    Raises:
        ProxyUnavailableError: Nothing answered, it answered an error, or the answer was not a proxy's.
    """
    try:
        with httpx2.Client(timeout=STATS_TIMEOUT_S, transport=transport) as client:
            response = client.get(url)
        response.raise_for_status()
        return ProxyStats.model_validate(response.json())
    except httpx2.HTTPStatusError as err:
        raise ProxyUnavailableError(f"{url} answered {err.response.status_code}") from err
    except (httpx2.HTTPError, OSError) as err:
        raise ProxyUnavailableError(f"nothing is answering at {url}: is `llm-gc start` running?") from err
    except (ValueError, ValidationError) as err:
        # ValidationError is a ValueError, and so is a body that is not JSON.
        raise ProxyUnavailableError(f"{url} answered, but not like an LLM-GC proxy") from err


def render_stats(stats: ProxyStats, base_url: str) -> str:
    """The counters as a short report. With no requests behind them there is nothing to measure."""
    lines = [f"LLM-GC proxy at {base_url}"]
    if stats.requests == 0:
        return "\n".join([*lines, "  no requests yet"])
    share = "" if stats.ratio is None else f"   ({stats.ratio:.0%} of it)"
    rows = [
        ("requests", f"{stats.requests:,}"),
        ("  through LLM-GC", f"{stats.gc_applied:,}"),
        ("  forwarded untouched", f"{stats.gc_bypassed:,}   (unreadable, or an error on the request path)"),
        ("client history", f"{stats.tokens_in_client_history:,} tokens"),
        ("forwarded", f"{stats.tokens_forwarded:,} tokens{share}"),
        ("history rewinds", f"{stats.history_rewinds:,}"),
        ("restarts", f"{stats.history_restarts:,}"),
        ("sessions seeded", f"{stats.sessions_seeded:,}"),
    ]
    lines += [f"  {label:<22}{value}" for label, value in rows]
    lines.append("Counted with LLM-GC's own tokenizer, not what the provider bills.")
    return "\n".join(lines)
