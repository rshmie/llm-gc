"""Asking a running proxy for its counters, and showing them without claiming more than they say."""

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_gc.cli.style import INDENT, style
from llm_gc.exceptions import ProxyUnavailableError

STATS_TIMEOUT_S = 5.0
# Wide enough for the longest section title, "Conversations", and the gap after it.
_SECTION_WIDTH = 15


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
    """The counters as a short report, savings first. With no requests behind them there is nothing to measure."""
    lines = ["", INDENT + style("LLM-GC stats", bold=True) + "  " + style(base_url, fg="cyan"), ""]
    if stats.requests == 0:
        return "\n".join([*lines, INDENT + "No requests yet: point a client at this address and send a message.", ""])

    share = "" if stats.ratio is None else f" ({stats.ratio:.0%})"
    sections: list[tuple[str, list[tuple[int, str]]]] = [
        ("Tokens", [
            (stats.tokens_in_client_history, "the client sent"),
            (stats.tokens_forwarded, f"LLM-GC forwarded{share}"),
            (stats.tokens_saved, _describe_saving(stats)),
        ]),
        ("Requests", [
            (stats.requests, "received"),
            (stats.gc_applied, "through LLM-GC"),
            (stats.gc_bypassed, "passed on as sent " + _note("unreadable, or an error in LLM-GC")),
        ]),
        ("Conversations", [
            (stats.sessions_seeded, "tracked " + _note("new, or picked up again after a restart or an edit")),
            # Only a client sending LLM-GC's session header can rewind; Claude Code never does,
            # so these show only once they have happened.
            *([(stats.history_rewinds, "rewound " + _note("an edit or a retry went back to an earlier point"))]
              if stats.history_rewinds else []),
            *([(stats.history_restarts, "started over " + _note("an edit reached a condensed part: one request "
                                                                 "went whole"))]
              if stats.history_restarts else []),
        ]),
    ]
    width = max(len(f"{count:,}") for _, rows in sections for count, _ in rows)
    for title, rows in sections:
        for index, (count, text) in enumerate(rows):
            # Padded before styling: escape codes would count towards the width.
            label = style((title if index == 0 else "").ljust(_SECTION_WIDTH), bold=True)
            lines.append(f"{INDENT}{label}{count:>{width},}  {text}")
        lines.append("")
    lines += [INDENT + style("Counted with LLM-GC's own tokenizer, not what the provider bills.", dim=True), ""]
    return "\n".join(lines)


def _describe_saving(stats: ProxyStats) -> str:
    if stats.tokens_saved == 0:
        return "saved " + _note("nothing condensed yet")
    share = "" if stats.ratio is None else f" ({1 - stats.ratio:.0%})"
    return f"saved{share}"


def _note(text: str) -> str:
    return style(f"({text})", dim=True)
