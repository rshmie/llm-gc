"""Asking a running proxy for its counters, and showing them without claiming more than they say."""

import textwrap
from datetime import datetime

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_gc.cli.style import INDENT, style
from llm_gc.exceptions import ProxyUnavailableError

STATS_TIMEOUT_S = 5.0
# Wide enough for the longest section title, "LLM-GC's calls", and the gap after it.
_SECTION_WIDTH = 16
# The conversation table's first column: "Subagent 12 (afc5b6eb)" and a gap.
_LABEL_WIDTH = 26
_SHORT_ID_CHARS = 8
# Notes wrap here, so they read on a narrow terminal.
_NOTE_WIDTH = 96


class _Counters(BaseModel):
    # Unknown fields are ignored, so an older CLI still reads a newer proxy; known ones are required.
    model_config = ConfigDict(extra="ignore", frozen=True)


class ProviderUsageStats(_Counters):
    """What the provider reported on its replies: its own count, what it bills."""

    replies: int
    input_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    output_tokens: int


class SummaryStats(_Counters):
    """LLM-GC's own model calls: summaries and fact reads."""

    calls: int
    input_tokens: int
    output_tokens: int


class TrafficStats(_Counters):
    requests: int
    tokens_from_client: int
    tokens_forwarded: int
    tokens_saved: int
    provider: ProviderUsageStats
    summaries: SummaryStats


class WindowStats(_Counters):
    tokens: int
    reported: bool


class ConversationStats(_Counters):
    conversation_id: str
    agent: str | None
    model: str | None
    context_window: WindowStats | None
    first_request_at: datetime
    last_request_at: datetime
    collections: int
    collections_not_yet_sent: int
    traffic: TrafficStats
    part: str | None = None      # main, subagent or earlier_main, within a client session


class ClientSessionStats(_Counters):
    client_session: str
    first_request_at: datetime
    last_request_at: datetime
    traffic: TrafficStats
    outside_conversations: TrafficStats
    conversations: list[ConversationStats]


class ProxyStats(_Counters):
    """The proxy's `/stats` answer. Validated, because whatever is on that port may not be an LLM-GC proxy."""

    requests: int
    gc_applied: int
    gc_bypassed: int
    tokens_from_client: int
    tokens_forwarded: int
    tokens_saved: int
    ratio: float | None
    provider_usage: ProviderUsageStats
    summaries: SummaryStats
    history_rewinds: int
    history_restarts: int
    sessions_seeded: int
    client_sessions: list[ClientSessionStats]
    other_conversations: list[ConversationStats]


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
    """The counters as a short report: the totals, then each client session's conversations."""
    lines = ["", INDENT + style("LLM-GC stats", bold=True) + "  " + style(base_url, fg="cyan"), ""]
    if stats.requests == 0:
        return "\n".join([*lines, INDENT + "No requests yet: point a client at this address and send a message.", ""])

    lines += _totals(stats)
    lines += _measured_elsewhere(stats)
    for session in stats.client_sessions:
        lines += _client_session(session)
    if stats.other_conversations:
        lines += ["", INDENT + style("Conversations no client session named", bold=True)]
        lines += _table([(conversation.conversation_id, conversation) for conversation in stats.other_conversations])
    notes = ["Sent and Saved are counted with LLM-GC's own tokenizer, the same way on both sides; "
             "the provider's line is its own count, what it bills."]
    if any(c.collections_not_yet_sent for c in _all_conversations(stats)):
        notes.append("A collection not yet sent has saved nothing yet: its saving reaches the model with the "
                     "conversation's next request.")
    wrapped = [style(line, dim=True) for note in notes
               for line in textwrap.wrap(note, _NOTE_WIDTH, initial_indent=INDENT, subsequent_indent=INDENT)]
    return "\n".join([*lines, "", *wrapped, ""])


def _totals(stats: ProxyStats) -> list[str]:
    share = "" if stats.ratio is None else f" ({stats.ratio:.0%})"
    sections: list[tuple[str, list[tuple[int, str]]]] = [
        ("Tokens", [
            (stats.tokens_from_client, "your clients sent " + _note("whole requests: system prompt, tools and messages")),
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
    lines: list[str] = []
    for title, rows in sections:
        for index, (count, text) in enumerate(rows):
            # Padded before styling: escape codes would count towards the width.
            label = style((title if index == 0 else "").ljust(_SECTION_WIDTH), bold=True)
            lines.append(f"{INDENT}{label}{count:>{width},}  {text}")
        lines.append("")
    return lines


def _measured_elsewhere(stats: ProxyStats) -> list[str]:
    """What the provider reported, and what LLM-GC's own calls spent: neither is LLM-GC's own count of a request."""
    usage, spent = stats.provider_usage, stats.summaries
    provider = (f"Reported on {usage.replies:,} replies: {usage.input_tokens:,} input, "
                f"{usage.cache_read_input_tokens:,} cache read, {usage.cache_creation_input_tokens:,} cache write, "
                f"{usage.output_tokens:,} output." if usage.replies else "No reply has reported its usage yet.")
    own = (f"{spent.calls:,} calls to summarise and read facts: {spent.input_tokens:,} tokens in, "
           f"{spent.output_tokens:,} out." if spent.calls else "None yet: nothing has been summarised.")
    return [INDENT + style("Provider".ljust(_SECTION_WIDTH), bold=True) + provider,
            INDENT + style("LLM-GC's calls".ljust(_SECTION_WIDTH), bold=True) + own]


def _client_session(session: ClientSessionStats) -> list[str]:
    traffic = session.traffic
    saved = (f"saved {traffic.tokens_saved:,} of {traffic.tokens_from_client:,} "
             f"({traffic.tokens_saved / traffic.tokens_from_client:.0%})" if traffic.tokens_from_client else "")
    span = f"{_clock(session.first_request_at)} to {_clock(session.last_request_at)}"
    header = f"Claude Code session {session.client_session[:_SHORT_ID_CHARS]}"
    lines = ["", INDENT + style(header, bold=True) + style(f"  {span} · {traffic.requests:,} requests · {saved}", dim=True)]
    subagents = 0
    rows: list[tuple[str, ConversationStats | TrafficStats]] = []
    for conversation in session.conversations:
        if conversation.part == "subagent":
            subagents += 1
            rows.append((f"Subagent {subagents} ({(conversation.agent or '')[:_SHORT_ID_CHARS]})", conversation))
        else:
            rows.append(({"main": "Main", "earlier_main": "Earlier main"}.get(conversation.part or "", "Main"),
                         conversation))
    if session.outside_conversations.requests:
        rows.append(("Other requests", session.outside_conversations))
    return lines + _table(rows)


def _table(rows: list[tuple[str, ConversationStats | TrafficStats]]) -> list[str]:
    cells = []
    for label, row in rows:
        collections = window = ""
        if isinstance(row, ConversationStats):
            collections = f"{row.collections}" + (f" ({row.collections_not_yet_sent} not yet sent)"
                                                  if row.collections_not_yet_sent else "")
            window = _window(row.context_window)
        cells.append((label, row.traffic if isinstance(row, ConversationStats) else row, collections, window))
    width = max(len("Collections"), *(len(collections) for _, _, collections, _ in cells)) + 2
    head = f"{'Conversation':<{_LABEL_WIDTH}}{'Requests':>9}{'Sent':>12}{'Saved':>11}  {'Collections':<{width}}Window"
    lines = [INDENT * 2 + style(head, dim=True)]
    for label, traffic, collections, window in cells:
        lines.append(f"{INDENT * 2}{label:<{_LABEL_WIDTH}}{traffic.requests:>9,}{traffic.tokens_from_client:>12,}"
                     f"{traffic.tokens_saved:>11,}  {collections:<{width}}{window}".rstrip())
    return lines


def _window(window: WindowStats | None) -> str:
    if window is None:
        return ""
    tokens = window.tokens
    size = (f"{tokens // 1_000_000}M" if tokens % 1_000_000 == 0 else
            f"{tokens // 1000}k" if tokens % 1000 == 0 else f"{tokens:,}")
    return size if window.reported else f"{size} (assumed)"


def _all_conversations(stats: ProxyStats) -> list[ConversationStats]:
    return [c for session in stats.client_sessions for c in session.conversations] + stats.other_conversations


def _clock(moment: datetime) -> str:
    return moment.astimezone().strftime("%H:%M")


def _describe_saving(stats: ProxyStats) -> str:
    if stats.tokens_saved == 0:
        return "saved " + _note("nothing condensed yet")
    share = "" if stats.ratio is None else f" ({1 - stats.ratio:.0%})"
    return f"saved{share}"


def _note(text: str) -> str:
    return style(f"({text})", dim=True)
