"""Asking a running proxy for its counters, and showing them without claiming more than they say."""

import math
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

# Whose count each figure is, named above its figures. The two tokenizers differ, so a
# reader who takes one for the other misreads every number.
_OWN_COUNT = ("LLM-GC's own count", "an estimate, made the same way on both sides, so the share holds")
_PROVIDER_COUNT = ("The provider's own count", "what it bills")
_NOT_NETTED = (
    "Not taken from the saving: the two are not worth the same. With a warm cache a saved token "
    "would have been a cache read, the cheapest kind; these are the summary model's input and output."
)
_WEIGHED_AGAINST = "To weigh the saving, set it against the cache written on requests that carried a collection."
_RATE_NOTE = (
    "Model count: how many tokens the model counted for each of LLM-GC's, on the conversation's newest "
    "request. LLM-GC sizes the recap and the stale bound at this rate, so each is the share of the "
    "model's window its setting names."
)

Section = tuple[str, list[tuple[int, str]]]


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


class LastReplyStats(_Counters):
    """The newest answered request, counted by LLM-GC and by the provider."""

    tokens_forwarded: int
    provider_prompt_tokens: int
    token_rate: float | None


class TrafficStats(_Counters):
    requests: int
    tokens_from_client: int
    tokens_forwarded: int
    tokens_saved: int
    provider: ProviderUsageStats
    collection_cache_write_tokens: int
    last_reply: LastReplyStats | None
    summaries: SummaryStats


class WindowStats(_Counters):
    tokens: int
    reported: bool


class ConversationStats(_Counters):
    conversation_id: str
    agent: str | None
    model: str | None
    context_window: WindowStats | None
    collected: bool = True
    first_request_at: datetime
    last_request_at: datetime
    collections: int
    collections_not_yet_sent: int
    traffic: TrafficStats
    part: str | None = None  # main, subagent or earlier_main, within a client session


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
    collection_cache_write_tokens: int
    summaries: SummaryStats
    history_rewinds: int
    history_restarts: int
    sessions_seeded: int
    rewrites_refused: int
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
    for session in stats.client_sessions:
        lines += _client_session(session)
    if stats.other_conversations:
        lines += ["", INDENT + style("Conversations no client session named", bold=True)]
        lines += _table([(conversation.conversation_id, conversation) for conversation in stats.other_conversations])
    notes = []
    if any(c.traffic.last_reply is not None and c.traffic.last_reply.token_rate for c in _all_conversations(stats)):
        notes.append(_RATE_NOTE)
    if any(c.collections_not_yet_sent for c in _all_conversations(stats)):
        notes.append(
            "A collection not yet sent has saved nothing yet: its saving reaches the model with the "
            "conversation's next request."
        )
    if any(not c.collected for c in _all_conversations(stats)):
        notes.append(
            "Not collected: a subagent's conversation is recorded and sent as written "
            "(collect_subagents turns collection on)."
        )
    wrapped = [line for note in notes for line in _dim_lines(note, INDENT)]
    return "\n".join([*lines, *([""] + wrapped if wrapped else []), ""])


def _totals(stats: ProxyStats) -> list[str]:
    """LLM-GC's own count, then the provider's, each under its name; then what the proxy handled."""
    tokens: Section = (
        "Tokens",
        [
            (
                stats.tokens_from_client,
                "your clients sent " + _note("each request whole, added up: system prompt, tools and messages"),
            ),
            (stats.tokens_forwarded, "LLM-GC forwarded"),
            (stats.tokens_saved, _describe_saving(stats)),
        ],
    )
    handled: list[Section] = [
        (
            "Requests",
            [
                (stats.requests, "received"),
                (stats.gc_applied, "through LLM-GC"),
                (stats.gc_bypassed, "passed on as sent " + _note("unreadable, or an error in LLM-GC")),
                # LLM-GC's own fault, shown only once it has happened; the log has the provider's reason.
                *(
                    [
                        (
                            stats.rewrites_refused,
                            "refused as LLM-GC rewrote them " + _note("each sent again as written; the log says why"),
                        )
                    ]
                    if stats.rewrites_refused
                    else []
                ),
            ],
        ),
        (
            "Conversations",
            [
                (stats.sessions_seeded, "tracked " + _note("new, or picked up again after a restart or an edit")),
                # Only a client sending LLM-GC's session header can rewind; Claude Code never does,
                # so these show only once they have happened.
                *(
                    [(stats.history_rewinds, "rewound " + _note("an edit or a retry went back to an earlier point"))]
                    if stats.history_rewinds
                    else []
                ),
                *(
                    [
                        (
                            stats.history_restarts,
                            "started over " + _note("an edit reached a condensed part: one request went whole"),
                        )
                    ]
                    if stats.history_restarts
                    else []
                ),
            ],
        ),
    ]
    # One width for every count, so magnitudes compare at a glance across sections.
    width = max(len(f"{count:,}") for _, rows in (tokens, *handled) for count, _ in rows)
    lines = [
        _heading(*_OWN_COUNT),
        *_section(tokens, width),
        "",
        _heading(*_PROVIDER_COUNT),
        *_provider_counts(stats),
        "",
    ]
    for section in handled:
        lines += [*_section(section, width), ""]
    return lines[:-1]


def _heading(name: str, meaning: str) -> str:
    return INDENT + style(name, bold=True, underline=True) + style(f"  {meaning}", dim=True)


def _section(section: Section, width: int) -> list[str]:
    title, rows = section
    # Padded before styling: escape codes would count towards the width.
    return [
        f"{INDENT}{style((title if index == 0 else '').ljust(_SECTION_WIDTH), bold=True)}{count:>{width},}  {text}"
        for index, (count, text) in enumerate(rows)
    ]


def _provider_counts(stats: ProxyStats) -> list[str]:
    """What the provider reported on the client's replies and on LLM-GC's own calls: set beside the saving, never netted."""
    usage, spent, carried = stats.provider_usage, stats.summaries, stats.collection_cache_write_tokens
    on_collections = f" ({carried:,} on requests that carried a collection)" if carried else ""
    provider = (
        f"Reported on {usage.replies:,} replies: {usage.input_tokens:,} input, "
        f"{usage.cache_read_input_tokens:,} cache read, {usage.cache_creation_input_tokens:,} cache write"
        f"{on_collections}, {usage.output_tokens:,} output."
        if usage.replies
        else "No reply has reported its usage yet."
    )
    own = (
        f"{spent.calls:,} calls to summarise and read facts: {spent.input_tokens:,} tokens in, "
        f"{spent.output_tokens:,} out."
        if spent.calls
        else "None yet: nothing has been summarised."
    )
    note = " ".join([_NOT_NETTED, *([_WEIGHED_AGAINST] if carried else [])])
    return [
        *_labelled("Provider", provider),
        *_labelled("LLM-GC's calls", own),
        *(_dim_lines(note, INDENT + " " * _SECTION_WIDTH) if spent.calls else []),
    ]


def _labelled(label: str, text: str) -> list[str]:
    """A section's label and its text, wrapped under the text's first line."""
    under = INDENT + " " * _SECTION_WIDTH
    first, *rest = textwrap.wrap(text, _NOTE_WIDTH, initial_indent=under, subsequent_indent=under)
    return [INDENT + style(label.ljust(_SECTION_WIDTH), bold=True) + first.lstrip(), *rest]


def _client_session(session: ClientSessionStats) -> list[str]:
    traffic = session.traffic
    saved = (
        f"by LLM-GC's count, saved {traffic.tokens_saved:,} of {traffic.tokens_from_client:,} "
        f"({_share(traffic.tokens_saved, traffic.tokens_from_client)})"
        if traffic.tokens_from_client
        else ""
    )
    span = f"{_clock(session.first_request_at)} to {_clock(session.last_request_at)}"
    header = f"Claude Code session {session.client_session[:_SHORT_ID_CHARS]}"
    lines = [
        "",
        INDENT + style(header, bold=True) + style(f"  {span} · {traffic.requests:,} requests · {saved}", dim=True),
    ]
    subagents = 0
    rows: list[tuple[str, ConversationStats | TrafficStats]] = []
    for conversation in session.conversations:
        if conversation.part == "subagent":
            subagents += 1
            rows.append((f"Subagent {subagents} ({(conversation.agent or '')[:_SHORT_ID_CHARS]})", conversation))
        else:
            rows.append(
                ({"main": "Main", "earlier_main": "Earlier main"}.get(conversation.part or "", "Main"), conversation)
            )
    if session.outside_conversations.requests:
        rows.append(("Other requests", session.outside_conversations))
    return lines + _table(rows)


def _table(rows: list[tuple[str, ConversationStats | TrafficStats]]) -> list[str]:
    cells = []
    for label, row in rows:
        collections = window = rate = ""
        if isinstance(row, ConversationStats):
            collections = (
                "not collected"
                if not row.collected
                else f"{row.collections}"
                + (f" ({row.collections_not_yet_sent} not yet sent)" if row.collections_not_yet_sent else "")
            )
            window = _window(row.context_window)
            last = row.traffic.last_reply
            rate = f"{last.token_rate:.2f}×" if last is not None and last.token_rate else ""
        cells.append((label, row.traffic if isinstance(row, ConversationStats) else row, collections, window, rate))
    width = max(len("Collections"), *(len(cell[2]) for cell in cells)) + 2
    window_width = max(len("Window"), *(len(cell[3]) for cell in cells)) + 2
    # Only once a reply has measured a rate: a column of blanks would say nothing.
    rate_head = "Model count" if any(cell[4] for cell in cells) else ""
    head = (
        f"{'Conversation':<{_LABEL_WIDTH}}{'Requests':>9}{'Sent':>12}{'Saved':>11}  {'Collections':<{width}}"
        f"{'Window':<{window_width}}{rate_head}"
    )
    lines = [INDENT * 2 + style(head.rstrip(), dim=True)]
    for label, traffic, collections, window, rate in cells:
        lines.append(
            f"{INDENT * 2}{label:<{_LABEL_WIDTH}}{traffic.requests:>9,}{traffic.tokens_from_client:>12,}"
            f"{traffic.tokens_saved:>11,}  {collections:<{width}}{window:<{window_width}}{rate}".rstrip()
        )
    return lines


def _window(window: WindowStats | None) -> str:
    if window is None:
        return ""
    tokens = window.tokens
    size = (
        f"{tokens // 1_000_000}M"
        if tokens % 1_000_000 == 0
        else f"{tokens // 1000}k"
        if tokens % 1000 == 0
        else f"{tokens:,}"
    )
    return size if window.reported else f"{size} (assumed)"


def _all_conversations(stats: ProxyStats) -> list[ConversationStats]:
    return [c for session in stats.client_sessions for c in session.conversations] + stats.other_conversations


def _clock(moment: datetime) -> str:
    return moment.astimezone().strftime("%H:%M")


def _describe_saving(stats: ProxyStats) -> str:
    if stats.tokens_saved == 0 or not stats.tokens_from_client:
        return "saved " + _note("nothing condensed yet")
    return f"saved ({_share(stats.tokens_saved, stats.tokens_from_client)})"


def _share(part: int, whole: int) -> str:
    """A whole percent, from the counts rather than the served ratio, which is already rounded. A saving too
    small to round to 1% reads "<1%", never "0%" beside the tokens it saved.

    Halves round up, as the dashboard's `Math.round` does, so both show the same figure; `round()` would
    round them to even.
    """
    percent = math.floor(part / whole * 100 + 0.5)
    return "<1%" if part > 0 and percent == 0 else f"{percent}%"


def _note(text: str) -> str:
    return style(f"({text})", dim=True)


def _dim_lines(text: str, indent: str) -> list[str]:
    return [
        style(line, dim=True)
        for line in textwrap.wrap(text, _NOTE_WIDTH, initial_indent=indent, subsequent_indent=indent)
    ]
