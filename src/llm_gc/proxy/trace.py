"""One client session's requests as a trace: its turns, the requests each made, the subagents each started.

Application code. Built on every read from the request log, so a record settled since
the last read changes the trace with it. Structure only; the dashboard adds what it may
show of the words.

A turn is a span of the main conversation's time: it begins at a message the person
typed that the conversation kept, and runs to the next. The main conversation's
requests, side requests and requests not yet recognised belong to the turn their time
falls in. A subagent belongs to the turn whose reply started it, even while it runs on
after the person has typed again; a subagent no call is known to have started belongs
to the turn its first request falls in.
"""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal

from llm_gc.llm.metering import ModelUsage
from llm_gc.proxy.request_log import RequestRecord

RowKind = Literal["turn", "tool_round", "report", "notice", "reply", "task", "retry", "replaced", "pending",
                  "unclaimed", "closing"]
# What a request was, by its newest message, when that is not a message the person typed.
_FROM_NEWEST: dict[str, RowKind] = {"tool_results": "tool_round", "relayed": "report", "client": "notice",
                                    "assistant": "reply"}


def build_trace(records: Sequence[RequestRecord], client_session: str, *,
                summaries: ModelUsage | None = None) -> dict[str, Any] | None:
    """The trace of `client_session`, or None when no request named it.

    `summaries` are LLM-GC's own model calls for the session's conversations.
    """
    mine = [r for r in records if r.origin is not None and r.origin.session_id == client_session]
    if not mine:
        return None
    main = [r for r in mine if _agent(r) is None]
    mains = list(dict.fromkeys(r.conversation_id for r in main if r.conversation_id is not None))
    kinds = _kinds(mine, _closing(main, mains))
    starts = [r for r in main if kinds[r] == "turn"]
    at = [start.forwarded_at for start in starts]

    def by_time(record: RequestRecord) -> int:
        return bisect_right(at, record.forwarded_at) - 1

    # Every request's turn: the main conversation's by time; a subagent's by the request whose reply
    # started it, taken in the order the subagents started, so one a subagent started finds its turn.
    # Keyed by request ID, which is how a subagent names the request that started it; the records of
    # one client request are sent together, so they always fall in the same turn.
    turn_of = {r.request_id: by_time(r) for r in main}
    agents: dict[str, list[RequestRecord]] = {}
    for record in mine:
        if (agent := _agent(record)) is not None:
            agents.setdefault(agent, []).append(record)
    spawned: dict[str, list[str]] = {}
    agent_turn: dict[str, int] = {}
    for agent, requests in agents.items():
        link = next((r.spawned_by for r in requests if r.spawned_by is not None), None)
        if link is not None:
            spawned.setdefault(link.request_id, []).append(agent)
        turn = turn_of.get(link.request_id, by_time(requests[0])) if link is not None else by_time(requests[0])
        agent_turn[agent] = turn
        turn_of.update(dict.fromkeys((r.request_id for r in requests), turn))

    def row(record: RequestRecord) -> dict[str, Any]:
        # Only the record whose reply made the calls started the subagents: a request sent again after
        # a refusal shares its ID with the refused record, which had no reply.
        made_calls = record.outline is not None and bool(record.outline.tool_calls)
        return _row(record, kinds[record], spawned.get(record.request_id, []) if made_calls else [])

    nodes = {agent: {"agent": agent, "label": f"Subagent {number}", "conversation_id": requests[-1].conversation_id,
                     "collected": requests[-1].collected, "model": requests[-1].model,
                     "spawned_by": asdict(link) if (link := next((r.spawned_by for r in requests
                                                                    if r.spawned_by is not None), None)) else None,
                     "started_at": requests[0].forwarded_at.isoformat(), "ended_at": _ended(requests),
                     "totals": _totals(requests), "requests": [row(r) for r in requests]}
             for number, (agent, requests) in enumerate(agents.items(), start=1)}

    turns = []
    for index, start in enumerate(starts):
        requests = [r for r in main if turn_of[r.request_id] == index]
        everything = [r for r in mine if turn_of[r.request_id] == index]
        turns.append({"index": index + 1, "conversation_id": start.conversation_id, "position": start.position,
                      "part": "main" if start.conversation_id == mains[-1] else "earlier_main",
                      "new_conversation": index > 0 and start.conversation_id != starts[index - 1].conversation_id,
                      "started_at": start.forwarded_at.isoformat(), "ended_at": _ended(everything),
                      "totals": _totals(everything), "requests": [row(r) for r in requests],
                      "subagents": [nodes[agent] for agent, turn in agent_turn.items() if turn == index]})
    return {"client_session": client_session, "started_at": mine[0].forwarded_at.isoformat(), "ended_at": _ended(mine),
            "totals": _totals(mine),
            "summaries": asdict(summaries) if summaries is not None else None,
            # Requests before the first message the person typed that a conversation kept: checks, titles.
            "before": [row(r) for r in main if turn_of[r.request_id] < 0],
            "turns": turns,
            "subagents_before": [nodes[agent] for agent, turn in agent_turn.items() if turn < 0]}


def _agent(record: RequestRecord) -> str | None:
    return record.origin.agent_id if record.origin is not None else None


def _closing(main: Sequence[RequestRecord], mains: Sequence[str]) -> set[RequestRecord]:
    """The last request of each main conversation a newer one replaced, when nothing carried its message on.

    After /compact it is the client asking for the summary the new conversation opens
    with; after an edited message, the message the new conversation no longer carries.
    Either way the person is not known to have typed it, so it begins no turn.
    """
    closing = set()
    for conversation in mains[:-1]:
        last = [r for r in main if r.conversation_id == conversation][-1]
        if last.kept is None and last.newest == "typed":
            closing.add(last)
    return closing


def _kinds(records: Sequence[RequestRecord], closing: set[RequestRecord]) -> dict[RequestRecord, RowKind]:
    """What each request was, from its newest message and how its conversation settled it."""
    contested = _contested(records)
    kinds: dict[RequestRecord, RowKind] = {}
    seen: set[tuple[str, str]] = set()
    for record in records:
        if record.conversation_id is not None:
            key = (record.conversation_id, record.newest_hash)
            if key in seen:
                kinds[record] = "retry"
                continue
            seen.add(key)
        kind: RowKind
        if record.kept is False:
            kind = "replaced"
        elif record.newest != "typed":
            kind = _FROM_NEWEST[record.newest]
        elif _agent(record) is not None and (record.conversation_id is not None or record.spawned_by is not None):
            kind = "task"
        elif record.conversation_id is None:
            kind = "unclaimed"
        elif record in closing:
            kind = "closing"
        elif record.kept is None and record in contested:
            kind = "pending"
        else:
            kind = "turn"
        kinds[record] = kind
    return kinds


def _contested(records: Sequence[RequestRecord]) -> set[RequestRecord]:
    """Requests whose position a later request of the same conversation filled with another newest message.

    Until one of them is carried on, neither is known to be kept, so a typed message there
    is not yet a turn.
    """
    contested: set[RequestRecord] = set()
    later: dict[tuple[str, int], set[str]] = {}
    for record in reversed(records):
        if record.conversation_id is None:
            continue
        hashes = later.setdefault((record.conversation_id, record.position), set())
        if hashes - {record.newest_hash}:
            contested.add(record)
        hashes.add(record.newest_hash)
    return contested


def _row(record: RequestRecord, kind: RowKind, spawned: list[str]) -> dict[str, Any]:
    outline = record.outline
    return {"request_id": record.request_id, "kind": kind, "conversation_id": record.conversation_id,
            "position": record.position, "model": record.model,
            "forwarded_at": record.forwarded_at.isoformat(),
            "ended_at": record.ended_at.isoformat() if record.ended_at is not None else None,
            "status": outline.status if outline is not None else None,
            "finished": outline.finished if outline is not None else False,
            "stop_reason": outline.stop_reason if outline is not None else None,
            "tool_calls": [call.name for call in outline.tool_calls] if outline is not None else [],
            "spawned": spawned, "tokens_from_client": record.tokens_from_client,
            "tokens_forwarded": record.tokens_forwarded, "collected": record.collected,
            "carries_collections": record.carries_collections, "switch_reasons": list(record.switch_reasons),
            "error_type": outline.error_type if outline is not None else None,
            "resent_as_written": record.resent_as_written,
            "provider": asdict(outline.usage) if outline is not None and outline.usage is not None else None}


def _totals(records: Sequence[RequestRecord]) -> dict[str, int]:
    usage = [r.outline.usage for r in records if r.outline is not None and r.outline.usage is not None]
    sent = sum(r.tokens_from_client for r in records)
    forwarded = sum(r.tokens_forwarded for r in records)
    return {"requests": len(records), "tokens_from_client": sent, "tokens_forwarded": forwarded,
            # Tokens the provider wrote to its cache on requests that carried a collection: the recap changed,
            # so the prompt after the system prompt was cached again. As the provider counts them.
            "collection_cache_write_tokens": sum(r.outline.usage.cache_creation_input_tokens for r in records
                                                 if r.carries_collections and r.outline is not None
                                                 and r.outline.usage is not None),
            "tokens_saved": sent - forwarded,
            "input_tokens": sum(u.input_tokens for u in usage),
            "cache_read_input_tokens": sum(u.cache_read_input_tokens for u in usage),
            "cache_creation_input_tokens": sum(u.cache_creation_input_tokens for u in usage),
            "output_tokens": sum(u.output_tokens for u in usage)}


def _ended(records: Sequence[RequestRecord]) -> str | None:
    """When the last of them came back; None while any is still running, or when there are none."""
    if not records or any(r.ended_at is None for r in records):
        return None
    return max(r.ended_at for r in records if r.ended_at is not None).isoformat()
