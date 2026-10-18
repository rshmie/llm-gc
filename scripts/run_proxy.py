#!/usr/bin/env python3
"""Try LLM-GC offline: no API key, no network, no cost.

The real proxy is `llm-gc start`. This script is for looking at the engine work.

**Demo (default):**

    python scripts/run_proxy.py

Starts the real proxy app against a stub upstream, replays a conversation through
real HTTP, and prints what the client sent versus what was forwarded, per request:
the dashboard shows the generations moving, this shows the wire.

**Serve offline, with the live dashboard:**

    python scripts/run_proxy.py --serve --dashboard

Serves the proxy on 9900 and the dashboard on 9901 with canned replies from the
upstream and canned summaries, then talk through it (`python scripts/chat.py`) and
watch http://127.0.0.1:9901. Summaries and facts are scripted and read by patterns, so
the wording is not a model's; the arithmetic is real.

The dashboard shows conversation content, so it is off unless asked for and only
ever listens on localhost.

Developer tooling, not library code, which is why it lives in `scripts/`.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from llm_gc.cli.logs import configure_logging  # noqa: E402
from llm_gc.cli.serve import serve_proxy  # noqa: E402
from llm_gc.cli.settings import load_settings  # noqa: E402
from llm_gc.config import GCConfig  # noqa: E402
from llm_gc.config.constants import (  # noqa: E402
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_LAST_N_TURNS_TO_KEEP,
    DEFAULT_MIN_CUT_TOKENS,
    DEFAULT_PROXY_CONTEXT_WINDOW,
)
from llm_gc.engine import GcService, build_gc_service  # noqa: E402
from llm_gc.extraction import PatternFactReader  # noqa: E402
from llm_gc.llm import FakeProvider, LLMResponse, StopReason  # noqa: E402
from llm_gc.proxy import DEFAULT_PROXY_PORT, SESSION_HEADER, create_app  # noqa: E402
from llm_gc.utils import count_tokens  # noqa: E402

SESSION = "demo"

# A conversation with facts worth remembering and filler worth forgetting, so the
# three generations all have something to do. The last turn deliberately asks about
# something stated near the start, which is the question the recap exists to
# answer.
CONVERSATION = [
    "We need to decide on the storage layer for the new ingest service before Friday.",
    "the database is postgres, and we are keeping ninety days of raw events.",
    "the cache is redis, sitting in front of the read path for the dashboard queries.",
    "We walked through the numbers for the ingest spike and it peaked near forty thousand a second.",
    "the queue is rabbitmq for now, though the platform team keeps suggesting we move to kafka.",
    "I spent the morning cleaning up the stale feature branches nobody had touched.",
    "We paired on the migration script for an hour and found two columns never backfilled.",
    "Also the flaky integration run came back again, same timeout in the fixture teardown.",
    "We went over the incident timeline and the gap was in the alerting, not the ingest path.",
    "Sketched the onboarding flow on the whiteboard, it needs one fewer screen than the current one.",
    "The second confirmation step has never caught a real mistake, so we are dropping it.",
    "So remind me, which database are we actually targeting for the ingest service?",
]

# Varied, because a stub that answers identically every time makes the forwarded
# context a wall of one sentence and hides what the proxy is actually doing to it.
STUB_REPLIES = [
    "Understood. I have noted that and will carry it forward.",
    "That follows from what we agreed earlier, so no change needed.",
    "Worth writing down - it changes the plan for the next step.",
    "Agreed. I would treat that as settled unless something contradicts it.",
    "Noted. That is consistent with the constraints you gave at the start.",
]


def _stub_reply(index: int) -> str:
    return STUB_REPLIES[index % len(STUB_REPLIES)]

_SCRIPTED_SUMMARIES = [
    "Earlier turns set a Friday deadline and chose the ingest storage layer.",
    "Postgres was chosen with ninety days of raw event retention, partitioned by day.",
    "Redis was placed in front of the dashboard read path with a short TTL.",
    "Ingest peaked near forty thousand events a second, so partitioning matters most.",
    "RabbitMQ was kept as the queue; Kafka was judged not worth its cost at this volume.",
    "Stale branches were cleaned and the migration script revealed two unbackfilled columns.",
    "A flaky teardown timeout was traced to a connection held past the test.",
    "The incident gap was in alerting rather than ingest, so thresholds are the fix.",
    "The onboarding flow lost a screen; the second confirmation never caught a mistake.",
] * 6



def demo_config() -> GCConfig:
    """Scaled so the recap's budget bites: older spans fold, recent ones are archived or summarised."""
    return GCConfig(
        context_window=900,
        keep_threshold=0.55,
        archive_threshold=0.25,
        last_n_turns_to_keep=3,
        # Scaled to this demo's small window; real conversations use far larger batches.
        min_cut_tokens=60,
    )


def serve_config(args: argparse.Namespace) -> GCConfig:
    """The library defaults, with the two knobs worth turning while watching."""
    return GCConfig(context_window=args.context_window, min_cut_tokens=args.min_cut_tokens,
                    last_n_turns_to_keep=args.keep_last)


def build_engine(config: GCConfig, *, compactor_kind: str) -> tuple[GcService, str]:
    """The proxy's engine, and a label saying which summariser it calls."""
    if compactor_kind == "noop":
        return (build_gc_service(config, summary_provider=None),
                "NoOpCompactor (baseline: wraps the run, does not summarise)")
    provider = FakeProvider(
        responses=[
            LLMResponse(
                text=text,
                stop_reason=StopReason.COMPLETE,
                input_tokens=0,
                output_tokens=count_tokens(text),
                model="scripted",
                provider="offline",
            )
            for text in _SCRIPTED_SUMMARIES
        ]
    )
    # Canned summaries are not a model: facts are read by patterns, from the user's own words.
    return (build_gc_service(config, summary_provider=provider, fact_reader=PatternFactReader()),
            "LLMCompactor -> scripted provider (wording canned, arithmetic real)")


def _stub_upstream(forwarded: list[dict]):
    """A stand-in provider that records what it was sent and always answers.

    Recording the forwarded body is the whole point of the demo: the interesting
    number is not what the provider said, it is what it was *asked*.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        forwarded.append(json.loads(request.content))
        reply = _stub_reply(len(forwarded) - 1)
        return httpx2.Response(
            200,
            json={
                "id": "msg_stub",
                "type": "message",
                "role": "assistant",
                "model": "stub-model",
                "content": [{"type": "text", "text": reply}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": count_tokens(reply)},
            },
        )

    return handler

def demo(compactor_kind: str) -> None:
    """Replay a conversation through the real proxy over real HTTP."""
    service, label = build_engine(demo_config(), compactor_kind=compactor_kind)
    forwarded: list[dict] = []

    def run_now(coro) -> None:
        # Post-response aging is fire-and-forget in production. Here it is driven to
        # completion so each printed line reflects state the previous request
        # actually produced, rather than racing it.
        try:
            coro.send(None)
        except StopIteration:
            pass

    app = create_app(
        service,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_stub_upstream(forwarded))),
        schedule=run_now,
    )
    client = TestClient(app)

    print(f"\ncompactor: {label}")
    print("upstream:  stub (records what it was asked, always answers)\n")
    print(f"{'req':>4} {'client sent':>22} {'forwarded':>22} {'saved':>7}  what the proxy added")
    print("-" * 100)

    history: list[dict] = []
    for index, text in enumerate(CONVERSATION):
        history.append({"role": "user", "content": text})
        response = client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-5", "max_tokens": 512, "messages": list(history)},
            headers={"x-api-key": "sk-client-key", SESSION_HEADER: SESSION},
        )
        assert response.status_code == 200, response.text

        sent_turns = len(history)
        sent_tokens = sum(count_tokens(m["content"]) for m in history)
        out = forwarded[-1]["messages"]
        out_turns = len(out)
        out_tokens = sum(count_tokens(m["content"]) for m in out)
        saved = sent_tokens - out_tokens

        kinds = []
        if any(m["content"].startswith("[Memory") for m in out):
            kinds.append("memory")
        if any(m["content"].startswith("[Compacted") for m in out):
            kinds.append("summaries")
        note = ", ".join(kinds) or "verbatim passthrough"

        print(f"{index:>4} {f'{sent_turns:>2} turns / {sent_tokens:>4} tok':>22} "
              f"{f'{out_turns:>2} turns / {out_tokens:>4} tok':>22} {saved:>+7}  {note}")

        # The reply the client received, which a real client resends next time.
        history.append({"role": "assistant", "content": response.json()["content"][0]["text"]})

    print("-" * 100)
    stats = client.get("/stats").json()
    print(f"\n{'':>4} requests={stats['requests']}  gc_applied={stats['gc_applied']}  "
          f"gc_bypassed={stats['gc_bypassed']}")
    print(f"{'':>4} client sent {stats['tokens_from_client']} tok -> forwarded "
          f"{stats['tokens_forwarded']} tok   saved {stats['tokens_saved']:+d}  ({stats['ratio']}x)")

    final = forwarded[-1]["messages"]
    print("\nWHAT THE PROVIDER RECEIVED ON THE LAST REQUEST\n")
    for message in final:
        for line in message["content"].split("\n"):
            print(f"  {message['role']:>9}: {line[:88]}")

    print("\nEvery older exchange is in the front at the level its score earned: summarised,")
    print("archived to the line naming it, or folded out of view once the budget ran out.")
    print("The last turn asks which database was chosen; turn 1 said postgres. At this")
    print("budget that exchange is folded: the model is told those turns exist, not what")
    print("they said - and LLM-GC still holds them word for word.\n")
    # Generations belong to the session, not to the shared GenerationalMemory.
    snapshot = service.session_manager.store.get(SESSION).snapshot
    levels = Counter(span.level.value for span in snapshot.old)
    print(f"  old gen: {len(snapshot.old)} spans {dict(levels)}   permanent gen: {len(snapshot.active_knowledge)} facts\n")

def serve(args: argparse.Namespace) -> None:
    """Serve the proxy offline, and the dashboard beside it if asked."""
    config = serve_config(args)
    service, label = build_engine(config, compactor_kind=args.compactor)
    settings = load_settings(None, port=args.port, dashboard=args.dashboard, dashboard_port=args.dashboard_port,
                             engine=config.model_dump())
    print(f"\ncompactor: {label}")
    print("upstream:  stub (canned replies, no network)")
    print(f"settings:  min_cut_tokens={config.min_cut_tokens}  last_n_turns_to_keep={config.last_n_turns_to_keep}  "
          f"context_window={config.context_window}")
    print(f"\nproxy      http://127.0.0.1:{args.port}   (stats: /stats)")
    if args.dashboard:
        print(f"dashboard  http://127.0.0.1:{args.dashboard_port}")
    print("talk:      python scripts/chat.py\n")
    serve_proxy(service, settings, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_stub_provider)))


_STUB_PARAGRAPHS = [
    "That fits what we settled earlier. I would keep the current approach and revisit it only if the "
    "numbers change, since switching now costs more than it saves.",
    "Worth writing down: it changes the plan for the next step, and it is the kind of detail that is "
    "easy to lose a few turns from now.",
    "I would treat that as settled unless something contradicts it. The open question is still the "
    "rollout order, which depends on how the migration behaves under load.",
]


def _stub_provider(request: httpx2.Request) -> httpx2.Response:
    """An offline stand-in for the provider: canned replies, streamed or not.

    Long enough that a conversation accumulates compactable tokens at a rate worth
    watching. It echoes the start of the question so each reply is distinct.
    """
    payload = json.loads(request.content)
    turns = payload.get("messages", [])
    question = turns[-1]["content"] if turns and isinstance(turns[-1]["content"], str) else "that"
    text = f"(stub reply {len(turns)}) On \"{question[:60]}\": {_STUB_PARAGRAPHS[len(turns) % len(_STUB_PARAGRAPHS)]}"
    usage = {"input_tokens": sum(count_tokens(json.dumps(t)) for t in turns), "output_tokens": count_tokens(text)}
    message = {"id": f"msg_stub_{len(turns)}", "type": "message", "role": "assistant", "model": "stub",
               "stop_reason": "end_turn", "stop_sequence": None, "usage": usage}
    if not payload.get("stream"):
        return httpx2.Response(200, json={**message, "content": [{"type": "text", "text": text}]})

    events = [
        ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        *[("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": word + " "}}) for word in text.split()],
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": usage["output_tokens"]}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx2.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true", help="serve offline instead of running the demo")
    parser.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument("--compactor", choices=("noop", "llm"), default="llm",
                        help="noop = the baseline that expands; llm = LLMCompactor over scripted summaries")
    serving = parser.add_argument_group("serve mode")
    serving.add_argument("--dashboard", action="store_true", help="also serve the live dashboard (localhost only)")
    serving.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT)
    serving.add_argument("--min-cut-tokens", type=int, default=DEFAULT_MIN_CUT_TOKENS,
                         help="tokens a collection must condense (lower it to see collections sooner)")
    serving.add_argument("--keep-last", type=int, default=DEFAULT_LAST_N_TURNS_TO_KEEP,
                         help="newest turns always kept verbatim")
    serving.add_argument("--context-window", type=int, default=DEFAULT_PROXY_CONTEXT_WINDOW,
                         help="the proxied model's window; sets the recap budget")
    args = parser.parse_args()
    # The application chooses how the library's logs are shown: what each request was and
    # why one was passed on untouched. They name sessions and counts, never content.
    configure_logging("INFO", "text")

    if args.serve:
        serve(args)
    else:
        demo(args.compactor)


if __name__ == "__main__":
    main()
