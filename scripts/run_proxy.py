#!/usr/bin/env python3
"""Run the proxy, or drive a conversation through it and watch what it forwards.

Two modes.

**Demo (default, no API key, no network):**

    python scripts/run_proxy.py

Starts the real proxy app against a stub upstream, replays a conversation through
real HTTP, and prints what the client sent versus what was forwarded, per request.
This is the proxy's equivalent of `trace_session.py`: the engine's own trace shows
the generations moving, this shows the wire.

**Serve (real traffic to a real provider), with the live dashboard:**

    export ANTHROPIC_API_KEY=...      # for LLM-GC's own compaction calls
    python scripts/run_proxy.py --serve --dashboard

Then talk through it (`python scripts/chat.py`) or point any client at it and
change nothing else:

    client = anthropic.Anthropic(base_url="http://127.0.0.1:9900", api_key="sk-...")

and watch http://127.0.0.1:9901 while you do. Add `--stub-upstream` to run the
whole loop offline, with canned replies instead of a model.

The client's key is forwarded from its own request and never stored. The key in
the environment is a *separate* credential, used only for LLM-GC's compaction
calls — see doc/design/llm-provider.md for why those must not be the same key.
The dashboard shows conversation content, so it is off unless asked for and only
ever listens on localhost.

Developer tooling, not library code, which is why it lives in `scripts/`.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from llm_gc.config import GCConfig  # noqa: E402
from llm_gc.config.constants import (  # noqa: E402
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_LAST_N_TURNS_TO_KEEP,
    DEFAULT_MIN_CUT_TOKENS,
)
from llm_gc.engine.compaction import LLMCompactor, NoOpCompactor  # noqa: E402
from llm_gc.engine.gc_service import GcService  # noqa: E402
from llm_gc.engine.generations import GenerationalMemory  # noqa: E402
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy  # noqa: E402
from llm_gc.events import EventBus  # noqa: E402
from llm_gc.extraction import KnowledgeExtractor  # noqa: E402
from llm_gc.llm import AnthropicClient, FakeProvider, LLMResponse, StopReason  # noqa: E402
from llm_gc.monitoring import ContextHealthMonitor  # noqa: E402
from llm_gc.proxy import DEFAULT_PROXY_PORT, SESSION_HEADER, create_app  # noqa: E402
from llm_gc.scoring import (  # noqa: E402
    DecisionScorer,
    DensityScorer,
    RecencyScorer,
    ReferenceScorer,
    RelevanceScorer,
    RelevanceType,
)
from llm_gc.session import SessionManager  # noqa: E402
from llm_gc.utils import count_tokens  # noqa: E402
from llm_gc.visualizer import create_dashboard_app  # noqa: E402

SESSION = "demo"

# A conversation with facts worth remembering and filler worth forgetting, so the
# three generations all have something to do. The last turn deliberately asks about
# something stated near the start, which is the question the memory block exists to
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
    """Scaled so a twelve-turn scripted conversation is cut more than once."""
    return GCConfig(
        context_window=400,
        gc_threshold=0.5,
        keep_threshold=0.55,
        archive_threshold=0.25,
        min_compactable_tokens=10,
        last_n_turns_to_keep=3,
        # Scaled to this demo's small window; real conversations use far larger batches.
        min_cut_tokens=60,
        max_memory_ratio=0.15,
    )


def serve_config(args: argparse.Namespace) -> GCConfig:
    """The library defaults, with the two knobs worth turning while watching."""
    return GCConfig(context_window=args.context_window, min_cut_tokens=args.min_cut_tokens,
                    last_n_turns_to_keep=args.keep_last)


def build_engine(config: GCConfig, *, api_key: str | None, compactor_kind: str) -> tuple[GcService, GenerationalMemory, str]:
    """The production engine graph. Only the compactor varies."""
    bus = EventBus()

    if compactor_kind == "noop":
        compactor = NoOpCompactor(event_bus=bus)
        label = "NoOpCompactor (baseline: wraps the run, does not summarise)"
    elif api_key:
        compactor = LLMCompactor(event_bus=bus, provider=AnthropicClient(api_key=api_key))
        label = "LLMCompactor -> AnthropicClient (live compaction calls)"
    else:
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
        compactor = LLMCompactor(event_bus=bus, provider=provider)
        label = "LLMCompactor -> scripted provider (no ANTHROPIC_API_KEY; wording canned, arithmetic real)"

    memory = GenerationalMemory(
        knowledge_extractor=KnowledgeExtractor(event_bus=bus),
        compactor=compactor,
    )
    service = GcService(
        session_manager=SessionManager(),
        generational_memory=memory,
        relevance_scorer=RelevanceScorer(
            scorers=[RecencyScorer(decay_rate=0.15), DensityScorer(), DecisionScorer(), ReferenceScorer()],
            weights={
                RelevanceType.RECENCY: 0.6,
                RelevanceType.DENSITY: 0.2,
                RelevanceType.DECISION: 0.1,
                RelevanceType.REFERENCE: 0.1,
            },
            event_bus=bus,
        ),
        sweeper=Sweeper(
            sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=bus
        ),
        event_bus=bus,
        gc_config=config,
    )
    return service, memory, label


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


def demo(compactor_kind: str, api_key: str | None) -> None:
    """Replay a conversation through the real proxy over real HTTP."""
    service, memory, label = build_engine(demo_config(), api_key=api_key, compactor_kind=compactor_kind)
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
    print(f"{'':>4} client history {stats['tokens_in_client_history']} tok -> forwarded "
          f"{stats['tokens_forwarded']} tok   saved {stats['tokens_saved']:+d}  ({stats['ratio']}x)")

    final = forwarded[-1]["messages"]
    print("\nWHAT THE PROVIDER RECEIVED ON THE LAST REQUEST\n")
    for message in final:
        for line in message["content"].split("\n"):
            print(f"  {message['role']:>9}: {line[:88]}")

    print("\nThe last turn asks which database was chosen. Turn 1 said postgres and was")
    print("archived long before this request. Look for it in the memory block above:")
    print("that is the whole claim, arriving over HTTP.\n")
    # Generations belong to the session, not to the shared GenerationalMemory.
    snapshot = service.session_manager.store.get(SESSION).snapshot
    print(f"  old gen: {len(snapshot.old)} summaries   "
          f"permanent gen: {len(snapshot.active_knowledge)} facts\n")


def serve(args: argparse.Namespace, api_key: str | None) -> None:
    """Run the proxy for real traffic, and the dashboard beside it if asked."""
    import uvicorn

    config = serve_config(args)
    service, _, label = build_engine(config, api_key=api_key, compactor_kind=args.compactor)
    stub = httpx2.AsyncClient(transport=httpx2.MockTransport(_stub_provider)) if args.stub_upstream else None
    proxy_app = create_app(service, http_client=stub)
    apps = [(proxy_app, args.port)]
    if args.dashboard:
        store = service.session_manager.store
        monitor = ContextHealthMonitor(
            event_bus=service.event_bus, gc_config=config,
            read_generations=lambda session_id: (state.snapshot if session_id is not None
                                                 and (state := store.get(session_id)) is not None else None))
        apps.append((create_dashboard_app(service, monitor, read_proxy_stats=proxy_app.state.metrics.as_dict),
                     args.dashboard_port))

    print(f"\ncompactor: {label}")
    if not api_key and args.compactor != "noop":
        print("WARNING: no ANTHROPIC_API_KEY, so compaction summaries are canned. "
              "Set it for real summarisation.")
    print(f"upstream:  {'stub (canned replies, no network)' if args.stub_upstream else 'https://api.anthropic.com'}")
    print(f"settings:  min_cut_tokens={config.min_cut_tokens}  last_n_turns_to_keep={config.last_n_turns_to_keep}  "
          f"context_window={config.context_window}")
    print(f"\nproxy      http://127.0.0.1:{args.port}   (stats: /stats)")
    if args.dashboard:
        print(f"dashboard  http://127.0.0.1:{args.dashboard_port}")
    print("talk:      python scripts/chat.py\n")
    # Localhost only, no auth. Binding wider needs authentication first
    # (CLAUDE.md §10): the client's provider key passes through this process, and
    # the dashboard shows every conversation it holds.
    servers = [uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info"))
               for app, port in apps]

    async def run_all() -> None:
        # One event loop for both, so the dashboard reads sessions from the loop
        # that writes them.
        await asyncio.gather(*(server.serve() for server in servers))

    asyncio.run(run_all())


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
    parser.add_argument("--serve", action="store_true", help="run a real server instead of the demo")
    parser.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument("--compactor", choices=("noop", "llm"), default="llm",
                        help="noop = the baseline that expands; llm = LLMCompactor")
    serving = parser.add_argument_group("serve mode")
    serving.add_argument("--dashboard", action="store_true", help="also serve the live dashboard (localhost only)")
    serving.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT)
    serving.add_argument("--stub-upstream", action="store_true", help="answer with canned replies; no network")
    serving.add_argument("--min-cut-tokens", type=int, default=DEFAULT_MIN_CUT_TOKENS,
                         help="compactable tokens a cut needs (lower it to see cuts sooner)")
    serving.add_argument("--keep-last", type=int, default=DEFAULT_LAST_N_TURNS_TO_KEEP,
                         help="newest turns always kept verbatim")
    serving.add_argument("--context-window", type=int, default=200_000,
                         help="the proxied model's window; sets the memory budget")
    args = parser.parse_args()

    # From the environment, never a flag, so a key never lands in shell history.
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if args.serve:
        serve(args, api_key)
    else:
        demo(args.compactor, api_key)


if __name__ == "__main__":
    main()
