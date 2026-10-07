"""Run the proxy, and the dashboard beside it, on one event loop. Application code."""

import asyncio

import httpx2
import uvicorn

from llm_gc.cli.settings import ProxySettings
from llm_gc.engine import GcService
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.proxy import create_app
from llm_gc.visualizer import TranscriptRecorder, create_dashboard_app

# Localhost only, with no authentication. Binding wider needs authentication first
# (CLAUDE.md §10): the client's provider key passes through this process, and the
# dashboard shows every conversation it holds.
BIND_HOST = "127.0.0.1"


def serve_proxy(service: GcService, settings: ProxySettings, *, http_client: httpx2.AsyncClient | None = None) -> None:
    """Serve until interrupted: the proxy always, the dashboard when the settings ask for it.

    Args:
        service: The wired engine both apps read from.
        settings: Ports, upstream and whether the dashboard runs.
        http_client: Replaces the client that forwards to the upstream, for an offline run.
    """
    # Transcripts are a second copy of every conversation, so they exist only for the dashboard.
    transcripts = TranscriptRecorder() if settings.dashboard else None
    proxy_app = create_app(service, upstream_base_url=settings.upstream, http_client=http_client,
                           on_client_history=transcripts.record if transcripts is not None else None,
                           on_reply=transcripts.record_reply if transcripts is not None else None)
    apps = [(proxy_app, settings.port)]
    if transcripts is not None:
        store = service.session_manager.store
        monitor = ContextHealthMonitor(
            event_bus=service.event_bus, gc_config=settings.engine,
            read_generations=lambda session_id: (state.snapshot if session_id is not None
                                                 and (state := store.get(session_id)) is not None else None))
        apps.append((create_dashboard_app(service, monitor, read_proxy_stats=proxy_app.state.metrics.as_dict,
                                          transcripts=transcripts), settings.dashboard_port))

    # No access log: the dashboard polls, and a line per poll buried the proxy's own lines,
    # which already record each request. uvicorn's access logger is shared by the process,
    # so it cannot be kept for the proxy alone.
    servers = [uvicorn.Server(uvicorn.Config(app, host=BIND_HOST, port=port, log_level="info", access_log=False))
               for app, port in apps]

    async def run_all() -> None:
        # One event loop for both, so the dashboard reads sessions from the loop that writes them.
        await asyncio.gather(*(server.serve() for server in servers))

    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        # Each server has already shut down cleanly: uvicorn re-raises the Ctrl-C after its
        # shutdown so the program running it sees the signal too.
        print("\nLLM-GC proxy stopped.")
