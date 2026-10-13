"""Run the proxy, and the dashboard beside it, on one event loop. Application code."""

import asyncio
import socket
from collections.abc import Iterable

import httpx2
import uvicorn

from llm_gc.cli.settings import ProxySettings, create_model_window_lookup
from llm_gc.engine import GcService
from llm_gc.monitoring import ContextHealthMonitor
from llm_gc.proxy import create_app
from llm_gc.visualizer import TranscriptRecorder, create_dashboard_app


BIND_HOST = "127.0.0.1"


def find_busy_port(ports: Iterable[int]) -> int | None:
    """The first of `ports` that cannot be bound on `BIND_HOST`, or None when all are free.

    It is checked before startup: uvicorn reports a taken port only once serving has begun, as a
    bare OS error after the banner has already promised the address. The probe binds the
    way uvicorn does, with `SO_REUSEADDR` - so a port left in TIME_WAIT by a stopped proxy
    is not mistaken for one in use.
    """
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((BIND_HOST, port))
            except OSError:
                return port
    return None


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
                           on_reply=transcripts.record_reply if transcripts is not None else None,
                           window_lookup=create_model_window_lookup(settings, http_client=http_client),
                           collect_subagents=settings.collect_subagents)
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
    # so it cannot be kept for the proxy alone. Its startup lines are dropped too, below
    # DEBUG: the banner already names each address, and two servers print every line twice.
    uvicorn_level = "debug" if settings.log_level == "DEBUG" else "warning"
    servers = [uvicorn.Server(uvicorn.Config(app, host=BIND_HOST, port=port, log_level=uvicorn_level,
                                             access_log=False))
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
