"""Run the context-health dashboard: python -m llm_gc.visualizer"""

import logging

import uvicorn

from llm_gc.config.constants import DEFAULT_DASHBOARD_PORT
from llm_gc.visualizer.server import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Bound to localhost on purpose: the scaffold serves one local demo session
    # and has no auth - it must never listen on an external interface.
    uvicorn.run(create_app(), host="127.0.0.1", port=DEFAULT_DASHBOARD_PORT)


if __name__ == "__main__":
    main()
