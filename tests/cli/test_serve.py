"""How `llm-gc start` serves: the port probe it runs first, against real sockets on localhost, and its servers' log."""

import logging
import socket

from fastapi import FastAPI

from llm_gc.cli.serve import BIND_HOST, create_server, find_busy_port


def _listening_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((BIND_HOST, 0))
    sock.listen()
    return sock


def test_a_port_something_listens_on_is_reported():
    with _listening_socket() as taken:
        port = taken.getsockname()[1]
        assert find_busy_port([port]) == port


def test_free_ports_report_nothing():
    with _listening_socket() as sock:
        port = sock.getsockname()[1]
    # Closed: free again, whatever TIME_WAIT the close left behind.
    assert find_busy_port([port]) is None


def test_the_first_busy_port_in_order_is_the_one_named():
    with _listening_socket() as first, _listening_socket() as second:
        ports = [first.getsockname()[1], second.getsockname()[1]]
        assert find_busy_port(ports) == ports[0]


def test_uvicorns_own_lines_are_written_by_the_applications_handler(caplog):
    """Its warnings and errors, a route's traceback among them, must come out in the format the settings chose."""
    create_server(FastAPI(), 0, "INFO")

    with caplog.at_level(logging.WARNING):
        logging.getLogger("uvicorn.error").warning("Invalid HTTP request received.")

    assert "Invalid HTTP request received." in caplog.text, "it reaches the root logger's handlers"
