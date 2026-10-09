"""The port probe `llm-gc start` runs before serving, against real sockets on localhost."""

import socket

from llm_gc.cli.serve import BIND_HOST, find_busy_port


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
