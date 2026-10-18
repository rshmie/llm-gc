"""The client's own session and agent: a label on a conversation, never its identity."""

import logging

import pytest

from llm_gc.models import ClientOrigin
from llm_gc.proxy.session_identity import (CLIENT_AGENT_HEADER, CLIENT_SESSION_HEADER, SESSION_HEADER, client_origin,
                                           named_session)

SESSION = "c7a78c2f-d25c-44e4-af6f-dba30717fb9b"
AGENT = "a1c50dab6bf4df2c1"


class TestReadingTheClientsSession:
    def test_a_main_conversation_names_its_session_and_no_agent(self):
        assert client_origin({CLIENT_SESSION_HEADER: SESSION}) == ClientOrigin(session_id=SESSION)

    def test_a_sub_agent_names_both(self):
        origin = client_origin({CLIENT_SESSION_HEADER: SESSION, CLIENT_AGENT_HEADER: AGENT})
        assert origin == ClientOrigin(session_id=SESSION, agent_id=AGENT)

    def test_no_session_names_nothing(self):
        assert client_origin({}) is None

    def test_an_agent_alone_names_nothing(self):
        # Conversations are grouped by session; an agent id alone has nothing to group under.
        assert client_origin({CLIENT_AGENT_HEADER: AGENT}) is None

    @pytest.mark.parametrize("value", ["", "   ", "x" * 65, "abc\nINFO forged line", "two words", "<b>id</b>"])
    def test_a_value_that_is_not_a_plain_id_is_ignored(self, value):
        """The id reaches the log and the page, so a forged log line or markup must not."""
        assert client_origin({CLIENT_SESSION_HEADER: value}) is None

    def test_a_malformed_agent_still_keeps_the_session(self):
        origin = client_origin({CLIENT_SESSION_HEADER: SESSION, CLIENT_AGENT_HEADER: "not an id"})
        assert origin == ClientOrigin(session_id=SESSION)


class TestReadingTheSessionAClientNames:
    """LLM-GC's own header names the session, under the same rule as the client's ids."""

    def test_a_plain_id_names_the_session(self):
        assert named_session({SESSION_HEADER: " chat-42 "}) == "chat-42"

    def test_no_header_names_none_and_says_nothing(self, caplog):
        caplog.set_level(logging.WARNING, logger="llm_gc.proxy.session_identity")
        assert named_session({}) is None
        assert caplog.records == []

    @pytest.mark.parametrize("value", ["", "x" * 65, "chats/42", "two words", "abc\nINFO forged line"])
    def test_any_other_value_is_ignored_with_a_warning_that_does_not_repeat_it(self, value, caplog):
        caplog.set_level(logging.WARNING, logger="llm_gc.proxy.session_identity")
        assert named_session({SESSION_HEADER: value}) is None
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert record.args == (SESSION_HEADER,) and record.header_length == len(value)  # type: ignore[attr-defined]
