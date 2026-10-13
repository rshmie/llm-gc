"""The client's own session and agent: a label on a conversation, never its identity."""

import pytest

from llm_gc.models import ClientOrigin
from llm_gc.proxy.session_identity import CLIENT_AGENT_HEADER, CLIENT_SESSION_HEADER, client_origin

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
