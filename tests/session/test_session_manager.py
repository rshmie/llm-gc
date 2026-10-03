from datetime import datetime, timedelta, timezone

from llm_gc.models import Message
from llm_gc.session import SessionManager


def _msg(content: str = "x", turn_index: int = 0) -> Message:
    return Message(role="user", content=content, token_count=1, turn_index=turn_index)


class TestGetOrCreate:
    def test_it_returns_the_same_state_object_across_calls(self):
        manager = SessionManager()
        assert manager.get_or_create("s1") is manager.get_or_create("s1")

    def test_it_touches_the_session(self):
        manager = SessionManager()
        state = manager.get_or_create("s1")
        state.last_updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
        manager.get_or_create("s1")
        assert datetime.now(timezone.utc) - state.last_updated_at < timedelta(seconds=5)

    def test_different_sessions_never_share_state(self):
        manager = SessionManager()
        a = manager.get_or_create("a")
        a.snapshot = a.snapshot.with_turn(_msg("only in a"))
        assert manager.get_or_create("b").snapshot.young == ()


class TestLiveness:
    """A writer that began before a close must be able to tell, so a closed
    session stays closed."""

    def test_an_open_session_is_live(self):
        manager = SessionManager()
        assert manager.is_live(manager.get_or_create("s1"))

    def test_a_closed_session_is_not_live(self):
        manager = SessionManager()
        state = manager.get_or_create("s1")
        manager.close_session("s1")
        assert not manager.is_live(state)

    def test_a_reopened_id_is_a_different_session(self):
        """Same id, new conversation: the old state object must not count as live,
        or a late commit would write the old conversation into the new one."""
        manager = SessionManager()
        old = manager.get_or_create("s1")
        manager.close_session("s1")
        manager.get_or_create("s1")
        assert not manager.is_live(old)


class TestCloseSession:
    def test_close_removes_the_session_and_all_its_data(self):
        manager = SessionManager()
        state = manager.get_or_create("s1")
        state.snapshot = state.snapshot.with_turn(_msg("secret"))
        manager.close_session("s1")
        assert manager.store.get("s1") is None
        assert manager.get_or_create("s1").snapshot.young == ()

    def test_closing_an_unknown_session_is_harmless(self):
        SessionManager().close_session("never-opened")


class TestIdleSweeper:
    def test_sweep_closes_only_idle_sessions(self):
        manager = SessionManager(session_timeout_s=1800)
        manager.get_or_create("stale")
        manager.get_or_create("fresh")
        manager.store.get("stale").last_updated_at = datetime.now(timezone.utc) - timedelta(hours=1)

        assert manager.sweep_once() == ["stale"]
        assert manager.store.get("stale") is None
        assert manager.store.get("fresh") is not None

    def test_sweep_closes_nothing_when_all_sessions_are_recent(self):
        manager = SessionManager(session_timeout_s=1800)
        manager.get_or_create("s1")
        assert manager.sweep_once() == []
