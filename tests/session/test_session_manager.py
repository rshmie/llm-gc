from datetime import datetime, timedelta, timezone

from llm_gc.events import Event, EventBus, EventType
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


class TestACloseIsAnnounced:
    """Anything keeping its own copy of a session's data learns of the close from the bus."""

    def _manager(self, **kwargs) -> tuple[SessionManager, list[tuple[str, bool]]]:
        """A manager on a bus, and each close it announced: the session, and whether it was still held."""
        bus = EventBus()
        manager = SessionManager(event_bus=bus, **kwargs)
        heard: list[tuple[str, bool]] = []

        def note(event: Event) -> None:
            session_id = event.data["session_id"]
            heard.append((session_id, manager.store.get(session_id) is not None))

        bus.subscribe(EventType.SESSION_CLOSED, note)
        return manager, heard

    def test_a_close_is_announced_once_after_the_session_is_gone(self):
        manager, heard = self._manager()
        manager.get_or_create("s1")
        manager.close_session("s1")
        manager.close_session("s1")
        assert heard == [("s1", False)]

    def test_closing_a_session_never_opened_announces_nothing(self):
        manager, heard = self._manager()
        manager.close_session("never-opened")
        assert heard == []

    def test_an_idle_close_and_a_close_past_the_cap_are_both_announced(self):
        manager, heard = self._manager(session_timeout_s=1800, max_sessions=1)
        manager.get_or_create("first")
        manager.get_or_create("second")
        manager.store.get("second").last_updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
        manager.sweep_once()
        assert heard == [("first", False), ("second", False)]


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


class TestABreakIsNotTheEndOfAConversation:
    def test_by_default_a_session_idle_through_a_long_break_is_kept(self):
        """Live run, 2026-10-17: a 30-minute limit closed a conversation its person was still in."""
        manager = SessionManager()
        manager.get_or_create("s")
        manager.store.get("s").last_updated_at = datetime.now(timezone.utc) - timedelta(hours=3)

        assert manager.sweep_once() == []

    def test_past_the_cap_the_least_recently_active_is_closed(self):
        manager = SessionManager(max_sessions=2)
        manager.get_or_create("oldest")
        manager.get_or_create("newer")
        manager.store.get("oldest").last_updated_at = datetime.now(timezone.utc) - timedelta(minutes=10)

        manager.get_or_create("newest")

        assert sorted(manager.store.active_session_ids()) == ["newer", "newest"]

    def test_returning_to_a_held_session_closes_nothing(self):
        manager = SessionManager(max_sessions=2)
        manager.get_or_create("a")
        manager.get_or_create("b")

        manager.get_or_create("a")

        assert sorted(manager.store.active_session_ids()) == ["a", "b"]
