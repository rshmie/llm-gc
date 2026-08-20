from llm_gc.models import Message
from llm_gc.session import SessionState, SessionStore


class TestSessionState:
    def test_new_state_starts_empty_and_timestamped(self):
        state = SessionState(session_id="s1")
        assert state.session_id == "s1"
        assert state.messages == []
        assert state.created_at is not None
        assert state.last_updated_at is not None

    def test_touch_advances_last_updated_at_without_moving_created_at(self):
        state = SessionState(session_id="s1")
        created = state.created_at
        first_touch = state.last_updated_at
        state.touch()
        assert state.created_at == created
        assert state.last_updated_at >= first_touch


class TestSessionStore:
    def test_get_or_create_makes_a_session_on_first_sight(self):
        store = SessionStore()
        state = store.get_or_create("s1")
        assert state.session_id == "s1"
        assert store.get("s1") is state

    def test_get_or_create_returns_the_same_object_on_repeat(self):
        store = SessionStore()
        first = store.get_or_create("s1")
        first.messages.append(Message(role="user", content="hi"))
        second = store.get_or_create("s1")
        # Same object, so state written on the first call is still there.
        assert second is first
        assert len(second.messages) == 1

    def test_get_returns_none_for_unknown_session(self):
        assert SessionStore().get("missing") is None

    def test_remove_drops_the_session_and_returns_it(self):
        store = SessionStore()
        store.get_or_create("s1")
        removed = store.remove("s1")
        assert removed is not None
        assert removed.session_id == "s1"
        assert store.get("s1") is None

    def test_remove_of_unknown_session_is_a_harmless_none(self):
        assert SessionStore().remove("never-existed") is None

    def test_count_reflects_active_sessions(self):
        store = SessionStore()
        assert store.count() == 0
        store.get_or_create("s1")
        store.get_or_create("s2")
        assert store.count() == 2
        store.remove("s1")
        assert store.count() == 1

    def test_active_session_ids_is_a_snapshot_copy(self):
        store = SessionStore()
        store.get_or_create("s1")
        store.get_or_create("s2")
        ids = store.active_session_ids()
        assert sorted(ids) == ["s1", "s2"]
        # Mutating the store must not change the already-returned list.
        store.remove("s1")
        assert sorted(ids) == ["s1", "s2"]
