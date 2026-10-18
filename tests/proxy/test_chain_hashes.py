"""Conversation lineage: a hash per turn that identifies everything up to it."""

from hypothesis import given
from hypothesis import strategies as st

from llm_gc.models import Message
from llm_gc.proxy.session_identity import chain_hashes


def _turn(text: str, role: str = "user", raw=None) -> Message:
    return Message(role=role, content=text, raw_content=raw)


_texts = st.lists(st.text(max_size=20), max_size=8)


class TestThePrefixProperty:
    @given(head=_texts, tail=_texts)
    def test_a_conversation_and_its_continuation_agree_on_the_shared_part(self, head, tail):
        """What session lookup relies on: request N's history is a prefix of
        request N+1's, so N's last hash appears in N+1's list."""
        short = chain_hashes([_turn(t) for t in head])
        long = chain_hashes([_turn(t) for t in head + tail])
        assert long[: len(short)] == short

    def test_an_edit_changes_every_hash_from_that_turn_on(self):
        """Hashing turns independently would let two conversations that merely
        share one turn collide on it."""
        original = chain_hashes([_turn("a"), _turn("b"), _turn("c"), _turn("d")])
        edited = chain_hashes([_turn("a"), _turn("B"), _turn("c"), _turn("d")])
        assert original[0] == edited[0]
        assert all(o != e for o, e in zip(original[1:], edited[1:]))


class TestWhatCountsAsTheSameTurn:
    def test_moving_text_across_a_turn_boundary_is_a_different_conversation(self):
        assert chain_hashes([_turn("ab"), _turn("c")])[-1] != chain_hashes([_turn("a"), _turn("bc")])[-1]

    def test_the_role_is_part_of_the_turn(self):
        assert chain_hashes([_turn("hi", "user")]) != chain_hashes([_turn("hi", "assistant")])

    def test_a_cache_marker_moved_to_a_newer_turn_is_not_an_edit(self):
        """Claude Code marks its newest turns for caching and moves the markers on with
        every request; counted as an edit, every request would start a new conversation."""
        marked = [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        unmarked = [{"type": "text", "text": "hi"}]
        nested = [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "text", "text": "ok", "cache_control": {"type": "ephemeral"}}],
            }
        ]
        plain_nested = [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "ok"}]}]

        assert chain_hashes([_turn("hi", "assistant", marked)]) == chain_hashes([_turn("hi", "assistant", unmarked)])
        assert chain_hashes([_turn("ok", raw=nested)]) == chain_hashes([_turn("ok", raw=plain_nested)])

    def test_a_plain_string_is_the_same_turn_as_its_one_text_block(self):
        assert chain_hashes([_turn("hi", raw="hi")]) == chain_hashes(
            [_turn("hi", raw=[{"type": "text", "text": "hi"}])]
        )

    def test_changing_what_a_turn_says_is_still_an_edit(self):
        marked = [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]
        assert chain_hashes([_turn("hi", raw=marked)]) != chain_hashes(
            [_turn("ho", raw=[{"type": "text", "text": "ho"}])]
        )

    def test_key_order_in_provider_content_does_not_count_as_an_edit(self):
        """An SDK may serialise the same block with its keys in another order;
        treating that as an edit would cold-start the conversation for nothing."""
        one = [{"type": "text", "text": "hi"}]
        other = [{"text": "hi", "type": "text"}]
        assert chain_hashes([_turn("hi", raw=one)]) == chain_hashes([_turn("hi", raw=other)])

    def test_instructions_rewritten_inside_an_earlier_turn_are_not_an_edit(self):
        """Claude Code rewrites the <system-reminder> elements in earlier turns between
        requests; counted as an edit, one conversation was seeded as a second session."""
        small = "<system-reminder>date</system-reminder>"
        large = "<system-reminder>CLAUDE.md, memory, date</system-reminder>"
        said = {"type": "text", "text": "Help me design a URL shortener."}
        before = [{"type": "text", "text": small}, said]
        after = [{"type": "text", "text": large}, said]
        inline = [{"type": "text", "text": f"{large}\n\nHelp me design a URL shortener."}]
        result = [{"type": "tool_result", "tool_use_id": "t1", "content": f"compiled\n{small}"}]
        bare_result = [{"type": "tool_result", "tool_use_id": "t1", "content": "compiled"}]

        assert chain_hashes([_turn("x", raw=before)]) == chain_hashes([_turn("x", raw=after)])
        assert chain_hashes([_turn("x", raw=before)]) == chain_hashes([_turn("x", raw=inline)])
        assert chain_hashes([_turn("x", raw=result)]) == chain_hashes([_turn("x", raw=bare_result)])

    def test_an_assistant_quoting_the_element_has_still_said_it(self):
        quoted = "<system-reminder>a</system-reminder> is how Claude Code marks its notes"
        other = "<system-reminder>b</system-reminder> is how Claude Code marks its notes"
        assert chain_hashes([_turn("x", "assistant", quoted)]) != chain_hashes([_turn("x", "assistant", other)])

    def test_thinking_removed_from_an_earlier_reply_is_not_an_edit(self):
        """Live run: Claude Code removed earlier thinking when the conversation moved to another model, and
        the conversation was started over, losing the collection already applied."""
        said = {"type": "text", "text": "Postgres, partitioned by day."}
        thought = [{"type": "thinking", "thinking": "", "signature": "EqQB" + "x" * 300}, said]
        redacted = [{"type": "redacted_thinking", "data": "EmwK" + "y" * 300}, said]

        assert chain_hashes([_turn("Postgres.", "assistant", thought)]) == chain_hashes(
            [_turn("Postgres.", "assistant", [said])]
        )
        assert chain_hashes([_turn("Postgres.", "assistant", redacted)]) == chain_hashes(
            [_turn("Postgres.", "assistant", [said])]
        )

    def test_non_text_content_is_part_of_the_identity(self):
        """Two turns with the same text but different attached images are not the
        same turn."""

        def image(data: str):
            return [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            ]

        assert chain_hashes([_turn("what is this?", raw=image("AAA"))]) != chain_hashes(
            [_turn("what is this?", raw=image("BBB"))]
        )


class TestRawContentStaysPrivate:
    def test_it_is_excluded_from_serialisation_and_repr(self):
        secret = [{"type": "image", "source": {"type": "base64", "data": "SECRET-BYTES"}}]
        message = Message(role="user", content="look", raw_content=secret)
        assert "raw_content" not in message.model_dump()
        assert "SECRET-BYTES" not in message.model_dump_json()
        assert "SECRET-BYTES" not in repr(message)
