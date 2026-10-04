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

    def test_key_order_in_provider_content_does_not_count_as_an_edit(self):
        """An SDK may serialise the same block with its keys in another order;
        treating that as an edit would cold-start the conversation for nothing."""
        one = [{"type": "text", "text": "hi"}]
        other = [{"text": "hi", "type": "text"}]
        assert chain_hashes([_turn("hi", raw=one)]) == chain_hashes([_turn("hi", raw=other)])

    def test_non_text_content_is_part_of_the_identity(self):
        """Two turns with the same text but different attached images are not the
        same turn."""
        def image(data: str):
            return [{"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}]
        assert chain_hashes([_turn("what is this?", raw=image("AAA"))]) != \
            chain_hashes([_turn("what is this?", raw=image("BBB"))])


class TestRawContentStaysPrivate:
    def test_it_is_excluded_from_serialisation_and_repr(self):
        secret = [{"type": "image", "source": {"type": "base64", "data": "SECRET-BYTES"}}]
        message = Message(role="user", content="look", raw_content=secret)
        assert "raw_content" not in message.model_dump()
        assert "SECRET-BYTES" not in message.model_dump_json()
        assert "SECRET-BYTES" not in repr(message)
