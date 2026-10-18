import pytest
from llm_gc.models.message import (CLIENT_SPEAKER, TOOL_SPEAKER, Message, TextPart, parts_by_writer, person_text,
                                   said_by_user, speaker_label)
from tests.models.support import COMMAND, NOTICE, SKILL_RULES, turn_with_client_text
from pydantic import ValidationError

def test_invalid_role_raises_error():
    with pytest.raises(ValidationError):
        Message(role="doctor", content="hi")

def test_valid_message_creation():
    msg = Message(role="user", content="Hello", token_count=5)
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.token_count == 5

def test_default_token_count():
    msg = Message(role="assistant", content="Hi")
    assert msg.token_count == 0

def test_preview_content():
    msg = Message(role="user", content="Hello, how are you?")
    assert msg.preview() == "Hello, how are you?"


def test_preview_content_with_truncation():
    msg = Message(role="system", content="This is a test message for previewing." * 10, token_count=100)
    assert len(msg.preview(max_length = 50)) == 53
    assert msg.preview().endswith("...")

class TestTextTheClientWrote:
    """A user turn can hold text the client wrote; the person's words are the rest of it."""

    TEXT = "<command-name>/audit</command-name>\n\nKeep at most 5 rows.\n\nAnd use Postgres."
    EXPANSION = (37, 57)

    def test_the_turn_reads_in_order_as_the_persons_part_and_the_clients(self):
        turn = Message(role="user", content=self.TEXT, client_written=(self.EXPANSION,))

        assert parts_by_writer(turn) == [TextPart(False, "<command-name>/audit</command-name>"),
                                         TextPart(True, "Keep at most 5 rows."), TextPart(False, "And use Postgres.")]
        assert person_text(turn) == "<command-name>/audit</command-name>\n\nAnd use Postgres."

    def test_a_turn_the_client_wrote_none_of_is_one_part_untouched(self):
        turn = Message(role="user", content="  Use Postgres.\n")

        assert parts_by_writer(turn) == [TextPart(False, "  Use Postgres.\n")]
        assert person_text(turn) == "  Use Postgres.\n"

    def test_a_turn_the_client_wrote_all_of_has_no_words_of_the_persons(self):
        turn = Message(role="user", content="<task-notification>done</task-notification>", client_written=((0, 43),))
        assert person_text(turn) == ""

    @pytest.mark.parametrize("ranges", [((0, 999),), ((5, 3),), ((10, 20), (15, 30)), ((20, 30), (0, 10))])
    def test_a_range_outside_the_text_or_out_of_order_is_refused(self, ranges):
        with pytest.raises(ValidationError):
            Message(role="user", content=self.TEXT, client_written=ranges)

    def test_only_a_user_turn_can_hold_it(self):
        with pytest.raises(ValidationError):
            Message(role="assistant", content=self.TEXT, client_written=(self.EXPANSION,))


class TestWhoATurnWithClientTextBelongsTo:
    def test_a_turn_the_person_wrote_part_of_is_still_theirs(self):
        turn = turn_with_client_text(0, (False, COMMAND), (True, SKILL_RULES))
        assert said_by_user(turn) and speaker_label(turn) == "user"

    def test_a_turn_the_client_wrote_all_of_is_the_clients(self):
        turn = turn_with_client_text(0, (True, NOTICE))
        assert not said_by_user(turn) and speaker_label(turn) == CLIENT_SPEAKER


class TestATurnOfToolResultsIsNamedAsOne:
    RESULT = {"type": "tool_result", "tool_use_id": "t1", "content": "42 passed"}

    def test_a_turn_holding_only_tool_results_is_the_tools(self):
        turn = Message(role="user", content="[tool result] 42 passed", raw_content=[self.RESULT])
        assert speaker_label(turn) == TOOL_SPEAKER

    def test_a_turn_that_also_holds_text_keeps_the_users_label(self):
        """What a person types after interrupting a tool round arrives beside the round's results."""
        typed = {"type": "text", "text": "Why do we keep the mypy override?"}
        turn = Message(role="user", content="[tool result] 42 passed\nWhy do we keep the mypy override?",
                       raw_content=[self.RESULT, typed])
        assert speaker_label(turn) == "user"
