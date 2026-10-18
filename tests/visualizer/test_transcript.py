"""The dashboard's copy of each conversation, as the client last sent it."""

import uuid

from llm_gc.models import Message
from llm_gc.utils import count_tokens
from llm_gc.visualizer import TranscriptRecorder
from tests.models.support import turn_with_typed_words


class TestRecording:
    def test_a_history_recorded_again_does_not_have_its_instructions_counted_again(self, tokenised):
        """The client sends the whole conversation on every request, and the proxy records it
        each time, so these counts would be repeated for every turn of every request."""
        reminder = f"{uuid.uuid4().hex} the date is today"
        turn = Message(role="user", content="a question", token_count=3, turn_index=0, instructions=(reminder,))
        recorder = TranscriptRecorder()

        recorder.record("s", [turn])
        recorder.record("s", [turn])

        assert tokenised == [reminder]
        recorded = recorder.get("s")
        assert recorded is not None and recorded[0].instruction_tokens == count_tokens(reminder)

    def test_a_persons_words_beside_tool_results_are_theirs_alone(self):
        """The session trace names a turn by them, and must not name it by a tool's output."""
        recorder = TranscriptRecorder()

        recorder.record("s", [turn_with_typed_words(2, "t1", "Stop, use MySQL.", output="42 passed")])

        recorded = recorder.get("s")
        assert recorded is not None and recorded[0].person_text == "Stop, use MySQL."
