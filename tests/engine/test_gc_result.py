"""A pass's result and its report: the same measures, and only the result holds the conversation."""

import pytest
from pydantic import ValidationError

from llm_gc.engine import AgingDegradation, CutOutcome, DegradationReason, GCResult, GCStatus
from llm_gc.models import Message


def _result(tail: list[Message] | None = None) -> GCResult:
    front = Message(role="user", content="Earlier: the vault code is 4417.", token_count=9, turn_index=-1)
    if tail is None:
        tail = [
            Message(role="user", content="Ship it on Friday.", token_count=6, turn_index=4),
            Message(role="assistant", content="Friday it is.", token_count=4, turn_index=5),
        ]
    return GCResult(
        final_messages=[front, *tail],
        gc_run_id="run-1",
        status=GCStatus.COMPLETED,
        tokens_before=40,
        tokens_in_final=19,
        kept_count=0,
        compacted_count=2,
        archived_count=2,
        compacted_tokens=12,
        archived_tokens=9,
        duration_ms=3.0,
        degradations=[
            AgingDegradation(reason=DegradationReason.ARCHIVE_FAILED, turn_indices=[2], detail="LLMResponseError")
        ],
        outcome=CutOutcome(compacted=[0, 1], archived=[2, 3]),
    )


class TestTheReport:
    def test_it_carries_every_measure_of_the_result(self):
        """Compared by value, so a measure declared on GCResult alone, instead of on the
        shared base, fails here rather than silently never reaching a subscriber."""
        result = _result()
        report = result.report()

        assert report.model_dump(exclude={"tail_turns", "tail_tokens"}) == result.model_dump(exclude={"final_messages"})
        assert report.tokens_saved == result.tokens_saved

    def test_it_describes_the_tail_by_turns_and_tokens_and_leaves_the_front_out(self):
        report = _result().report()

        assert report.tail_turns == [4, 5]
        assert report.tail_tokens == 10

    def test_a_context_of_only_the_front_has_an_empty_tail(self):
        report = _result(tail=[]).report()

        assert report.tail_turns == []
        assert report.tail_tokens == 0

    def test_it_holds_none_of_the_conversations_words(self):
        payload = _result().report().model_dump_json()

        assert "vault code" not in payload, "the front's text"
        assert "Friday" not in payload, "the tail's text"

    def test_one_subscriber_cannot_change_what_the_next_one_reads(self):
        report = _result().report()

        with pytest.raises(ValidationError):
            report.tail_tokens = 0
