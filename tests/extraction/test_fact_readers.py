"""Reading a collection's facts: by a model, or by patterns when there is none.

The model's answer is untrusted output. What matters most is what it must never
do: file a fact against a turn it was not shown, keep a sentence as a topic, or
lose the topics already on record that let a change of plan replace the old value.
"""

import asyncio
import json

import pytest

from llm_gc.exceptions import LLMResponseError, LLMTimeoutError
from llm_gc.extraction import LLMFactReader, PatternFactReader
from llm_gc.llm import FakeProvider, LLMResponse, StopReason
from llm_gc.llm.prompts import FACTS_CLOSE, FACTS_OPEN, FACTS_PROMPT, TRANSCRIPT_CLOSE, TRANSCRIPT_OPEN, load_prompt
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import file_knowledge
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType


def _said(turn: int, role: str, text: str, tokens: int = 20) -> Message:
    return Message(role=role, content=text, token_count=tokens, turn_index=turn)


def _answer(facts: list[dict] | str, stop: StopReason = StopReason.COMPLETE) -> LLMResponse:
    text = facts if isinstance(facts, str) else json.dumps({"facts": facts})
    return LLMResponse(text=text, stop_reason=stop, input_tokens=0, output_tokens=10, model="m", provider="fake")


def _read(provider: FakeProvider, spans, on_record=(), **options) -> list[KnowledgeEntry]:
    return asyncio.run(LLMFactReader(provider, **options).read(spans, list(on_record)))


SPANS = [(_said(6, "user", "Let's use Stripe for payments, with a 4% fee."), _said(7, "assistant", "Agreed.")),
         (_said(8, "user", "Change of plan: we're moving to AWS."), _said(9, "assistant", "Noted."))]


class TestReadingByModel:
    def test_each_fact_is_filed_against_its_turn_under_its_topic(self):
        provider = FakeProvider(responses=[_answer([
            {"turn": 6, "topic": "Payments", "value": "Stripe", "kind": "decision"},
            {"turn": 6, "topic": "platform fee", "value": " 4% ", "kind": "fact"}])])

        facts = _read(provider, SPANS)

        assert [(f.message_turn, f.topic_label, f.content, f.knowledge_type) for f in facts] == [
            (6, "payments", "Stripe", KnowledgeType.DECISION), (6, "platform fee", "4%", KnowledgeType.FACT)]

    def test_the_model_is_shown_the_facts_on_record_and_the_turns_as_delimited_data(self):
        provider = FakeProvider(responses=[_answer([])])
        on_record = [KnowledgeEntry(message_turn=2, content="Google Cloud", topic_label="cloud",
                                    knowledge_type=KnowledgeType.DECISION),
                     KnowledgeEntry(message_turn=1, content="old", topic_label="gone",
                                    knowledge_type=KnowledgeType.FACT, knowledge_status=KnowledgeStatus.SUPERSEDED)]

        _read(provider, SPANS, on_record)

        [call] = provider.calls
        content = call.messages[0].content
        assert call.system == load_prompt(FACTS_PROMPT)
        assert content.index(FACTS_OPEN) < content.index("cloud: Google Cloud") < content.index(FACTS_CLOSE)
        assert "gone: old" not in content, "a replaced value is not on record"
        assert content.index(TRANSCRIPT_OPEN) < content.index("[turn 8] user: Change of plan") < content.index(
            TRANSCRIPT_CLOSE)

    @pytest.mark.parametrize("fact", [
        {"turn": 3, "topic": "cloud", "value": "AWS"},                                      # a turn it was not shown
        {"turn": 8, "topic": "if the region count is small and fixed", "value": "fine"},    # a sentence, not a topic
        {"turn": 8, "topic": "cloud", "value": "   "},                                      # no value
        {"turn": 8, "topic": "cloud", "value": "x" * 201},                                  # a paragraph, not a value
    ])
    def test_a_fact_that_is_not_a_short_topic_and_value_about_a_given_turn_is_dropped(self, fact):
        provider = FakeProvider(responses=[_answer([fact, {"turn": 8, "topic": "cloud", "value": "AWS"}])])

        assert [(f.message_turn, f.content) for f in _read(provider, SPANS)] == [(8, "AWS")]

    def test_who_stated_a_fact_comes_from_its_turn_never_from_the_model(self):
        """A tool result arrives in a user-role turn, but it is output, not the user's word."""
        tool_result = Message(role="user", content="shards: 3", token_count=5, turn_index=10,
                              raw_content=[{"type": "tool_result", "tool_use_id": "t1", "content": "shards: 3"}])
        provider = FakeProvider(responses=[_answer([{"turn": 8, "topic": "cloud", "value": "AWS"},
                                                    {"turn": 9, "topic": "region", "value": "eu-west-1"},
                                                    {"turn": 10, "topic": "shards", "value": "3"}])])

        facts = _read(provider, [*SPANS, (tool_result,)])

        assert [(f.message_turn, f.speaker) for f in facts] == [(8, "user"), (9, "assistant"), (10, "assistant")]

    def test_facts_come_back_in_the_order_they_were_said(self):
        """Filing is sequential, so a model that lists turn 8 before turn 6 must not
        make the earlier statement look like the replacement."""
        provider = FakeProvider(responses=[_answer([{"turn": 8, "topic": "cloud", "value": "AWS"},
                                                    {"turn": 6, "topic": "cloud", "value": "Google Cloud"}])])

        assert [f.message_turn for f in _read(provider, SPANS)] == [6, 8]

    def test_an_unknown_kind_is_a_plain_fact(self):
        provider = FakeProvider(responses=[_answer([{"turn": 8, "topic": "cloud", "value": "AWS", "kind": "vibe"}])])

        assert _read(provider, SPANS)[0].knowledge_type is KnowledgeType.FACT

    def test_json_wrapped_in_a_code_fence_is_read(self):
        fenced = '```json\n{"facts": [{"turn": 8, "topic": "cloud", "value": "AWS"}]}\n```'

        assert [f.content for f in _read(FakeProvider(responses=[_answer(fenced)]), SPANS)] == ["AWS"]

    @pytest.mark.parametrize("answer", [
        _answer("The turns settle the cloud: AWS."),                       # no JSON at all
        _answer('{"facts": "AWS"}'),                                       # not the expected shape
        _answer([{"turn": 8, "topic": "cloud", "value": "AWS"}], StopReason.TRUNCATED),   # cut off
    ])
    def test_an_answer_that_cannot_be_trusted_fails_the_read(self, answer):
        with pytest.raises(LLMResponseError):
            _read(FakeProvider(responses=[answer]), SPANS)

    def test_a_provider_failure_propagates(self):
        with pytest.raises(LLMTimeoutError):
            _read(FakeProvider(responses=[LLMTimeoutError("slow")]), SPANS)

    def test_a_large_collection_is_read_in_calls_that_each_see_what_the_last_one_read(self):
        """A change of plan in the second call must replace a value the first call
        read, so the second call has to see it on record."""
        provider = FakeProvider(responses=[_answer([{"turn": 6, "topic": "payments", "value": "Stripe"}]),
                                           _answer([{"turn": 8, "topic": "payments", "value": "Adyen"}])])

        facts = _read(provider, SPANS, max_batch_tokens=40)

        assert len(provider.calls) == 2, "each span of 40 tokens fills a call"
        assert "payments: Stripe" in provider.calls[1].messages[0].content
        assert [f.content for f in facts] == ["Stripe", "Adyen"]


class TestReadingByPatterns:
    def test_only_the_users_own_words_are_read(self):
        """The assistant's explanations yield sentence fragments, and tool results
        are output, not statements."""
        tool_result = Message(role="user", content="The config is broken.", token_count=5, turn_index=2,
                              raw_content=[{"type": "tool_result", "tool_use_id": "t1", "content": "The config is broken."}])
        spans = [(_said(0, "user", "The database is Postgres."),
                  _said(1, "assistant", "The trade-off is real but modest at this size."), tool_result)]

        facts = asyncio.run(PatternFactReader().read(spans, []))

        assert [(f.message_turn, f.topic_label, f.content, f.speaker) for f in facts] == [
            (0, "database", "Postgres", "user")]


class TestARelayedReportIsTheAssistantsSide:
    """A background subagent's report arrives under the user role; reading it as the user would let
    another agent's finding replace a decision the user made."""

    REPORT = Message(role="user", content="<agent-message>The config uses MySQL.</agent-message>", token_count=12,
                     turn_index=10, relayed=True)

    def test_the_model_is_shown_it_as_a_report_and_its_facts_are_not_the_users(self):
        provider = FakeProvider(responses=[_answer([{"turn": 10, "topic": "database", "value": "MySQL",
                                                     "kind": "decision"}])])

        [fact] = _read(provider, [(self.REPORT,)])

        assert "[turn 10] subagent report: <agent-message>" in provider.calls[0].messages[0].content
        assert fact.speaker == "assistant"

    def test_it_never_replaces_the_users_decision(self):
        """The failing case found in a live trace: the user chose Postgres, a subagent reported MySQL."""
        decided = KnowledgeEntry(message_turn=2, content="Postgres", topic_label="database",
                                 knowledge_type=KnowledgeType.DECISION, speaker="user")
        provider = FakeProvider(responses=[_answer([{"turn": 10, "topic": "database", "value": "MySQL",
                                                     "kind": "decision"}])])

        filed, _ = file_knowledge((decided,), tuple(_read(provider, [(self.REPORT,)], [decided])))

        assert [(f.content, f.knowledge_status) for f in filed] == [("Postgres", KnowledgeStatus.ACTIVE)]

    def test_patterns_do_not_read_it_as_the_user(self):
        report = self.REPORT.model_copy(update={"content": "<agent-message>The database is MySQL.</agent-message>"})
        assert asyncio.run(PatternFactReader().read([(report,)], [])) == []
