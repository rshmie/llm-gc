import asyncio
import json
import re

from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.engine.compaction import LLMCompactor, NoOpCompactor
from llm_gc.events import EventType
from llm_gc.extraction import LLMFactReader, PatternFactReader
from llm_gc.llm import FakeProvider, LLMResponse, MeteredProvider, StopReason
from llm_gc.llm.prompts import FACTS_PROMPT, load_prompt
from llm_gc.models import Message
from llm_gc.scoring import RelevanceType
from llm_gc.utils import count_tokens

from tests.engine.support import collect


def test_without_a_summary_provider_turns_are_kept_whole():
    service = build_gc_service(GCConfig(), summary_provider=None)

    assert isinstance(service.generational_memory.compactor, NoOpCompactor)


def test_a_summary_provider_is_what_the_compactor_calls_through_a_meter():
    provider = FakeProvider(responses=[])

    service = build_gc_service(GCConfig(), summary_provider=provider)

    compactor = service.generational_memory.compactor
    assert isinstance(compactor, LLMCompactor)
    assert isinstance(compactor.provider, MeteredProvider) and compactor.provider.inner is provider


def test_facts_are_read_by_the_model_that_summarises_or_by_patterns_without_one():
    provider = FakeProvider(responses=[])

    assert isinstance(
        build_gc_service(GCConfig(), summary_provider=None).generational_memory.fact_reader, PatternFactReader
    )
    reader = build_gc_service(GCConfig(), summary_provider=provider).generational_memory.fact_reader
    assert isinstance(reader, LLMFactReader) and isinstance(reader.provider, MeteredProvider)
    assert reader.provider.inner is provider


class _Model:
    """Answers the facts call with what each turn settles, and every other call with a summary."""

    SETTLED = {0: ("cloud", "Google Cloud"), 2: ("cloud", "AWS")}

    def __init__(self) -> None:
        self.facts_requests: list[str] = []

    async def generate(self, *, messages, system=None, max_output_tokens, timeout_s):
        if system == load_prompt(FACTS_PROMPT):
            request = messages[0].content
            self.facts_requests.append(request)
            facts = [
                {"turn": turn, "topic": topic, "value": value}
                for turn, (topic, value) in self.SETTLED.items()
                if f"[turn {turn}]" in request
            ]
            text = json.dumps({"facts": facts})
        else:
            text = "The assistant set out an architecture."
        return LLMResponse(
            text=text, stop_reason=StopReason.COMPLETE, input_tokens=0, output_tokens=5, model="m", provider="fake"
        )


def test_a_change_of_plan_in_a_later_collection_replaces_the_value_under_its_topic():
    """The model files the change under the topic on record, so the front tells the
    model the old value was replaced - once, and only by a real change."""
    model = _Model()
    service = build_gc_service(
        GCConfig(min_cut_tokens=0, last_n_messages_to_keep=2, context_window=100_000), summary_provider=model
    )
    said = [
        "We'll host on Google Cloud. What should the architecture be?",
        "A long answer. " * 60,
        "Change of plan: we're moving to AWS.",
        "Another long answer. " * 60,
        "Thanks.",
        "Welcome.",
    ]

    async def converse() -> None:
        for turn, text in enumerate(said):
            message = Message(
                role="user" if turn % 2 == 0 else "assistant",
                content=text,
                token_count=count_tokens(text),
                turn_index=turn,
            )
            service.gc_record("s", [message], [f"h{turn}"])
            await collect(service, "s")

    asyncio.run(converse())

    front = asyncio.run(service.gc_collect("s"))[0].content
    assert len(model.facts_requests) == 2, "one read per collection"
    assert "cloud: Google Cloud" in model.facts_requests[1], "the second read sees the first's fact on record"
    assert re.findall(r"↳ Replaced later.*", front) == ["↳ Replaced later (turn 2): cloud: AWS, not Google Cloud."]


def test_a_recommendation_never_marks_the_users_decision_replaced():
    """The decision-survival check's lost answer, end to end: the user chose 9
    shards, the assistant recommended 3, and the model filed the recommendation
    under the user's topic. The user's 9 must stay the one fact on that topic, with
    no "Replaced later" note telling the model otherwise."""

    class _Recommending(_Model):
        SETTLED = {0: ("index shards", "9, with 2 replicas each"), 1: ("index shards", "recommended 3")}

    model = _Recommending()
    service = build_gc_service(
        GCConfig(min_cut_tokens=0, last_n_messages_to_keep=2, context_window=100_000), summary_provider=model
    )
    said = [
        "Let's use 9 shards with 2 replicas each for the index.",
        "I'd recommend 3 primaries instead. " * 40,
        "How do we handle growth?",
        "Reindex through an alias. " * 40,
        "Thanks.",
        "Welcome.",
    ]

    async def converse() -> None:
        for turn, text in enumerate(said):
            message = Message(
                role="user" if turn % 2 == 0 else "assistant",
                content=text,
                token_count=count_tokens(text),
                turn_index=turn,
            )
            service.gc_record("s", [message], [f"h{turn}"])
            await collect(service, "s")

    asyncio.run(converse())

    front = asyncio.run(service.gc_collect("s"))[0].content
    assert "Replaced later" not in front
    snapshot = service.session_manager.store.get("s").snapshot
    assert [(f.content, f.speaker) for f in snapshot.active_knowledge] == [("9, with 2 replicas each", "user")]


def test_a_recommendation_in_the_same_exchange_never_outranks_the_users_decision():
    """The check's lost answer, a second way: the user's decision and the assistant's
    recommendation are read in one call, the model listing the recommendation first.
    The user's value must be the one note on the topic."""

    class _OutOfOrder(_Model):
        async def generate(self, *, messages, system=None, max_output_tokens, timeout_s):
            if system == load_prompt(FACTS_PROMPT) and "[turn 0]" in messages[0].content:
                text = json.dumps(
                    {
                        "facts": [
                            {"turn": 1, "topic": "index shards", "value": "recommended 3"},
                            {"turn": 0, "topic": "index shards", "value": "9"},
                        ]
                    }
                )
                return LLMResponse(
                    text=text,
                    stop_reason=StopReason.COMPLETE,
                    input_tokens=0,
                    output_tokens=5,
                    model="m",
                    provider="fake",
                )
            return await super().generate(
                messages=messages, system=system, max_output_tokens=max_output_tokens, timeout_s=timeout_s
            )

    model = _OutOfOrder()
    service = build_gc_service(
        GCConfig(min_cut_tokens=0, last_n_messages_to_keep=2, context_window=100_000), summary_provider=model
    )
    said = [
        "Let's use 9 shards. Is that reasonable?",
        "I'd recommend 3 primaries instead. " * 40,
        "How do we handle growth?",
        "Reindex through an alias. " * 40,
    ]

    async def converse() -> None:
        for turn, text in enumerate(said):
            message = Message(
                role="user" if turn % 2 == 0 else "assistant",
                content=text,
                token_count=count_tokens(text),
                turn_index=turn,
            )
            service.gc_record("s", [message], [f"h{turn}"])
            await collect(service, "s")

    asyncio.run(converse())

    snapshot = service.session_manager.store.get("s").snapshot
    assert [(f.content, f.speaker) for f in snapshot.knowledge] == [("9", "user")]
    assert "Replaced later" not in asyncio.run(service.gc_collect("s"))[0].content


def test_every_component_reports_on_the_one_bus():
    # A component on a private bus would be invisible to the dashboard and the monitor.
    service = build_gc_service(GCConfig(), summary_provider=None)

    assert service.generational_memory.compactor.event_bus is service.event_bus


def test_the_shipped_weights_let_an_old_span_reach_the_top_band():
    """Every level must be reachable by a span of any age, or one the conversation
    returns to could never be shown in full again."""
    config = GCConfig()
    service = build_gc_service(config, summary_provider=None)

    assert {scorer.scorer_name for scorer in service.span_scorer.importance} == {
        RelevanceType.DECISION,
        RelevanceType.DENSITY,
    }
    assert service.span_scorer.best_score_however_old >= config.keep_threshold + config.level_margin


def test_two_services_share_no_state():
    first = build_gc_service(GCConfig(), summary_provider=None)
    second = build_gc_service(GCConfig(), summary_provider=None)

    assert first.event_bus is not second.event_bus
    assert first.session_manager is not second.session_manager


def test_the_engines_sessions_announce_their_closes_on_its_own_bus():
    """The dashboard and the monitor listen on the service's bus; a close announced anywhere else reaches neither."""
    service = build_gc_service(GCConfig(), summary_provider=None)
    heard: list[str] = []
    service.event_bus.subscribe(EventType.SESSION_CLOSED, lambda event: heard.append(event.data["session_id"]))

    service.session_manager.get_or_create("s1")
    service.session_manager.close_session("s1")

    assert heard == ["s1"]
