from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.engine.compaction import LLMCompactor, NoOpCompactor
from llm_gc.llm import FakeProvider
from llm_gc.scoring import RelevanceType


def test_without_a_summary_provider_turns_are_kept_whole():
    service = build_gc_service(GCConfig(), summary_provider=None)

    assert isinstance(service.generational_memory.compactor, NoOpCompactor)


def test_a_summary_provider_is_what_the_compactor_calls():
    provider = FakeProvider(responses=[])

    service = build_gc_service(GCConfig(), summary_provider=provider)

    compactor = service.generational_memory.compactor
    assert isinstance(compactor, LLMCompactor)
    assert compactor.provider is provider


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
        RelevanceType.DECISION, RelevanceType.DENSITY}
    assert service.span_scorer.best_score_however_old >= config.keep_threshold + config.level_margin


def test_two_services_share_no_state():
    first = build_gc_service(GCConfig(), summary_provider=None)
    second = build_gc_service(GCConfig(), summary_provider=None)

    assert first.event_bus is not second.event_bus
    assert first.session_manager is not second.session_manager
