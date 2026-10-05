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

    assert service.relevance_scorer.event_bus is service.event_bus
    assert service.sweeper.event_bus is service.event_bus
    assert service.generational_memory.compactor.event_bus is service.event_bus
    assert service.generational_memory.knowledge_extractor.event_bus is service.event_bus


def test_each_weighted_signal_has_its_scorer():
    # Scores are divided by the sum of all weights: a weight with no scorer would
    # deflate every score, and a scorer with no weight would count for nothing.
    service = build_gc_service(GCConfig(), summary_provider=None)

    wired = {scorer.scorer_name for scorer in service.relevance_scorer.scorers}
    assert wired == set(service.relevance_scorer.weights) == {
        RelevanceType.RECENCY, RelevanceType.DENSITY, RelevanceType.DECISION, RelevanceType.REFERENCE}


def test_two_services_share_no_state():
    first = build_gc_service(GCConfig(), summary_provider=None)
    second = build_gc_service(GCConfig(), summary_provider=None)

    assert first.event_bus is not second.event_bus
    assert first.session_manager is not second.session_manager
