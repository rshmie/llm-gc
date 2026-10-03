import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from llm_gc.config import GCConfig
from llm_gc.config.constants import APP_VERSION
from llm_gc.engine import ContextComposer, GarbageCollector
from llm_gc.engine.compaction import NoOpCompactor
from llm_gc.engine.generations import GenerationalMemory
from llm_gc.engine.generations.permanent_generation import PermanentGeneration
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy
from llm_gc.events import EventBus
from llm_gc.extraction import KnowledgeExtractor
from llm_gc.monitoring import ContextHealth, ContextHealthMonitor
from llm_gc.scoring import DecisionScorer, DensityScorer, RecencyScorer, ReferenceScorer, RelevanceScorer, RelevanceType
from llm_gc.visualizer.demo_conversation import DemoConversationFeeder

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# Demo-session tuning, local to the visualizer (the same lesson as
# max_recent_transitions: knobs that only shape a peripheral surface don't
# belong in the shared GCConfig defaults). A tiny window and a fast recency
# decay make GC engage after a couple of clicks instead of after 90k tokens.
DEMO_CONTEXT_WINDOW = 120
DEMO_GC_THRESHOLD = 0.5
DEMO_MIN_COMPACTABLE_TOKENS = 3
DEMO_LAST_N_TURNS_TO_KEEP = 3
DEMO_RECENCY_DECAY_RATE = 0.25


def _build_demo_config() -> GCConfig:
    return GCConfig(context_window=DEMO_CONTEXT_WINDOW, gc_threshold=DEMO_GC_THRESHOLD,
                    min_compactable_tokens=DEMO_MIN_COMPACTABLE_TOKENS,
                    last_n_turns_to_keep=DEMO_LAST_N_TURNS_TO_KEEP)


def _build_relevance_scorer(event_bus: EventBus) -> RelevanceScorer:
    # All four lightweight scorers run for real; only recency carries weight so
    # the demo's classifications are predictable. SimilarityScorer joins in
    # Phase 7 - it needs a SentenceTransformer download at startup, and the
    # monitor's data path (what this scaffold proves) never looks inside scores.
    return RelevanceScorer(
        scorers=[RecencyScorer(decay_rate=DEMO_RECENCY_DECAY_RATE), DensityScorer(),
                 DecisionScorer(), ReferenceScorer()],
        weights={RelevanceType.RECENCY: 1.0, RelevanceType.DENSITY: 0.0,
                 RelevanceType.DECISION: 0.0, RelevanceType.REFERENCE: 0.0},
        event_bus=event_bus)


def create_app() -> FastAPI:
    """Wire one demo session: engine + monitor on one shared bus, plus the routes.

    The dashboard is a read-only consumer of the same EventBus every other
    consumer reads from - the whole point of this scaffold is proving that
    data path. The one write path (POST /gc/run) exists because Phase 5 has
    no sessions yet: nothing else holds a conversation to collect, so the
    server owns a synthetic one and calls collect() directly.
    """
    app = FastAPI(title="LLM-GC Context Health Dashboard", version=APP_VERSION)

    event_bus = EventBus()
    gc_config = _build_demo_config()
    generational_memory = GenerationalMemory(event_bus=event_bus,
                                             knowledge_extractor=KnowledgeExtractor(event_bus=event_bus),
                                             permanent_generation=PermanentGeneration(event_bus=event_bus),
                                             compactor=NoOpCompactor(event_bus=event_bus))
    garbage_collector = GarbageCollector(
        event_bus=event_bus, gc_config=gc_config,
        relevance_scorer=_build_relevance_scorer(event_bus),
        sweeper=Sweeper(sweeper_strategy=ThresholdSweepStrategy(gc_config=gc_config),
                        gc_config=gc_config, event_bus=event_bus),
        context_composer=ContextComposer(event_bus=event_bus, generational_memory=generational_memory,
                                         compactor=NoOpCompactor(event_bus=event_bus)))
    monitor = ContextHealthMonitor(event_bus=event_bus, gc_config=gc_config,
                                   generational_memory=generational_memory)

    app.state.garbage_collector = garbage_collector
    app.state.monitor = monitor
    app.state.feeder = DemoConversationFeeder()
    app.state.conversation = []

    @app.get("/")
    def serve_dashboard() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/health")
    def get_health() -> ContextHealth:
        return app.state.monitor.get_snapshot()

    @app.get("/config")
    def get_config() -> dict:
        # The snapshot deliberately doesn't carry config (it reports state, not
        # settings); the meter's threshold tick needs these two values once.
        return {"context_window": gc_config.context_window, "gc_threshold": gc_config.gc_threshold}

    @app.post("/gc/run")
    async def run_gc() -> dict:
        new_turns = app.state.feeder.next_turns()
        app.state.conversation = app.state.conversation + new_turns
        gc_result = await app.state.garbage_collector.collect(app.state.conversation)
        app.state.conversation = gc_result.final_messages
        logger.info("Dashboard-triggered GC run finished",
                    extra={"gc_run_id": gc_result.gc_run_id, "status": gc_result.status.value})
        return {"gc_run_id": gc_result.gc_run_id, "status": gc_result.status.value,
                "tokens_before": gc_result.tokens_before, "tokens_in_final": gc_result.tokens_in_final,
                "tokens_saved": gc_result.tokens_saved,
                "turns_added": len(new_turns), "conversation_length": len(app.state.conversation)}

    return app
