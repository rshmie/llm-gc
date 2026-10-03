#!/usr/bin/env python3
"""Watch a conversation move through the GC engine, one turn at a time.

Run it:

    python scripts/trace_session.py
    python scripts/trace_session.py --window 1200 --threshold 0.4

This is developer tooling, not library code, which is why it lives in `scripts/`
rather than under `src/llm_gc/`. Nobody who installs the package should be able to
import it.

It exists because the engine is easier to understand in motion than on the page.
Every stage prints what it decided and why, so a conversation that starts verbatim
and ends up partly summarised, partly archived, and partly forgotten can be read as
a sequence of decisions rather than inferred from a final number.

**It learns everything from the event bus.** The trace subscribes to the same events
the dashboard subscribes to and calls no engine internals. That is the point of the
instrumentation design, and it is also the honest test of it: anything this trace
cannot show you is something the dashboard could not show you either.
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Run from a checkout without installing: make `src/` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gc.config import GCConfig  # noqa: E402
from llm_gc.engine.compaction import BaseCompactor, LLMCompactor, NoOpCompactor  # noqa: E402
from llm_gc.engine.gc_service import GcService  # noqa: E402
from llm_gc.engine.generations import GenerationalMemory  # noqa: E402
from llm_gc.engine.generations.permanent_generation import PermanentGeneration  # noqa: E402
from llm_gc.engine.sweep import Sweeper, ThresholdSweepStrategy  # noqa: E402
from llm_gc.events import Event, EventBus, EventType  # noqa: E402
from llm_gc.extraction import KnowledgeExtractor  # noqa: E402
from llm_gc.llm import AnthropicClient, FakeProvider, LLMProvider, LLMResponse, StopReason  # noqa: E402
from llm_gc.models import Message, SweepClassification  # noqa: E402
from llm_gc.scoring import (  # noqa: E402
    DecisionScorer,
    DensityScorer,
    RecencyScorer,
    ReferenceScorer,
    RelevanceScorer,
    RelevanceType,
)
from llm_gc.session import SessionManager  # noqa: E402
from llm_gc.utils import count_tokens  # noqa: E402

SESSION_ID = "trace"

# A conversation with structure, not filler. Some turns state a fact the extractor
# can find ("X is Y"); two of them contradict an earlier fact, so archiving the
# older one and then meeting the newer one produces a real supersession. The rest
# are ordinary chatter, which is what makes the scorers' job non-trivial: if every
# turn mattered there would be nothing to collect.
CONVERSATION: list[tuple[str, str]] = [
    ("user", "We need to decide on the storage layer for the new ingest service before Friday."),
    ("assistant", "Understood. What are the constraints we are working within on throughput and retention?"),
    ("user", "the database is postgres, and we are keeping ninety days of raw events."),
    ("assistant", "Postgres with ninety days of raw events is workable if we partition by day and archive older partitions."),
    ("user", "the cache is redis, sitting in front of the read path for the dashboard queries."),
    ("assistant", "Redis in front of the read path makes sense. I would set a short TTL so stale aggregates expire quickly."),
    ("user", "We walked through the numbers for the ingest spike last quarter and it peaked around forty thousand events a second."),
    ("assistant", "Forty thousand a second is above what a single writer handles comfortably, so the partitioning matters more than the cache."),
    ("user", "Can you remind me what we said about the retention window earlier?"),
    ("assistant", "Ninety days of raw events, with older partitions archived rather than dropped outright."),
    ("user", "the queue is rabbitmq for now, though the platform team keeps suggesting we move to kafka."),
    ("assistant", "RabbitMQ is fine at this volume. Kafka earns its operational cost at an order of magnitude more throughput."),
    ("user", "I spent the morning cleaning up the stale feature branches nobody had touched since the migration."),
    ("assistant", "Worth doing. Stale branches make the review queue harder to read than it needs to be."),
    ("user", "We paired on the migration script for an hour and found two columns that were never backfilled."),
    ("assistant", "Two unbackfilled columns would explain the nulls in the weekly report. I would backfill before the cutover."),
    ("user", "Also the flaky integration run came back again, same timeout in the fixture teardown."),
    ("assistant", "A timeout in teardown usually means a connection is being held open past the test. Worth checking the pool."),
    ("user", "the database is mysql now, the platform team moved us last week without telling anyone."),
    ("assistant", "That changes the partitioning plan considerably, since MySQL handles day partitions differently from Postgres."),
    ("user", "We went over the incident timeline from last week and the gap was in the alerting, not the ingest path."),
    ("assistant", "If the gap was in alerting then the ingest changes will not help. The alert thresholds are the thing to fix."),
    ("user", "Sketched the onboarding flow on the whiteboard, it needs one fewer screen than the current one."),
    ("assistant", "Fewer screens is the right direction. The second confirmation step has never caught a real mistake."),
    ("user", "So remind me, which database are we actually targeting for the ingest service?"),
    ("assistant", "Based on what you told me, MySQL, following last week's platform migration."),
]


@dataclass
class RunTrace:
    """What one `gc_update` pass did, assembled purely from events."""

    promoted: list[tuple[list[int], int, int]] = field(default_factory=list)
    archived: list[tuple[int, list[str]]] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    degraded: list[tuple[str, list[int], str]] = field(default_factory=list)

    def clear(self) -> None:
        self.promoted.clear()
        self.archived.clear()
        self.superseded.clear()
        self.degraded.clear()


class Tracer:
    """An event-bus subscriber, exactly like the dashboard's monitor.

    It holds no reference to the engine and calls nothing on it. Everything it
    prints arrived as an event, which is what makes the instrumentation claim
    checkable rather than aspirational: a stage that emits nothing is invisible
    here, and would be invisible to any other consumer too.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.current = RunTrace()
        self.last_gc_result = None
        # Whole-run totals, kept separately from `current` (which is cleared each
        # pass). The closing report is computed from these rather than written as
        # prose about the code: a hardcoded "this is fixed" line keeps printing
        # after the thing it describes regresses, which is the exact dishonesty
        # this tool exists to catch elsewhere.
        self.compaction_deltas: list[int] = []
        self.old_gen_evictions = 0
        self.young_archives = 0
        self.degradation_count = 0
        event_bus.subscribe(EventType.MESSAGE_PROMOTED_TO_OLD_GEN, self._on_promoted)
        event_bus.subscribe(EventType.MESSAGE_ARCHIVED, self._on_archived)
        event_bus.subscribe(EventType.KNOWLEDGE_ENTRY_SUPERSEDED, self._on_superseded)
        event_bus.subscribe(EventType.AGING_DEGRADED, self._on_degraded)
        event_bus.subscribe(EventType.GC_FINISHED, self._on_gc_finished)

    def _on_promoted(self, event: Event) -> None:
        self.current.promoted.append(
            (event.data["source_turn_indices"], event.data["tokens_before"], event.data["tokens_after"])
        )
        self.compaction_deltas.append(event.data["tokens_after"] - event.data["tokens_before"])

    def _on_archived(self, event: Event) -> None:
        message = event.data["message"]
        entries = event.data["extracted_knowledge_entries"]
        self.current.archived.append((message.turn_index, [entry.topic_label for entry in entries]))
        # Both hops are archives; `source_generation` is what tells a young turn's
        # first hop from a summary's last one.
        if event.data["source_generation"] == "old":
            self.old_gen_evictions += 1
        else:
            self.young_archives += 1

    def _on_superseded(self, event: Event) -> None:
        for entry in event.data["superseded_knowledge_entries"]:
            self.current.superseded.append(entry.topic_label)

    def _on_degraded(self, event: Event) -> None:
        degradation = event.data["degradation"]
        self.current.degraded.append(
            (degradation.reason.value, degradation.turn_indices, degradation.detail)
        )
        self.degradation_count += 1

    def _on_gc_finished(self, event: Event) -> None:
        self.last_gc_result = event.data["gc_result"]


# Canned summaries for the offline run, in the shape a real model would answer:
# plain prose, past tense, decisions and values preserved. They are short on
# purpose — the saving a real summary produces is the thing this trace exists to
# show, and with no API key available this is the honest way to show it. The text
# is scripted; the token arithmetic around it is entirely real.
_SCRIPTED_SUMMARIES = [
    "The user set a Friday deadline for choosing the ingest storage layer and gave the constraints.",
    "Postgres was chosen with ninety days of raw event retention, partitioned by day.",
    "Redis was placed in front of the dashboard read path with a short TTL.",
    "Ingest peaked near forty thousand events a second last quarter, so partitioning matters most.",
    "The retention window was restated as ninety days, with old partitions archived rather than dropped.",
    "RabbitMQ was kept as the queue; Kafka was judged not worth its operational cost at this volume.",
    "Stale feature branches were cleaned up and the migration script revealed two unbackfilled columns.",
    "A flaky integration timeout in fixture teardown was traced to a connection held past the test.",
    "The platform team moved the database to MySQL, which changes the day-partitioning plan.",
    "The incident gap was in alerting rather than ingest, so alert thresholds are the fix.",
    "The onboarding flow was redrawn with one fewer screen; the second confirmation never caught a mistake.",
]


def _offline_provider() -> LLMProvider:
    """A provider that answers from the script above, so the trace runs with no key."""
    return FakeProvider(
        responses=[
            LLMResponse(
                text=text,
                stop_reason=StopReason.COMPLETE,
                input_tokens=0,
                output_tokens=count_tokens(text),
                model="scripted",
                provider="offline",
            )
            for text in _SCRIPTED_SUMMARIES
        ]
    )


def build_compactor(event_bus: EventBus, kind: str, api_key: str | None) -> tuple[BaseCompactor, str]:
    """Pick the compactor, and say plainly which one you got.

    Three cases, because being vague about which one ran would undermine the whole
    point of the trace:
      - `noop`: the baseline the engine ships with. It expands.
      - `llm` with a key: the real thing, calling Anthropic.
      - `llm` without a key: the real LLMCompactor driven by scripted summaries, so
        every guard, every token count and every event is real and only the
        summary *wording* is canned.
    """
    if kind == "noop":
        return NoOpCompactor(event_bus=event_bus), "NoOpCompactor (baseline: wraps the run, does not summarise)"
    if api_key:
        return (
            LLMCompactor(event_bus=event_bus, provider=AnthropicClient(api_key=api_key)),
            "LLMCompactor -> AnthropicClient (live model calls)",
        )
    return (
        LLMCompactor(event_bus=event_bus, provider=_offline_provider()),
        "LLMCompactor -> scripted provider (no ANTHROPIC_API_KEY; summary wording is canned, arithmetic is real)",
    )


def build_engine(config: GCConfig, compactor_kind: str, api_key: str | None) -> tuple[GcService, GenerationalMemory, Tracer, str]:
    """Wire the real engine. No fakes, no stubs — this is the production graph.

    The compactor is the one thing swapped, which is the Strategy pattern paying
    off: nothing else in this function changes between the two runs.
    """
    event_bus = EventBus()
    tracer = Tracer(event_bus)
    compactor, compactor_label = build_compactor(event_bus, compactor_kind, api_key)

    generational_memory = GenerationalMemory(
        event_bus=event_bus,
        knowledge_extractor=KnowledgeExtractor(event_bus=event_bus),
        permanent_generation=PermanentGeneration(event_bus=event_bus),
        compactor=compactor,
    )

    # Similarity is left out on purpose: it downloads a SentenceTransformer at
    # startup, and a trace tool that takes ninety seconds to boot gets run once.
    relevance_scorer = RelevanceScorer(
        scorers=[RecencyScorer(decay_rate=0.15), DensityScorer(), DecisionScorer(), ReferenceScorer()],
        weights={
            RelevanceType.RECENCY: 0.6,
            RelevanceType.DENSITY: 0.2,
            RelevanceType.DECISION: 0.1,
            RelevanceType.REFERENCE: 0.1,
        },
        event_bus=event_bus,
    )

    service = GcService(
        session_manager=SessionManager(),
        generational_memory=generational_memory,
        relevance_scorer=relevance_scorer,
        sweeper=Sweeper(
            sweeper_strategy=ThresholdSweepStrategy(gc_config=config), gc_config=config, event_bus=event_bus
        ),
        event_bus=event_bus,
        gc_config=config,
    )
    return service, generational_memory, tracer, compactor_label


def _bar(current: int, window: int, width: int = 28) -> str:
    """A token-pressure bar, so crossing the threshold is something you see."""
    filled = min(width, int(width * current / window)) if window else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


async def trace(config: GCConfig, *, verbose: bool, compactor_kind: str, api_key: str | None) -> None:
    service, memory, tracer, compactor_label = build_engine(config, compactor_kind, api_key)
    threshold_tokens = int(config.context_window * config.gc_threshold)

    print(f"\ncompactor: {compactor_label}")
    print(f"context_window={config.context_window}  gc_threshold={config.gc_threshold} "
          f"-> GC engages at {threshold_tokens} tokens")
    print(f"keep>={config.keep_threshold}  archive<{config.archive_threshold}  "
          f"min_compactable_tokens={config.min_compactable_tokens}  "
          f"last_n_turns_to_keep={config.last_n_turns_to_keep}\n")
    print("=" * 100)

    raw_tokens = 0

    for turn_index, (role, content) in enumerate(CONVERSATION):
        token_count = count_tokens(content)
        raw_tokens += token_count
        tracer.current.clear()

        await service.gc_update(
            SESSION_ID,
            Message(role=role, content=content, token_count=token_count, turn_index=turn_index),
        )
        result = tracer.last_gc_result
        assert result is not None, "gc_update must always emit GC_FINISHED"

        before = result.tokens_before
        print(f"\nturn {turn_index:>2}  {role:<9} {token_count:>3} tok  {content[:58]!r}")
        print(f"          {_bar(before, config.context_window)} {before:>5}/{config.context_window} tok"
              f"   {result.status.value}")

        if result.sweep_result is None:
            # Below the pressure threshold the turn is recorded and nothing ages.
            # Aging is not free — it trades verbatim text for a summary, and with a
            # real model it costs an API call — so it waits until it is needed.
            continue

        counts = result.sweep_result.classification_counts
        print(f"          swept: KEEP {counts[SweepClassification.KEEP]}  "
              f"COMPACT {counts[SweepClassification.COMPACT]}  "
              f"ARCHIVE {counts[SweepClassification.ARCHIVE]}")

        if verbose:
            for entry in result.sweep_result.sweep_entries:
                flag = f"  <- {entry.override_reason}" if entry.override_applied else ""
                print(f"            turn {entry.turn_index:>2}  score {entry.relevance_score:.2f}  "
                      f"{entry.classification.value:<8}{flag}")

        for turns, tokens_before, tokens_after in tracer.current.promoted:
            delta = tokens_after - tokens_before
            sign = "+" if delta > 0 else ""
            print(f"          -> old gen:       turns {turns}  {tokens_before} -> {tokens_after} tok "
                  f"({sign}{delta})")
        for reason, turns, detail in tracer.current.degraded:
            # The point of the whole exercise: a pass that fell short says so,
            # rather than leaving a consumer to notice the counts do not add up.
            print(f"          !! DEGRADED [{reason}] turns {turns} stayed verbatim")
            print(f"             {detail}")
        for turn, topics in tracer.current.archived:
            print(f"          -> permanent gen: turn {turn}  facts {topics}")
        for topic in tracer.current.superseded:
            print(f"          -> superseded:    {topic!r} (an older fact was overridden)")

        saved = before - result.tokens_in_final
        degraded_note = f"   [{len(result.degradations)} degradation(s)]" if result.degradations else ""
        print(f"          result: {before} -> {result.tokens_in_final} tok  "
              f"saved {'+' if saved > 0 else ''}{saved}{degraded_note}")

    # ---------------------------------------------------------------- final state
    # The query is the turn about to be sent. The proxy has it; a dashboard
    # reading the context does not, which is why it is optional.
    last_turn_text = CONVERSATION[-1][1]
    final_context = await service.gc_collect(SESSION_ID, query=last_turn_text)
    memory_block = final_context[0] if final_context and final_context[0].turn_index == -1 else None
    sent_tokens = sum(message.token_count for message in final_context)
    old_gen = memory.get_old_gen()
    permanent = memory.get_permanent_gen()
    young = [m for m in final_context if m not in old_gen and m is not memory_block]

    print("\n" + "=" * 100)
    print("\nWHERE EVERY TURN ENDED UP\n")
    print(f"  young generation (verbatim):   {len(young):>3} turns")
    print(f"  old generation (summarised):   {len(old_gen):>3} summaries")
    print(f"  permanent generation (facts):  {len(permanent):>3} entries")

    print("\nWHAT THE MODEL ACTUALLY RECEIVES\n")
    for message in final_context:
        if message is memory_block:
            print(f"  MEM turn   -  {message.token_count:>3} tok  (retrieved facts, see below)")
            continue
        marker = "old" if message in old_gen else "   "
        print(f"  {marker} turn {message.turn_index:>2}  {message.token_count:>3} tok  {message.content[:70]!r}")

    if memory_block is not None:
        print(f"\n  the injected memory block, ranked against {last_turn_text[:40]!r}:")
        for line in memory_block.content.split("\n"):
            print(f"    {line[:94]}")

    print("\nPERMANENT GENERATION (extracted facts)\n")
    for entry in permanent:
        print(f"      turn {entry.message_turn:>2}  {entry.topic_label!r} = {entry.content[:60]!r}")

    print(f"\n  raw conversation: {raw_tokens} tok")
    print(f"  context sent:     {sent_tokens} tok   ({sent_tokens / raw_tokens:.2f}x)")

    # The two things this trace is designed to make impossible to miss. Both are
    # known, recorded gaps, and both are the reason the next pieces are being
    # built — but reading them off a real run is different from being told.
    _print_honesty_check(tracer, memory_block, permanent, old_gen, raw_tokens, sent_tokens)


def _check(ok: bool, claim: str, evidence: str) -> None:
    """One line of the closing report: a verdict, a claim, and the number behind it.

    Every line states the evidence it was decided on, so a reader can disagree with
    the verdict. A report that only prints conclusions is asking to be trusted,
    which is the opposite of what this tool is for.
    """
    print(f"  [{'ok ' if ok else 'GAP'}]  {claim}")
    print(f"         {evidence}")


def _print_honesty_check(tracer, memory_block, permanent, old_gen, raw_tokens, sent_tokens) -> None:
    """What this run does and does not entitle us to claim.

    Computed from the events this run emitted, never written as prose about the
    code. The difference matters: a hardcoded "this is fixed" line keeps printing
    long after the thing it describes regresses, and a tool whose job is honesty
    cannot be the one component exempt from checking itself. An earlier version of
    this section did exactly that - it carried hand-edited FIXED labels under a
    heading that still read STILL BROKEN.
    """
    print("\nHONESTY CHECK - what this run can and cannot claim\n")

    deltas = tracer.compaction_deltas
    if not deltas:
        _check(False, "compaction shrinks the context: NOT EXERCISED",
               "no run was promoted to old gen, so this run proves nothing either way")
    else:
        shrank = [d for d in deltas if d < 0]
        _check(len(shrank) == len(deltas),
               f"compaction shrinks the context: {len(shrank)} of {len(deltas)} promotions reduced tokens",
               f"best {min(deltas):+d} tok, worst {max(deltas):+d} tok per run")

    if not permanent:
        _check(False, "archived turns reach the model: NOT EXERCISED",
               "nothing was archived, so no fact could be injected")
    elif memory_block is None:
        _check(False, f"archived turns reach the model: 0 of {len(permanent)} injected",
               "nothing stored matched the final turn - retrieval is lexical, so a "
               "paraphrase misses")
    else:
        injected = memory_block.content.count("\n")
        _check(True, f"archived turns reach the model: {injected} of {len(permanent)} facts injected",
               f"{memory_block.token_count} tok of memory; the rest did not match this turn")

    if tracer.old_gen_evictions == 0 and len(old_gen) <= 1:
        _check(False, "old generation is bounded: NOT EXERCISED",
               f"{len(old_gen)} summaries filed and none evicted - too short a run to tell")
    else:
        _check(tracer.old_gen_evictions > 0,
               f"old generation is bounded: {tracer.old_gen_evictions} summaries evicted, "
               f"{len(old_gen)} remaining",
               "a single run cannot prove a bound; "
               "tests/engine/test_old_gen_sweep.py measures it across lengths")

    _check(sent_tokens < raw_tokens,
           f"the context is smaller than the raw conversation: {sent_tokens} vs {raw_tokens} tok",
           f"{sent_tokens / raw_tokens:.2f}x, from {tracer.young_archives} young archives, "
           f"{len(deltas)} compactions and {tracer.old_gen_evictions} evictions")

    _check(tracer.degradation_count == 0,
           f"every pass did what it intended: {tracer.degradation_count} degradation(s)",
           "a degradation means turns stayed verbatim - correct, but larger than planned")

    print("\n  Known gaps this run cannot show, because no run can:")
    print("    - retrieval is lexical, so a paraphrase misses ('datastore' vs topic 'database')")
    print("    - a subject that is a quantity gets an odd label ('forty thousand a second')")
    print("    - summary *quality* is unmeasured here; that needs an eval, not a trace")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # A small window by default so GC engages a third of the way in; the real
    # default (128k) would need a conversation nobody wants to read. 400 rather
    # than something roomier because a roomier window collapses hard once and then
    # coasts below the threshold forever, so the run never reaches a second pass
    # and old-gen eviction is never exercised - a default that leaves half the
    # engine untouched is a poor default for a tool meant to show the engine.
    parser.add_argument("--window", type=int, default=400, help="context_window in tokens")
    parser.add_argument("--threshold", type=float, default=0.5, help="fraction of the window that triggers GC")
    parser.add_argument("--keep", type=float, default=0.55, help="score at or above which a turn stays verbatim")
    parser.add_argument("--archive", type=float, default=0.25, help="score below which a turn is archived")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every turn's score and classification")
    parser.add_argument("--compactor", choices=("noop", "llm"), default="noop",
                        help="noop = the shipped baseline that expands; llm = LLMCompactor")
    args = parser.parse_args()

    # Read from the environment rather than a flag, so a key never lands in shell
    # history. Absent is the normal case and the trace says so rather than failing.
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    config = GCConfig(
        context_window=args.window,
        gc_threshold=args.threshold,
        keep_threshold=args.keep,
        archive_threshold=args.archive,
        min_compactable_tokens=10,
        last_n_turns_to_keep=3,
    )
    asyncio.run(trace(config, verbose=args.verbose, compactor_kind=args.compactor, api_key=api_key))


if __name__ == "__main__":
    main()
