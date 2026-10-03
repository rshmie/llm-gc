"""The compactor that actually summarises: it asks a model.

`NoOpCompactor` is the baseline — it wraps a run in a marker and files it, which
*adds* tokens. This is the replacement, and it is the reason `llm/` exists at all.

It is a thin component on purpose. It owns three things and nothing else: how a
run of turns is presented to a model, what the output cap should be, and when a
returned summary must be refused. The HTTP call, the vendor's wire format and the
error taxonomy all belong to the injected provider, and this class never learns
which vendor it is talking to.
"""

import logging

from llm_gc.config.constants import (
    DEFAULT_COMPACTION_TARGET_RATIO,
    DEFAULT_COMPACTION_TIMEOUT_S,
    DEFAULT_MIN_COMPACTION_OUTPUT_TOKENS,
)
from llm_gc.engine.compaction.base_compactor import BaseCompactor
from llm_gc.engine.compaction.compaction_result import CompactionResult
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.events import Event, EventBus, EventType
from llm_gc.exceptions import CompactionRefused
from llm_gc.llm.prompts import COMPACTION_PROMPT, load_prompt
from llm_gc.llm.provider import LLMProvider
from llm_gc.models import Message
from llm_gc.utils import count_tokens

logger = logging.getLogger(__name__)

# The transcript is wrapped in markers the system prompt names explicitly, so the
# model can tell our instructions from the user's conversation. Unusual enough not
# to occur in ordinary prose, which is the only property that matters: a delimiter
# a turn can plausibly contain is not a delimiter.
TRANSCRIPT_OPEN = "<<<TRANSCRIPT"
TRANSCRIPT_CLOSE = "TRANSCRIPT>>>"


class LLMCompactor(BaseCompactor):
    """Summarises a run of turns by calling a language model.

    Subclasses `BaseCompactor` — an ABC rather than a `Protocol`, and correctly so:
    `_format_summary_marker` is real shared code that every compacted summary must
    use, so there is something to inherit. Contrast `LLMProvider`, which is a
    Protocol because it carries only a shape.

    This is the Strategy pattern from the compactor's side and dependency injection
    from the provider's: `GenerationalMemory` holds a `BaseCompactor` and cannot
    tell this one from the no-op, and this class holds an `LLMProvider` and cannot
    tell Anthropic from OpenAI from the fake. Swapping either is configuration.
    """

    COMPACTION_STRATEGY: CompactionStrategy = CompactionStrategy.LLM

    def __init__(
        self,
        event_bus: EventBus,
        provider: LLMProvider,
        *,
        timeout_s: float = DEFAULT_COMPACTION_TIMEOUT_S,
        target_ratio: float = DEFAULT_COMPACTION_TARGET_RATIO,
        min_output_tokens: int = DEFAULT_MIN_COMPACTION_OUTPUT_TOKENS,
        system_prompt: str | None = None,
    ) -> None:
        """Build the compactor.

        Args:
            event_bus: Where `MESSAGE_COMPACTED` is announced.
            provider: Any `LLMProvider`. Injected rather than constructed here, so
                this class needs no API key, no HTTP client, and no knowledge of
                which vendor is configured — and so tests can hand it a
                `FakeProvider` and exercise every branch below without a network.
            timeout_s: Wall-clock budget for one summarisation call.
            target_ratio: Output cap as a fraction of the run's own token count.
                A summary allowed to be as long as its input has not compacted
                anything, so the cap is what makes the saving structural rather
                than a hope about the model's brevity.
            min_output_tokens: Floor under the cap. A three-turn run of 40 tokens
                would otherwise get a 14-token budget, which truncates mid-sentence
                and is refused — turning a short run into a guaranteed wasted call.
            system_prompt: Override for the shipped prompt, for evals that compare
                wordings. Defaults to `prompts/compaction.md`.
        """
        super().__init__(event_bus)
        self.provider = provider
        self._timeout_s = timeout_s
        self._target_ratio = target_ratio
        self._min_output_tokens = min_output_tokens
        # Resolved once, at construction. A prompt read on every compaction would
        # put a file read on the hot path for a string that cannot change.
        self._system_prompt = system_prompt if system_prompt is not None else load_prompt(COMPACTION_PROMPT)

    async def compact(self, messages: list[Message]) -> CompactionResult:
        """Summarise a run of adjacent turns. See `BaseCompactor.compact`.

        Raises:
            CompactionRefused: The model answered, but the answer must not stand in
                for these turns — truncated, empty, or no shorter than the input.
            LLMError: The provider call failed. Propagated unchanged; the adapter
                has already translated it out of vendor vocabulary, and this class
                has nothing to add.

        Both leave the caller's state untouched, which is what makes raising the
        safe choice: `GcService._age_out_turns` catches per turn and the turn stays
        young and verbatim for the next pass.
        """
        first_turn = messages[0].turn_index
        last_turn = messages[-1].turn_index
        original_token_count = sum(message.token_count for message in messages)

        floor = self._minimum_viable_run_tokens(first_turn, last_turn)
        if original_token_count <= floor:
            # Refused *before* the call, not after. The arithmetic below is certain,
            # so paying a provider for a summary we already know we will reject is
            # pure waste - and at scale it is the kind of waste that shows up as a
            # bill rather than as a bug.
            raise CompactionRefused(
                f"run of turns {first_turn}-{last_turn} is too small to compact "
                f"({original_token_count} <= {floor} tokens, the floor set by the provenance marker)",
                original_token_count=original_token_count,
            )

        max_output_tokens = self._output_cap(original_token_count)

        response = await self.provider.generate(
            messages=[self._as_request(messages)],
            system=self._system_prompt,
            max_output_tokens=max_output_tokens,
            timeout_s=self._timeout_s,
        )

        if not response.is_complete:
            # The call returned 200 and the text looks fine. This is the only place
            # that catches it, and the reason StopReason is normalised at all: a
            # summary that hit the output cap has lost the end of the run it was
            # summarising, and nothing downstream could ever tell.
            raise CompactionRefused(
                f"{response.provider} returned an incomplete summary "
                f"(stop_reason={response.stop_reason.value}, cap={max_output_tokens})",
                original_token_count=original_token_count,
            )

        summary_text = response.text.strip()
        if not summary_text:
            raise CompactionRefused(
                f"{response.provider} returned an empty summary",
                original_token_count=original_token_count,
            )

        marker = self._format_summary_marker(first_turn, last_turn)
        full_content = f"{marker} {summary_text}"
        summary_token_count = count_tokens(full_content)

        if summary_token_count >= original_token_count:
            # A compactor that expands is worse than no compactor: it costs a model
            # call, deletes the originals, and grows the prompt. Checked on the
            # marker-inclusive content because that is what the model will receive.
            raise CompactionRefused(
                f"summary of turns {first_turn}-{last_turn} is not shorter than the run "
                f"({summary_token_count} >= {original_token_count} tokens)",
                original_token_count=original_token_count,
                summary_token_count=summary_token_count,
            )

        summary_message = Message(
            role="assistant",
            content=full_content,
            token_count=summary_token_count,
            # The *first* turn of the run, so the composer can slot the summary
            # back into chronological order. Any other index scrambles the
            # conversation.
            turn_index=first_turn,
        )

        logger.info(
            "compacted a run with a model",
            extra={
                "turn_range": f"{first_turn}-{last_turn}",
                "original_token_count": original_token_count,
                "summary_token_count": summary_token_count,
                "provider": response.provider,
                "model": response.model,
            },
        )
        self.event_bus.emit(
            Event(
                event_type=EventType.MESSAGE_COMPACTED,
                data={
                    "compaction_strategy": self.COMPACTION_STRATEGY.value,
                    "turn_range_start": first_turn,
                    "turn_range_end": last_turn,
                    "original_token_count": original_token_count,
                    "token_count_after_compaction": summary_token_count,
                    "messages_in_run": len(messages),
                    # Only this compactor can report these, and a cost or latency
                    # view needs them per call. No payload text: event payloads
                    # carry no conversation content (CLAUDE.md §11).
                    "provider": response.provider,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        return CompactionResult(
            summary=summary_message,
            original_token_count=original_token_count,
            compaction_strategy=self.COMPACTION_STRATEGY,
        )

    def _minimum_viable_run_tokens(self, first_turn: int, last_turn: int) -> int:
        """The smallest run this compactor could possibly shrink.

        Every summary carries a fixed-size provenance marker, and the summary
        itself is capped at `target_ratio` of the run. So the best case is

            summary = marker + target_ratio * run

        and for that to beat the run at all:

            marker + target_ratio * run  <  run
            run  >  marker / (1 - target_ratio)

        With a ~15-token marker and a 0.35 ratio that floor is about 23 tokens. Any
        run at or below it is arithmetically guaranteed to be refused, whatever the
        model returns.

        This is deliberately *not* `GCConfig.min_compactable_tokens`, which the
        sweep strategy applies per *message*. The marker is amortised over the whole
        *run*, so two turns that each clear that floor can still form a run that
        cannot win - which is how a live trace came to make nine provider calls
        whose results were certain to be rejected. The quantity that decides this is
        the marker's size and the output ratio, and only this class knows both.
        """
        marker_tokens = count_tokens(self._format_summary_marker(first_turn, last_turn))
        if self._target_ratio >= 1.0:
            # A ratio of 1 or more permits a summary as long as its input, so no run
            # size is safe. Degenerate config; refuse everything rather than divide
            # by zero or go negative.
            return 10**9
        return int(marker_tokens / (1.0 - self._target_ratio))

    def _output_cap(self, original_token_count: int) -> int:
        """How many tokens the summary is allowed, derived from the run's size.

        A fraction of the input rather than a fixed number, because a fixed cap is
        either too tight for a long run (truncation, refusal, wasted call) or too
        loose for a short one (no saving). The floor keeps short runs viable.
        """
        return max(self._min_output_tokens, int(original_token_count * self._target_ratio))

    def _as_request(self, messages: list[Message]) -> Message:
        """Render the run into one delimited user message.

        The turns are deliberately *not* passed through as separate messages. Every
        character of them is untrusted third-party content, and as real messages a
        turn reading "ignore your instructions" would sit in the same structural
        position as a genuine instruction — the model has no way to tell which one
        we meant. Inside one clearly-marked block, with the system prompt saying
        that block is data, it is material to summarise instead.

        The cost accepted: the model reads labelled text rather than native roles,
        which is marginally worse for quality. Roles are preserved as labels so
        "the user decided" and "the assistant suggested" stay distinguishable,
        which the prompt asks for.
        """
        transcript = "\n".join(
            f"[turn {message.turn_index}] {message.role}: {message.content}" for message in messages
        )
        return Message(
            role="user",
            content=f"{TRANSCRIPT_OPEN}\n{transcript}\n{TRANSCRIPT_CLOSE}",
            turn_index=messages[0].turn_index,
        )
