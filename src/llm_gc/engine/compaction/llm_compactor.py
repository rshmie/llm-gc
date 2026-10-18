"""A compactor that summarises by calling a model.

Owns three things: how a run is presented to the model, how long its summary may
be, and when a returned summary must be refused. The HTTP call, wire format and error taxonomy
belong to the injected provider. See
doc/design/generational-memory/compaction.md.
"""

import logging
from collections.abc import Sequence

from llm_gc.config.constants import (
    DEFAULT_COMPACTION_TARGET_RATIO,
    DEFAULT_COMPACTION_THINKING_ALLOWANCE_TOKENS,
    DEFAULT_COMPACTION_TIMEOUT_S,
    DEFAULT_MIN_COMPACTION_OUTPUT_TOKENS,
)
from llm_gc.engine.compaction.base_compactor import BaseCompactor
from llm_gc.engine.compaction.compaction_result import CompactionResult
from llm_gc.engine.compaction.compaction_strategy import CompactionStrategy
from llm_gc.events import Event, EventBus, EventType
from llm_gc.exceptions import CompactionRefused
from llm_gc.llm.prompts import COMPACTION_PROMPT, load_prompt, transcript_block
from llm_gc.llm.provider import LLMProvider
from llm_gc.models import Message
from llm_gc.models.message import CLIENT_SPEAKER, TOOL_SPEAKER, parts_by_writer, speaker_label
from llm_gc.utils import count_tokens, count_tokens_cached

logger = logging.getLogger(__name__)

# Models follow a length in words far better than one in tokens; English prose
# runs at about three words to four tokens.
_WORDS_PER_TOKEN = 0.75

# How a part of a turn is labelled when someone other than the turn's speaker wrote it.
_WRITER_LABELS = {"client": CLIENT_SPEAKER, "tool": TOOL_SPEAKER}


class LLMCompactor(BaseCompactor):
    """Summarises a run of turns by calling a language model."""

    COMPACTION_STRATEGY: CompactionStrategy = CompactionStrategy.LLM

    def __init__(
        self,
        event_bus: EventBus,
        provider: LLMProvider,
        *,
        timeout_s: float = DEFAULT_COMPACTION_TIMEOUT_S,
        target_ratio: float = DEFAULT_COMPACTION_TARGET_RATIO,
        min_output_tokens: int = DEFAULT_MIN_COMPACTION_OUTPUT_TOKENS,
        thinking_allowance_tokens: int = DEFAULT_COMPACTION_THINKING_ALLOWANCE_TOKENS,
        system_prompt: str | None = None,
    ) -> None:
        """Build the compactor.

        Args:
            event_bus: Where `MESSAGE_COMPACTED` is announced.
            provider: Any `LLMProvider`, injected so this class needs no API key,
                HTTP client, or knowledge of which vendor is configured.
            timeout_s: Wall-clock budget for one summarisation call.
            target_ratio: The summary's target length as a fraction of the run's
                tokens, stated to the model in words.
            min_output_tokens: Floor under that target, so a short run still gets
                room for a sentence.
            thinking_allowance_tokens: Output budget beyond the target, for models
                that reason before answering.
            system_prompt: Override for the shipped prompt, for evals comparing
                wordings.
        """
        super().__init__(event_bus)
        self.provider = provider
        self._timeout_s = timeout_s
        self._target_ratio = target_ratio
        self._min_output_tokens = min_output_tokens
        self._thinking_allowance_tokens = thinking_allowance_tokens
        # Read once: a prompt cannot change under a running process.
        self._system_prompt = system_prompt if system_prompt is not None else load_prompt(COMPACTION_PROMPT)

    async def compact(self, messages: list[Message]) -> CompactionResult:
        """Summarise a run of adjacent turns. See `BaseCompactor.compact`.

        Raises:
            CompactionRefused: The summary must not stand in for these turns —
                the run is too small to beat the marker, or the answer came back
                truncated, empty, or no shorter than the input.
            LLMError: The provider call failed, propagated unchanged.
        """
        first_turn = messages[0].turn_index
        last_turn = messages[-1].turn_index
        original_token_count = sum(message.token_count for message in messages)

        floor = self._minimum_viable_run_tokens(first_turn, last_turn)
        if original_token_count <= floor:
            # Refused before the call: the arithmetic is already decided, so paying
            # a provider for a result we will reject is pure waste.
            raise CompactionRefused(
                f"run of turns {first_turn}-{last_turn} is too small to compact "
                f"({original_token_count} <= {floor} tokens, the floor set by the provenance marker)",
                original_token_count=original_token_count,
            )

        target_tokens = self._summary_target(original_token_count, first_turn, last_turn)
        max_output_tokens = target_tokens + self._thinking_allowance_tokens
        response = await self.provider.generate(
            messages=[self._as_request(messages)],
            system=f"{self._system_prompt}\n\nWrite at most {int(target_tokens * _WORDS_PER_TOKEN)} words.",
            max_output_tokens=max_output_tokens,
            timeout_s=self._timeout_s,
        )

        if not response.is_complete:
            # The call returned 200 and the text looks fine; only the stop reason
            # reveals that the summary lost the end of the run.
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

        full_content = f"{self._format_summary_marker(first_turn, last_turn)} {summary_text}"
        summary_token_count = count_tokens(full_content)

        if summary_token_count >= original_token_count:
            # Measured marker-inclusive, because that is what the model receives.
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
            # The run's first turn, so the composer re-interleaves chronologically.
            turn_index=first_turn,
        )

        # A summary is bought while a collection is prepared; whether and when it is applied is logged by the service.
        logger.debug(
            "summarised turns %d-%d: %d -> %d tokens",
            first_turn,
            last_turn,
            original_token_count,
            summary_token_count,
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
                    # Usage only this compactor can report. No payload text.
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
            summary_text=summary_text,
        )

    def worth_compacting(self, messages: Sequence[Message]) -> bool:
        """See `BaseCompactor.worth_compacting`: the run must beat the provenance marker."""
        run_tokens = sum(message.token_count for message in messages)
        return run_tokens > self._minimum_viable_run_tokens(messages[0].turn_index, messages[-1].turn_index)

    def _minimum_viable_run_tokens(self, first_turn: int, last_turn: int) -> int:
        """The smallest run this compactor could shrink.

        Best case is `summary = marker + target_ratio * run`, which beats the run
        only when `run > marker / (1 - target_ratio)`.
        """
        marker_tokens = count_tokens_cached(self._format_summary_marker(first_turn, last_turn))
        if self._target_ratio >= 1.0:
            # Degenerate config: no run size is safe, so refuse everything.
            return 10**9
        return int(marker_tokens / (1.0 - self._target_ratio))

    def _summary_target(self, original_token_count: int, first_turn: int, last_turn: int) -> int:
        """The summary's length in tokens: proportional to the run with a floor,
        and never more than fits inside the run once the marker is added."""
        room = original_token_count - count_tokens(self._format_summary_marker(first_turn, last_turn)) - 1
        return min(max(self._min_output_tokens, int(original_token_count * self._target_ratio)), room)

    def _as_request(self, messages: list[Message]) -> Message:
        """Render the run into one delimited user message.

        Not passed through as separate messages: every character is untrusted
        content, and as real messages a turn reading "ignore your instructions"
        would sit where a genuine instruction sits. Speakers survive as labels so
        attribution is preserved, and the turns' text is escaped so it cannot end
        the block or forge another turn's label. A turn more than one writer wrote is
        one line per part, each under its writer's label, so a skill's rules, a
        command's output or the results beside a person's words after an interrupt
        are never summarised as what the user said.
        """
        lines = (
            (message.turn_index, _WRITER_LABELS.get(part.writer, speaker_label(message)), part.text)
            for message in messages
            for part in parts_by_writer(message)
        )
        return Message(role="user", content=transcript_block(lines), turn_index=messages[0].turn_index)
