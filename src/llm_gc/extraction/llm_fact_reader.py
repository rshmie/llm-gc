"""Facts read by a model: what a collection's turns settled, filed under the topic they change.

One call reads every span a collection makes, with the facts already on record, so
a short "change of plan" exchange that is never summarised still replaces the value
it changes, under the topic that value was filed with. The model's answer is
untrusted output: each fact is validated, and one that names a turn outside the
spans, or is not a short topic and value, is dropped. See
doc/design/generational-memory/overview.md.
"""

import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ValidationError

from llm_gc.config.constants import (
    DEFAULT_FACTS_MAX_BATCH_TOKENS,
    DEFAULT_FACTS_ON_RECORD_MAX,
    DEFAULT_FACTS_OUTPUT_TOKENS,
    DEFAULT_FACTS_THINKING_ALLOWANCE_TOKENS,
    DEFAULT_FACTS_TIMEOUT_S,
    FACT_TOPIC_MAX_CHARS,
    FACT_TOPIC_MAX_WORDS,
    FACT_VALUE_MAX_CHARS,
)
from llm_gc.exceptions import LLMResponseError
from llm_gc.llm.prompts import FACTS_CLOSE, FACTS_OPEN, FACTS_PROMPT, escape_delimiters, load_prompt, transcript_block
from llm_gc.llm.provider import LLMProvider, LLMResponse
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.generation_snapshot import file_knowledge
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType
from llm_gc.models.message import person_text, said_by_user, speaker_label

logger = logging.getLogger(__name__)


class _Fact(BaseModel):
    turn: int
    topic: str
    value: str
    kind: str = KnowledgeType.FACT.value


class _Answer(BaseModel):
    facts: list[_Fact]


class LLMFactReader:
    """Reads a collection's facts by calling a language model. See `FactReader`."""

    speakers: frozenset[Literal["user", "assistant"]] = frozenset({"user", "assistant"})

    def __init__(self, provider: LLMProvider, *, timeout_s: float = DEFAULT_FACTS_TIMEOUT_S,
                 output_tokens: int = DEFAULT_FACTS_OUTPUT_TOKENS,
                 thinking_allowance_tokens: int = DEFAULT_FACTS_THINKING_ALLOWANCE_TOKENS,
                 max_batch_tokens: int = DEFAULT_FACTS_MAX_BATCH_TOKENS,
                 on_record_max: int = DEFAULT_FACTS_ON_RECORD_MAX, system_prompt: str | None = None) -> None:
        """Build the reader.

        Args:
            provider: Any `LLMProvider`; the one that writes the summaries, normally.
            timeout_s: Wall-clock budget for one call.
            output_tokens: Room for the list of facts itself.
            thinking_allowance_tokens: Output budget beyond the list, for models that
                reason before answering.
            max_batch_tokens: Turns read per call. A larger collection is read in
                several calls, each seeing the facts the ones before it read.
            on_record_max: Facts on record shown with each call, the newest kept.
            system_prompt: Override for the shipped prompt, for evals comparing wordings.
        """
        self.provider = provider
        self._timeout_s = timeout_s
        self._max_output_tokens = output_tokens + thinking_allowance_tokens
        self._max_batch_tokens = max_batch_tokens
        self._on_record_max = on_record_max
        self._system_prompt = system_prompt if system_prompt is not None else load_prompt(FACTS_PROMPT)

    async def read(self, spans: Sequence[Sequence[Message]], on_record: Sequence[KnowledgeEntry]
                   ) -> list[KnowledgeEntry]:
        """See `FactReader.read`.

        Raises:
            LLMResponseError: An answer that was cut off, or is not the expected JSON.
            LLMError: The provider call failed, propagated unchanged.
        """
        read: list[KnowledgeEntry] = []
        record = [fact for fact in on_record if fact.knowledge_status is KnowledgeStatus.ACTIVE]
        for batch in self._batches(spans):
            facts = await self._read_batch(batch, record)
            read += facts
            filed, _ = file_knowledge(tuple(record), tuple(facts))
            record = [fact for fact in filed if fact.knowledge_status is KnowledgeStatus.ACTIVE]
        return read

    def _batches(self, spans: Sequence[Sequence[Message]]) -> list[list[Message]]:
        """The spans' turns in calls of at most `max_batch_tokens`, never splitting a span."""
        batches: list[list[Message]] = []
        size = 0
        for span in spans:
            span_tokens = sum(turn.token_count for turn in span)
            if not batches or size + span_tokens > self._max_batch_tokens:
                batches.append([])
                size = 0
            batches[-1] += span
            size += span_tokens
        return [batch for batch in batches if batch]

    async def _read_batch(self, turns: list[Message], record: list[KnowledgeEntry]) -> list[KnowledgeEntry]:
        response = await self.provider.generate(messages=[self._as_request(turns, record)],
                                                system=self._system_prompt,
                                                max_output_tokens=self._max_output_tokens, timeout_s=self._timeout_s)
        if not response.is_complete:
            # A list cut short can still parse, and would silently lose its last facts.
            raise LLMResponseError(f"the facts answer was cut off (stop_reason={response.stop_reason.value})",
                                   provider=response.provider)
        return self._facts_in(response, {turn.turn_index: turn for turn in turns})

    def _as_request(self, turns: list[Message], record: list[KnowledgeEntry]) -> Message:
        """The facts on record and the turns, each in its own delimited block of data.

        The facts on record were read from the conversation too, so they are as
        untrusted as the turns, and are delimited and escaped the same way.

        A turn the person wrote is shown as their words alone, without what the client
        wrote into it: a fact's speaker comes from the turn it names, so the turn must
        hold only the speaker's words. A value only the client wrote becomes a fact
        when a later turn settles on it, as a value in tool output does.
        """
        shown = record[-self._on_record_max:]
        facts = "\n".join(escape_delimiters(f"{fact.topic_label}: {fact.content}") for fact in shown) or "(none yet)"
        transcript = transcript_block(
            (turn.turn_index, speaker_label(turn), person_text(turn) if said_by_user(turn) else turn.content)
            for turn in turns)
        return Message(role="user", turn_index=turns[0].turn_index,
                       content=f"{FACTS_OPEN}\n{facts}\n{FACTS_CLOSE}\n\n{transcript}")

    def _facts_in(self, response: LLMResponse, turns: dict[int, Message]) -> list[KnowledgeEntry]:
        """The answer's valid facts. Malformed JSON fails the read; a single bad fact is dropped."""
        text = response.text
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end < start:
            raise LLMResponseError("the facts answer held no JSON object", provider=response.provider)
        try:
            # From the first brace to the last, so a code fence or a stray word around it does not fail the read.
            answer = _Answer.model_validate_json(text[start:end + 1])
        except ValidationError as err:
            raise LLMResponseError("the facts answer did not match the expected shape",
                                   provider=response.provider) from err
        facts = [entry for fact in answer.facts if (entry := _entry(fact, turns)) is not None]
        if dropped := len(answer.facts) - len(facts):
            logger.info("facts dropped from a model's answer", extra={"dropped": dropped, "kept": len(facts)})
        # Filed in the order they were said, whatever order the model listed them in:
        # filing is sequential, and a later statement is what replaces an earlier one.
        return sorted(facts, key=lambda fact: fact.message_turn)


def _entry(fact: _Fact, turns: dict[int, Message]) -> KnowledgeEntry | None:
    """A validated fact, or None when it names a turn it was not given or is not a short topic and value.

    Its speaker is taken from the turn, never from the model: the user when the turn
    is something they wrote, the assistant for its replies, for tool results, for
    another agent's relayed report and for a turn the client wrote.
    """
    topic = " ".join(fact.topic.split()).strip(" .:").casefold()
    value = " ".join(fact.value.split()).strip()
    if (fact.turn not in turns or not topic or len(topic.split()) > FACT_TOPIC_MAX_WORDS
            or len(topic) > FACT_TOPIC_MAX_CHARS or not value or len(value) > FACT_VALUE_MAX_CHARS):
        return None
    kind = next((kind for kind in KnowledgeType if kind.value == fact.kind.casefold()), KnowledgeType.FACT)
    turn = turns[fact.turn]
    speaker: Literal["user", "assistant"] = "user" if said_by_user(turn) else "assistant"
    return KnowledgeEntry(message_turn=fact.turn, content=value, topic_label=topic, knowledge_type=kind, speaker=speaker)
