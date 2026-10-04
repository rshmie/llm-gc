"""The scripted conversations the check runs, and what each must still be able to answer.

Every turn is scripted, the assistant's included, so a run spends API calls only on
LLM-GC's own summaries and on the final questions. The loader validates the file:
a probe whose answer the conversation never states would be reported as a fact
LLM-GC lost, and the check exists to tell those apart.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATASET = Path(__file__).with_name("conversations.json")


@dataclass(frozen=True)
class Turn:
    role: str   # "user" or "assistant"
    text: str


@dataclass(frozen=True)
class Probe:
    """A question about the conversation, and how to tell a right answer from a wrong one."""

    id: str
    question: str
    accept: tuple[str, ...]          # a right answer names one of these
    established_at: int              # the turn that states the current value
    reject: tuple[str, ...] = ()     # values the conversation later replaced


@dataclass(frozen=True)
class Conversation:
    id: str
    turns: tuple[Turn, ...]
    probes: tuple[Probe, ...]


def load_conversations(path: Path = DATASET) -> list[Conversation]:
    """Read and validate the dataset; a malformed entry raises ValueError naming it."""
    raw = json.loads(path.read_text())
    conversations = [_conversation(item) for item in raw["conversations"]]
    ids = [conversation.id for conversation in conversations]
    if len(set(ids)) != len(ids):
        raise ValueError(f"conversation ids repeat: {ids}")
    return conversations


def mentions(text: str, phrase: str) -> bool:
    """Whether `text` contains `phrase` as whole words, ignoring case and spacing.

    Whole words, so an accepted "12" is not found inside "120".
    """
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, phrase.lower().split())) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def _conversation(item: dict) -> Conversation:
    conversation = Conversation(
        id=item["id"],
        turns=tuple(Turn(role=turn["role"], text=turn["text"]) for turn in item["turns"]),
        probes=tuple(Probe(id=probe["id"], question=probe["question"], accept=tuple(probe["accept"]),
                           established_at=probe["established_at"], reject=tuple(probe.get("reject", ())))
                     for probe in item["probes"]),
    )
    _validate(conversation)
    return conversation


def _validate(conversation: Conversation) -> None:
    where = f"conversation {conversation.id!r}"
    roles = [turn.role for turn in conversation.turns]
    # Alternating and ending on the assistant, so the questions arrive as the next user turn.
    if roles != ["user", "assistant"] * (len(roles) // 2) or not roles:
        raise ValueError(f"{where}: turns must alternate user, assistant and end with the assistant")
    if not conversation.probes:
        raise ValueError(f"{where}: no probes")
    for probe in conversation.probes:
        at = f"{where}, probe {probe.id!r}"
        if not 0 <= probe.established_at < len(conversation.turns):
            raise ValueError(f"{at}: established_at {probe.established_at} is not a turn")
        if not any(mentions(conversation.turns[probe.established_at].text, value) for value in probe.accept):
            raise ValueError(f"{at}: turn {probe.established_at} states none of {probe.accept}")
        if any(mentions(probe.question, value) for value in probe.accept):
            raise ValueError(f"{at}: the question gives away the answer")
