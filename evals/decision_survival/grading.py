"""Asking the questions, grading the answers, and finding where each answer was in the prompt."""

import re
from collections.abc import Sequence
from enum import StrEnum

from evals.decision_survival.dataset import Probe, mentions
from llm_gc.engine.generations.front import FRONT_HEADER


class Verdict(StrEnum):
    RIGHT = "right"
    WRONG = "wrong"  # states a value the conversation did not settle on
    UNKNOWN = "unknown"  # says the conversation does not say
    UNCLEAR = "unclear"  # names the current value and a replaced one
    MISSING = "missing"  # no numbered line for this question


class Presence(StrEnum):
    VERBATIM = "verbatim"  # in a turn the model was sent word for word
    CONDENSED = "condensed"  # only in the condensed block at the start of the prompt
    ABSENT = "absent"


_NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.*\S)")
_ASIDE = re.compile(r"\([^()]*\)")
# Leading punctuation is skipped, so "**Unknown**" and "\"unknown\"" count.
_OPENS_UNKNOWN = re.compile(r"^\W*unknown(?!\w)", re.IGNORECASE)


def question_message(probes: Sequence[Probe]) -> str:
    """Every question in one message, so all of them see the same prompt.

    Asked one at a time, each request could trigger a clean-up before the next.
    """
    lines = [f"{number}. {probe.question}" for number, probe in enumerate(probes, start=1)]
    return (
        "Answer each question from our conversation above. Give only the current value in a few "
        "words, one line per question, numbered to match. If the conversation does not say, "
        'answer "unknown".\n\n' + "\n".join(lines)
    )


def split_answers(reply: str, count: int) -> list[str | None]:
    """The reply's answer to each of `count` numbered questions; None where it gave none."""
    answers: list[str | None] = [None] * count
    for line in reply.splitlines():
        match = _NUMBERED_LINE.match(line)
        if match and 1 <= (number := int(match.group(1))) <= count and answers[number - 1] is None:
            answers[number - 1] = match.group(2)
    return answers


def grade(answer: str | None, probe: Probe) -> Verdict:
    """The verdict on one answer, read without its parenthetical asides.

    "MySQL 8 (changed from Postgres)" answers MySQL; the aside is history, not a
    second answer. An answer that is only an aside is read whole. An answer that
    opens with "unknown" is unknown whatever follows: the values in its reasoning
    are what it declined to choose between, not its answer.
    """
    if answer is None:
        return Verdict.MISSING
    answer = _ASIDE.sub(" ", answer).strip() or answer
    if _OPENS_UNKNOWN.match(answer):
        return Verdict.UNKNOWN
    right = any(mentions(answer, value) for value in probe.accept)
    replaced = any(mentions(answer, value) for value in probe.reject)
    if right and replaced:
        return Verdict.UNCLEAR
    if right:
        return Verdict.RIGHT
    if not replaced and mentions(answer, "unknown"):
        return Verdict.UNKNOWN
    return Verdict.WRONG


def presence(probe: Probe, forwarded: Sequence[dict]) -> Presence:
    """Where an accepted answer was in the request the model received.

    `forwarded` is that request's `messages`; its last message is the questions
    themselves and is not searched.
    """
    found_in = [message for message in forwarded[:-1] if any(mentions(_text(message), value) for value in probe.accept)]
    if not found_in:
        return Presence.ABSENT
    if all(_text(message).startswith(FRONT_HEADER) for message in found_in):
        return Presence.CONDENSED
    return Presence.VERBATIM


def _text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
