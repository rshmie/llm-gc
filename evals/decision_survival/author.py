"""Write the long conversations' assistant turns with a model, once, and freeze them into the dataset.

    ANTHROPIC_API_KEY=... python -m evals.decision_survival.author

long_scripts.json holds each conversation's user turns, the figure some answers must state, and its
probes. Every answer is checked before it is kept: one that leaves out its figure, or mentions another
probe's answer, is asked for again, so each answer is stated only where its probe says. Running this
again writes new text; the frozen dataset is what the check measures.
"""

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from evals.decision_survival.dataset import DATASET, load_conversations, mentions
from evals.decision_survival.harness import Effort
from evals.decision_survival.progress import Progress

SCRIPTS = Path(__file__).with_name("long_scripts.json")
DEFAULT_MODEL = "claude-sonnet-5-5"
ATTEMPTS = 3   # per answer, before the run stops and names the turn

PERSONA = ("You are an experienced software architect helping a product team design a system over a long "
           "conversation. Answer each message as you would in a detailed design review: concrete "
           "recommendations, trade-offs and specifics, with headings and lists where they help. "
           "Aim for 900 to 1,400 words.")

# The conversation so far, ending with the user's turn, and this answer's instructions -> the answer.
Answer = Callable[[list[dict[str, str]], str], str]


def author_conversation(script: dict[str, Any], answer: Answer, progress: Progress | None = None
                        ) -> dict[str, Any]:
    """The script with every assistant turn written and checked; raises naming a turn that could not be."""
    probes = script["probes"]
    turns: list[dict[str, str]] = []
    for pair in script["pairs"]:
        turns.append({"role": "user", "text": pair["user"]})
        index = len(turns)
        stated_here = [p for p in probes if p["established_at"] == index]
        allowed = stated_here + [p for p in probes if index in p.get("also_in", ())]
        forbidden = sorted({value for p in probes if p not in allowed for value in p["accept"]})
        instructions = _instructions(pair.get("must_state"), forbidden)
        for _ in range(ATTEMPTS):
            if progress is not None:
                progress.begin(f"{script['id']} turn {index}", "thinking…")
            text = answer(turns, instructions)
            problem = _problem(text, stated_here, forbidden)
            if progress is not None:
                progress.end(problem is None, f"{len(text.split()):,} words" if problem is None
                             else f"{problem}, asking again")
            if problem is None:
                break
        else:
            raise RuntimeError(f"{script['id']} turn {index}: {problem} after {ATTEMPTS} attempts")
        turns.append({"role": "assistant", "text": text})
    return {"id": script["id"], "turns": turns, "probes": probes}


def freeze(conversations: list[dict[str, Any]], dataset: Path = DATASET) -> None:
    """Add or replace these conversations in the dataset, which must still load afterwards."""
    data = json.loads(dataset.read_text())
    written = {conversation["id"] for conversation in conversations}
    data["conversations"] = [c for c in data["conversations"] if c["id"] not in written] + conversations
    # Written beside the dataset and validated before it replaces it, so a bad turn never lands.
    with tempfile.NamedTemporaryFile("w", dir=dataset.parent, suffix=".json", delete=False) as draft:
        draft.write(dump_dataset(data))
    try:
        load_conversations(Path(draft.name))
    except ValueError:
        os.unlink(draft.name)
        raise
    os.replace(draft.name, dataset)


def dump_dataset(data: dict[str, Any]) -> str:
    """The dataset in its own layout: one turn, and one probe, per line."""
    lines = ["{", '  "conversations": [']
    for c_index, conversation in enumerate(data["conversations"]):
        lines += ["    {", f'      "id": {json.dumps(conversation["id"])},', '      "turns": [']
        turns = conversation["turns"]
        for t_index, turn in enumerate(turns):
            comma = "," if t_index < len(turns) - 1 else ""
            lines.append(f'        {{"role": {json.dumps(turn["role"])}, '
                         f'"text": {json.dumps(turn["text"], ensure_ascii=False)}}}{comma}')
        lines += ["      ],", '      "probes": [']
        probes = conversation["probes"]
        for p_index, probe in enumerate(probes):
            fields = ", ".join(f"{json.dumps(key)}: {json.dumps(probe[key], ensure_ascii=False)}"
                               for key in ("id", "question", "accept", "reject", "established_at", "also_in")
                               if key in probe)
            lines.append(f"        {{{fields}}}" + ("," if p_index < len(probes) - 1 else ""))
        lines += ["      ]", "    }" + ("," if c_index < len(data["conversations"]) - 1 else "")]
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def _instructions(must_state: str | None, forbidden: list[str]) -> str:
    parts = []
    if must_state:
        parts.append(f"In this answer, make this recommendation explicitly, keeping its figure exactly: {must_state}")
    if forbidden:
        parts.append("Do not mention any of the following anywhere in this answer, not even in passing or as an "
                     f"example: {'; '.join(forbidden)}.")
    return " ".join(parts)


def _problem(text: str, stated_here: list[dict[str, Any]], forbidden: list[str]) -> str | None:
    if not text.strip():
        return "an empty answer"
    missing = [p["id"] for p in stated_here if not any(mentions(text, value) for value in p["accept"])]
    if missing:
        return f"leaves out {', '.join(missing)}"
    leaked = [value for value in forbidden if mentions(text, value)]
    return f"mentions {', '.join(leaked)}" if leaked else None


def _claude(model: str, effort: Effort, usage: dict[str, int], on_words: Callable[[int], None]) -> Answer:
    import anthropic
    from anthropic.types.beta import BetaOutputConfigParam

    client = anthropic.Anthropic()
    output_config: BetaOutputConfigParam = {"effort": effort}

    def answer(turns: list[dict[str, str]], instructions: str) -> str:
        messages: list[Any] = [{"role": turn["role"], "content": turn["text"]} for turn in turns[:-1]]
        # The breakpoint on the newest user turn caches the history; the per-turn instructions
        # come after it as a system message, so they never change the cached prefix.
        messages.append({"role": "user", "content": [
            {"type": "text", "text": turns[-1]["text"], "cache_control": {"type": "ephemeral"}}]})
        if instructions:
            messages.append({"role": "system", "content": instructions})
        with client.beta.messages.stream(model=model, max_tokens=16_000, system=PERSONA, messages=messages,
                                         output_config=output_config,
                                         betas=["server-side-fallback-2026-07-01"], fallbacks="default") as stream:
            words = 0
            for text in stream.text_stream:
                words += text.count(" ") + text.count("\n")
                on_words(words)
            message = stream.get_final_message()
        if message.stop_reason != "end_turn":
            raise RuntimeError(f"the answer stopped with {message.stop_reason!r}")
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            usage[key] = usage.get(key, 0) + (getattr(message.usage, key, 0) or 0)
        return "".join(block.text for block in message.content if block.type == "text")

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--only", action="append", help="write only this script id (repeatable)")
    args = parser.parse_args()

    scripts = [s for s in json.loads(SCRIPTS.read_text())["scripts"] if not args.only or s["id"] in args.only]
    usage: dict[str, int] = {}
    progress = Progress(total=sum(len(script["pairs"]) for script in scripts))
    answer = _claude(args.model, cast(Effort, args.effort), usage,
                     lambda words: progress.update(f"writing… {words:,} words"))
    print(f"{progress.total} answers to write, each a minute or so")
    conversations = [author_conversation(script, answer, progress) for script in scripts]
    freeze(conversations)
    print(f"\nwrote {', '.join(s['id'] for s in scripts)} into {DATASET.name}")
    print("tokens: " + ", ".join(f"{key} {value:,}" for key, value in usage.items()))


if __name__ == "__main__":
    main()
