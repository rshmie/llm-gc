"""Authoring the long conversations, with a stand-in for the model."""

import io
import json

import pytest

from evals.decision_survival.author import SCRIPTS, author_conversation, dump_dataset, freeze
from evals.decision_survival.progress import Progress
from evals.decision_survival.dataset import DATASET, load_conversations

LONG_SCRIPTS = json.loads(SCRIPTS.read_text())["scripts"]


def _obliging(turns, instructions):
    """States the figure it is told to and nothing else of note."""
    figure = (
        instructions.split("keeping its figure exactly: ", 1)[1].split(" Do not mention", 1)[0]
        if "keeping its figure exactly" in instructions
        else ""
    )
    return f"Here is how I would approach it. {figure}".strip()


class TestAuthoring:
    @pytest.mark.parametrize("script", LONG_SCRIPTS, ids=[s["id"] for s in LONG_SCRIPTS])
    def test_each_script_authors_into_a_conversation_the_check_accepts(self, script, tmp_path):
        """The user turns and figures as written leak no answer and state each where its probe says."""
        dataset = tmp_path / "conversations.json"
        dataset.write_text(DATASET.read_text())

        freeze([author_conversation(script, _obliging)], dataset)

        conversation = next(c for c in load_conversations(dataset) if c.id == script["id"])
        assert len(conversation.turns) == 2 * len(script["pairs"])

    def test_an_answer_that_mentions_another_probes_answer_is_asked_for_again(self):
        script = LONG_SCRIPTS[0]
        calls = []

        def leaky_once(turns, instructions):
            calls.append(len(turns))
            if len(calls) == 1:
                return "Hold seats for 7 minutes, and by the way the platform fee is 4%."
            return _obliging(turns, instructions)

        conversation = author_conversation(script, leaky_once)

        assert calls[:2] == [1, 1], "the first answer was asked for twice"
        assert "4%" not in conversation["turns"][1]["text"]

    def test_an_answer_that_never_states_its_figure_stops_the_run_naming_the_turn(self):
        with pytest.raises(RuntimeError, match="turn 1: leaves out seat-hold"):
            author_conversation(LONG_SCRIPTS[0], lambda turns, instructions: "Some thoughts.")

    def test_the_instructions_name_the_figure_and_every_other_answer(self):
        seen = []
        author_conversation(LONG_SCRIPTS[0], lambda t, i: (seen.append(i), _obliging(t, i))[1])

        first = seen[0]
        assert "held for exactly 7 minutes" in first
        assert "aws" in first and "750" in first and "7 minutes" not in first.split("Do not mention", 1)[1]

    def test_the_reply_to_a_change_of_mind_may_repeat_it(self):
        """Saying "AWS" in the answer to "we're moving to AWS" is allowed, by the probe's also_in."""
        seen = []
        author_conversation(LONG_SCRIPTS[0], lambda t, i: (seen.append((len(t), i)), _obliging(t, i))[1])

        reply_to_change = next(instructions for turn_count, instructions in seen if turn_count == 17)
        assert "aws" not in reply_to_change.split("Do not mention", 1)[1].split(";")


def test_the_dataset_layout_round_trips():
    """freeze rewrites the whole file, so the existing conversations must survive byte for byte."""
    original = DATASET.read_text()

    assert dump_dataset(json.loads(original)) == original


def test_a_conversation_that_fails_validation_never_replaces_the_dataset(tmp_path):
    dataset = tmp_path / "conversations.json"
    dataset.write_text(DATASET.read_text())
    broken = {"id": "broken", "turns": [{"role": "user", "text": "hi"}], "probes": []}

    with pytest.raises(ValueError):
        freeze([broken], dataset)

    assert dataset.read_text() == DATASET.read_text()
    assert not [p for p in tmp_path.iterdir() if p.name != "conversations.json"], "no draft left behind"


class TestProgress:
    def test_each_answer_reports_how_many_are_done_and_the_time_left(self):
        out = io.StringIO()  # not a terminal: finished lines only, no redrawn live line
        script = LONG_SCRIPTS[0]

        author_conversation(script, _obliging, Progress(total=len(script["pairs"]), out=out))

        lines = out.getvalue().splitlines()
        assert len(lines) == len(script["pairs"])
        assert lines[0].strip().startswith(f"[1/{len(script['pairs'])}] {script['id']} turn 1:")
        assert all("left" in line for line in lines[:-1])
        assert lines[-1].endswith("all done")
        assert "\r" not in out.getvalue(), "a log file gets no terminal control characters"

    def test_a_retry_is_reported_as_one_and_not_counted_as_done(self):
        out = io.StringIO()
        answers = iter(["Hold seats for 7 minutes; the fee is 4%."])

        def leaky_once(turns, instructions):
            return next(answers, None) or _obliging(turns, instructions)

        author_conversation(LONG_SCRIPTS[0], leaky_once, Progress(total=15, out=out))

        first, second = out.getvalue().splitlines()[:2]
        assert first.strip().startswith("retry") and "mentions 4%" in first
        assert second.strip().startswith("[1/15]")
