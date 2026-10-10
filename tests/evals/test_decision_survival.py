"""The decision-survival check's offline parts: the dataset, the grading, and presence."""

import json

import pytest

from evals.decision_survival.dataset import Probe, load_conversations, mentions
from evals.decision_survival.grading import Presence, Verdict, grade, presence, question_message, split_answers
from llm_gc.engine.generations.front import FRONT_HEADER

DATABASE = Probe(id="database", question="Which database?", accept=("mysql",), established_at=2,
                 reject=("postgres",))


def _dataset(tmp_path, **overrides):
    conversation = {
        "id": "c",
        "turns": [{"role": "user", "text": "the database is postgres"}, {"role": "assistant", "text": "noted"},
                  {"role": "user", "text": "the database is mysql now"}, {"role": "assistant", "text": "fine"}],
        "probes": [{"id": "database", "question": "Which database?", "accept": ["mysql"], "established_at": 2}],
        **overrides,
    }
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps({"conversations": [conversation]}))
    return path


class TestTheDataset:
    def test_the_shipped_conversations_load_and_state_every_answer_where_they_say(self):
        conversations = load_conversations()
        assert conversations
        for conversation in conversations:
            for probe in conversation.probes:
                stated = [i for i, turn in enumerate(conversation.turns)
                          if any(mentions(turn.text, value) for value in probe.accept)]
                # Only the turn that states it, and any turn the probe names as allowed to repeat it.
                assert probe.established_at in stated, (conversation.id, probe.id)
                assert set(stated) <= {probe.established_at, *probe.also_in}, (conversation.id, probe.id, stated)

    def test_a_probe_whose_answer_the_conversation_never_states_is_rejected(self, tmp_path):
        """It would be reported as a fact LLM-GC lost - the confusion the check exists to remove."""
        path = _dataset(tmp_path, probes=[{"id": "p", "question": "Which?", "accept": ["sqlite"], "established_at": 2}])
        with pytest.raises(ValueError, match="states none of"):
            load_conversations(path)

    def test_a_question_that_gives_away_its_answer_is_rejected(self, tmp_path):
        path = _dataset(tmp_path, probes=[{"id": "p", "question": "Is it mysql?", "accept": ["mysql"], "established_at": 2}])
        with pytest.raises(ValueError, match="gives away"):
            load_conversations(path)

    def test_turns_must_end_with_the_assistant(self, tmp_path):
        path = _dataset(tmp_path, turns=[{"role": "user", "text": "the database is mysql"}])
        with pytest.raises(ValueError, match="alternate"):
            load_conversations(path)


class TestMentions:
    def test_whole_words_only(self):
        assert mentions("It has 12 partitions.", "12")
        assert not mentions("It has 120 partitions.", "12")

    def test_case_and_spacing_do_not_matter(self):
        assert mentions("Ships on November   14th", "november 14th")
        assert mentions("MySQL 8", "mysql")


class TestAnswers:
    def test_each_numbered_line_answers_its_question(self):
        reply = "Sure:\n1. MySQL 8\n2) 30 days\n\n4: Priya\n9. stray"
        assert split_answers(reply, 4) == ["MySQL 8", "30 days", None, "Priya"]

    @pytest.mark.parametrize("answer, verdict", [
        ("MySQL 8", Verdict.RIGHT),
        ("Postgres 16", Verdict.WRONG),
        ("MySQL, after starting on Postgres", Verdict.UNCLEAR),
        ("unknown", Verdict.UNKNOWN),
        ("Postgres, though that is unknown", Verdict.WRONG),
        (None, Verdict.MISSING),
    ])
    def test_a_wrong_value_and_an_honest_unknown_are_different_verdicts(self, answer, verdict):
        assert grade(answer, DATABASE) is verdict

    @pytest.mark.parametrize("answer, verdict", [
        ("MySQL 8 (changed from Postgres 16)", Verdict.RIGHT),
        ("Postgres 16 (later MySQL)", Verdict.WRONG),
        ("MySQL or Postgres (not settled)", Verdict.UNCLEAR),
        ("(MySQL 8)", Verdict.RIGHT),
    ])
    def test_a_parenthetical_aside_is_history_not_a_second_answer(self, answer, verdict):
        """Models add "(changed from X)" even when asked for the current value only;
        graded as a second answer, a right one was scored unclear in both arms."""
        assert grade(answer, DATABASE) is verdict

    @pytest.mark.parametrize("answer, verdict", [
        ("Unknown. The user proposed MySQL 8, but no final decision was stated.", Verdict.UNKNOWN),
        ("**Unknown** - MySQL 8 was only suggested", Verdict.UNKNOWN),
        ("MySQL 8; the minor version is unknown", Verdict.RIGHT),
        ("Unknowns remain, but MySQL 8", Verdict.RIGHT),
    ])
    def test_an_answer_that_opens_with_unknown_is_unknown(self, answer, verdict):
        """The model declined to choose; naming the right value in its reasoning is not
        answering with it. Graded right, one such reply hid a loss behind 19/19."""
        assert grade(answer, DATABASE) is verdict

    def test_the_questions_go_in_one_numbered_message(self):
        message = question_message([DATABASE, Probe(id="r", question="Retention?", accept=("30 days",), established_at=0)])
        assert message.endswith("1. Which database?\n2. Retention?")


class TestPresence:
    QUESTIONS = {"role": "user", "content": "1. Which database?"}

    def test_found_in_a_turn_sent_word_for_word(self):
        forwarded = [{"role": "user", "content": [{"type": "text", "text": "the database is MySQL now"}]}, self.QUESTIONS]
        assert presence(DATABASE, forwarded) is Presence.VERBATIM

    def test_found_only_in_the_condensed_block(self):
        forwarded = [{"role": "user", "content": f"{FRONT_HEADER}\n- (turn 2) database: mysql"},
                     {"role": "assistant", "content": "later reply"}, self.QUESTIONS]
        assert presence(DATABASE, forwarded) is Presence.CONDENSED

    def test_the_question_itself_is_not_searched(self):
        forwarded = [{"role": "assistant", "content": "later reply"}, {"role": "user", "content": "is it mysql?"}]
        assert presence(DATABASE, forwarded) is Presence.ABSENT
