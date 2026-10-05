import pytest

from llm_gc.models import Message
from llm_gc.models.generation_snapshot import file_knowledge
from llm_gc.models.knowledge_entry import KnowledgeStatus, KnowledgeType
from llm_gc.extraction import KnowledgeExtractor


def _make_message(content: str) -> Message:
    return Message(role="user", content=content, token_count=10, turn_index=1)


def _make_extractor() -> KnowledgeExtractor:
    return KnowledgeExtractor()


class TestFactExtraction:
    def test_extracts_fact_with_is_pattern(self):
        extractor = _make_extractor()
        message = _make_message("The database is PostgreSQL")

        entries = extractor.extract_knowledge(message, message_turn=5)

        fact_entries = [e for e in entries if e.knowledge_type == KnowledgeType.FACT]
        assert len(fact_entries) >= 1
        assert any(e.content == "PostgreSQL" and "database" in e.topic_label for e in fact_entries)

    def test_extracts_fact_with_using_for_pattern(self):
        extractor = _make_extractor()
        message = _make_message("We're using Redis for caching")

        entries = extractor.extract_knowledge(message, message_turn=3)

        fact_entries = [e for e in entries if e.knowledge_type == KnowledgeType.FACT]
        assert len(fact_entries) >= 1
        assert any(e.content == "Redis" and "caching" in e.topic_label for e in fact_entries)

    def test_extracts_fact_with_runs_on_pattern(self):
        extractor = _make_extractor()
        message = _make_message("The API runs on port 8080")

        entries = extractor.extract_knowledge(message, message_turn=2)

        fact_entries = [e for e in entries if e.knowledge_type == KnowledgeType.FACT]
        assert len(fact_entries) >= 1
        assert any("8080" in e.content for e in fact_entries)

    def test_fact_entry_has_correct_message_turn(self):
        extractor = _make_extractor()
        message = _make_message("The database is PostgreSQL")

        entries = extractor.extract_knowledge(message, message_turn=7)

        assert all(e.message_turn == 7 for e in entries)


class TestDecisionExtraction:
    def test_extracts_decision_with_decided_to_pattern(self):
        extractor = _make_extractor()
        message = _make_message("We decided to use JWT for authentication")

        entries = extractor.extract_knowledge(message, message_turn=4)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any("JWT" in e.content for e in decision_entries)

    def test_extracts_decision_with_lets_go_with_pattern(self):
        extractor = _make_extractor()
        message = _make_message("Let's go with Redis for caching")

        entries = extractor.extract_knowledge(message, message_turn=6)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any("Redis" in e.content for e in decision_entries)

    def test_extracts_decision_with_chose_over_pattern(self):
        extractor = _make_extractor()
        message = _make_message("We chose microservices over monolith for architecture")

        entries = extractor.extract_knowledge(message, message_turn=9)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any("microservices" in e.content for e in decision_entries)

    def test_decision_without_topic_falls_back_to_general(self):
        extractor = _make_extractor()
        message = _make_message("We decided to use microservices")

        entries = extractor.extract_knowledge(message, message_turn=4)

        decision_entries = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]
        assert len(decision_entries) >= 1
        assert any(e.topic_label == "general" for e in decision_entries)


class TestPreferenceExtraction:
    def test_extracts_preference_with_prefer_pattern(self):
        extractor = _make_extractor()
        message = _make_message("I prefer tabs for indentation")

        entries = extractor.extract_knowledge(message, message_turn=2)

        pref_entries = [e for e in entries if e.knowledge_type == KnowledgeType.PREFERENCE]
        assert len(pref_entries) >= 1
        assert any("tabs" in e.content for e in pref_entries)

    def test_extracts_preference_with_rather_use_pattern(self):
        extractor = _make_extractor()
        message = _make_message("I'd rather use dataclasses for models")

        entries = extractor.extract_knowledge(message, message_turn=3)

        pref_entries = [e for e in entries if e.knowledge_type == KnowledgeType.PREFERENCE]
        assert len(pref_entries) >= 1
        assert any("dataclasses" in e.content for e in pref_entries)

    def test_preference_without_topic_falls_back_to_general(self):
        extractor = _make_extractor()
        message = _make_message("We prefer simplicity")

        entries = extractor.extract_knowledge(message, message_turn=1)

        pref_entries = [e for e in entries if e.knowledge_type == KnowledgeType.PREFERENCE]
        assert len(pref_entries) >= 1
        assert any(e.topic_label == "general" for e in pref_entries)


class TestMultipleEntries:
    def test_extracts_multiple_types_from_one_message(self):
        extractor = _make_extractor()
        message = _make_message("The database is PostgreSQL. We decided to use JWT for auth")

        entries = extractor.extract_knowledge(message, message_turn=5)

        types_found = {e.knowledge_type for e in entries}
        assert KnowledgeType.FACT in types_found
        assert KnowledgeType.DECISION in types_found


class TestNoMatch:
    def test_returns_empty_list_for_unrecognized_text(self):
        extractor = _make_extractor()
        message = _make_message("Thanks for the help yesterday")

        entries = extractor.extract_knowledge(message, message_turn=1)

        assert entries == []

    def test_returns_empty_list_for_empty_content(self):
        extractor = _make_extractor()
        message = _make_message("")

        entries = extractor.extract_knowledge(message, message_turn=1)

        assert entries == []


class TestTopicLabelsAreBounded:
    """The defect that put hundred-character "topics" into permanent generation.

    `[\\w\\s]+?` matched word characters *and whitespace*, so a topic could span an
    unbounded number of words and was bounded only by a full stop happening to
    appear. Topics are now counted in words.
    """

    def test_a_topic_is_at_most_four_words(self):
        extractor = _make_extractor()
        message = _make_message(
            "Alpha beta gamma delta epsilon zeta eta theta is the value we settled on"
        )

        entries = extractor.extract_knowledge(message, message_turn=1)

        for entry in entries:
            assert len(entry.topic_label.split()) <= 4, entry.topic_label

    def test_a_mid_sentence_clause_is_not_mistaken_for_a_topic(self):
        """The real example from a live run: "A timeout in teardown usually means a
        connection is being held open past the test" previously stored the topic
        'A timeout in teardown usually means a connection'. Nothing matches now, so
        the caller archives the turn verbatim instead — which claims nothing."""
        extractor = _make_extractor()
        message = _make_message(
            "A timeout in teardown usually means a connection is being held open past the test."
        )

        entries = extractor.extract_knowledge(message, message_turn=1)

        assert entries == []

    def test_the_subject_must_be_at_the_start_of_a_sentence(self):
        """The declarative subject sits at the sentence start. An "is" further in
        belongs to a subordinate clause, and what precedes it is a fragment."""
        extractor = _make_extractor()
        at_start = extractor.extract_knowledge(_make_message("The cache is redis."), message_turn=1)
        buried = extractor.extract_knowledge(
            _make_message("I think that whatever we end up choosing the cache is redis."), message_turn=1
        )

        assert any(e.topic_label == "cache" for e in at_start)
        assert buried == []


class TestTopicLabelsAreNormalised:
    def test_a_topic_never_carries_surrounding_whitespace(self):
        """`content` was stripped and `topic_label` was not, which is exactly where
        the leading space in the observed `' What'` came from."""
        extractor = _make_extractor()
        for text in ("The database is postgres", "we are using redis for   the cache  "):
            for entry in extractor.extract_knowledge(_make_message(text), message_turn=1):
                assert entry.topic_label == entry.topic_label.strip()

    def test_a_topic_is_lowercased_so_supersession_can_match(self):
        """Supersession compares topic labels exactly. Without normalisation,
        "the Database is X" and "the database is Y" are two unrelated active facts
        instead of one contradiction."""
        extractor = _make_extractor()
        upper = extractor.extract_knowledge(_make_message("The Database is postgres"), message_turn=1)
        lower = extractor.extract_knowledge(_make_message("the database is mysql"), message_turn=2)

        assert {e.topic_label for e in upper} & {e.topic_label for e in lower}

    def test_internal_whitespace_is_collapsed(self):
        extractor = _make_extractor()
        entries = extractor.extract_knowledge(
            _make_message("The  read   path is redis backed"), message_turn=1
        )

        for entry in entries:
            assert "  " not in entry.topic_label


class TestMeaninglessTopicsAreRejected:
    """Refusing to label is better than labelling wrongly.

    A rejected match means the archiving caller stores the turn verbatim as a RAW
    entry, so nothing is lost. A *stored* junk label is worse than nothing: because
    supersession matches on the label, it can never be contradicted and sits active
    forever.
    """

    def test_an_interrogative_subject_is_rejected(self):
        """The observed `' What'` case: "What are the constraints we are working
        within on throughput and retention?" named nothing retrievable."""
        extractor = _make_extractor()
        entries = extractor.extract_knowledge(
            _make_message("What are the constraints we are working within on throughput?"), message_turn=1
        )

        assert entries == []

    def test_a_pronoun_subject_is_rejected(self):
        extractor = _make_extractor()
        for text in ("It is fine.", "They are ready.", "That is interesting."):
            assert extractor.extract_knowledge(_make_message(text), message_turn=1) == []

    def test_a_real_subject_with_a_leading_article_still_extracts(self):
        """The stopword guard rejects topics that are *only* stopwords. "the
        database" must still work, or the guard has eaten the common case."""
        extractor = _make_extractor()
        entries = extractor.extract_knowledge(_make_message("The database is postgres"), message_turn=1)

        assert any(e.topic_label == "database" for e in entries)

    def test_a_fragment_content_is_rejected(self):
        extractor = _make_extractor()
        assert extractor.extract_knowledge(_make_message("The queue is x"), message_turn=1) == []


class TestMatchingIsPerSentence:
    """Matching ran over the whole message, so an "is" anywhere in two hundred
    words matched and the topic capture reached back toward the start."""

    def test_a_fact_in_a_later_sentence_is_found(self):
        extractor = _make_extractor()
        message = _make_message(
            "We walked through the numbers this morning. The database is postgres."
        )

        entries = extractor.extract_knowledge(message, message_turn=1)

        assert any(e.topic_label == "database" and "postgres" in e.content for e in entries)

    def test_content_stops_at_the_sentence_end(self):
        """Every DECISION and PREFERENCE pattern ends with `$`. Against a whole
        message that meant end-of-message, so the content capture swallowed every
        following sentence."""
        extractor = _make_extractor()
        message = _make_message("We decided to ship on Friday. Then we went home.")

        entries = extractor.extract_knowledge(message, message_turn=1)
        decisions = [e for e in entries if e.knowledge_type == KnowledgeType.DECISION]

        assert decisions
        assert all("went home" not in e.content for e in decisions)

    def test_a_subject_does_not_reach_across_a_sentence_boundary(self):
        extractor = _make_extractor()
        message = _make_message("Understood. What are the constraints on throughput?")

        for entry in extractor.extract_knowledge(message, message_turn=1):
            assert "Understood" not in entry.topic_label

    def test_two_sentences_each_stating_a_fact_both_extract(self):
        extractor = _make_extractor()
        message = _make_message("The database is postgres. The cache is redis.")

        topics = {e.topic_label for e in extractor.extract_knowledge(message, message_turn=1)}

        assert {"database", "cache"} <= topics


class TestOneEntryPerTypePerSentence:
    def test_alternative_phrasings_of_one_fact_do_not_both_file(self):
        """The patterns within a type are alternative phrasings of the same idea, so
        two matching means two descriptions of one fact. Storing both would make the
        later one supersede the earlier for no reason."""
        extractor = _make_extractor()
        message = _make_message("The api is using redis for the cache")

        facts = [
            e
            for e in extractor.extract_knowledge(message, message_turn=1)
            if e.knowledge_type == KnowledgeType.FACT
        ]

        assert len(facts) == 1


class TestSentencesAsPeopleWriteThem:
    """Real sentences end in a full stop and often open with "OK," or "Decision:".
    The tests above mostly used neither, and every one of these failed in use."""

    @pytest.mark.parametrize("text, topic, content", [
        ("We decided to use Postgres for the database.", "database", "use Postgres"),
        ("Let's go with Redis for caching.", "caching", "Redis"),
        ("I prefer tabs for indentation!", "indentation", "tabs"),
    ])
    def test_a_trailing_topic_is_read_through_the_full_stop(self, text, topic, content):
        entries = _make_extractor().extract_knowledge(_make_message(text), message_turn=1)
        assert [(e.topic_label, e.content) for e in entries] == [(topic, content)]

    @pytest.mark.parametrize("text", [
        "OK, the database is postgres.",
        "Decision: the database is postgres.",
        "OK, decision: the database is postgres.",
        "So, the database is postgres!",
    ])
    def test_an_opener_or_a_label_does_not_hide_the_subject(self, text):
        entries = _make_extractor().extract_knowledge(_make_message(text), message_turn=1)
        assert [(e.topic_label, e.content) for e in entries] == [("database", "postgres")]

    def test_a_qualifying_lead_in_is_not_stripped(self):
        """"In production" narrows the claim; dropping it would file a fact the
        conversation never stated, so the sentence is refused instead."""
        message = _make_message("In production, the database is postgres.")
        assert _make_extractor().extract_knowledge(message, message_turn=1) == []

    def test_a_colon_inside_a_value_is_not_a_label(self):
        entries = _make_extractor().extract_knowledge(_make_message("The deadline is 10:30 on Friday."), 1)
        assert [(e.topic_label, e.content) for e in entries] == [("deadline", "10:30 on Friday")]

    @pytest.mark.parametrize("text", ["The database is postgres, right?", "Should we go with Postgres for the database?"])
    def test_a_question_states_nothing(self, text):
        assert _make_extractor().extract_knowledge(_make_message(text), message_turn=1) == []

    def test_a_trailing_topic_is_not_given_to_a_compound_decision(self):
        """The topic slot would catch "cache" for a decision that is mostly about the
        database; a later cache decision would then retire the database one."""
        message = _make_message("We decided to use Postgres for the database, and Redis for the cache.")
        [entry] = _make_extractor().extract_knowledge(message, message_turn=1)
        assert entry.topic_label == "general"

    def test_a_decision_and_a_fact_about_one_thing_share_a_label(self):
        extractor = _make_extractor()
        decision = extractor.extract_knowledge(_make_message("We decided to use Postgres for the database."), 1)
        fact = extractor.extract_knowledge(_make_message("The database is mysql."), 2)
        assert decision[0].topic_label == fact[0].topic_label == "database"

    def test_a_change_of_mind_supersedes_and_unrelated_decisions_do_not(self):
        extractor = _make_extractor()
        said = ["OK, decision: the database is postgres.", "We decided to ship on Friday.",
                "We decided to use microservices.", "Actually, change of plan: the database is mysql."]
        entries = [entry for turn, text in enumerate(said)
                   for entry in extractor.extract_knowledge(_make_message(text), message_turn=turn)]

        filed, _ = file_knowledge((), tuple(entries))

        active = {e.content for e in filed if e.knowledge_status is KnowledgeStatus.ACTIVE}
        assert active == {"ship on Friday", "use microservices", "mysql"}
