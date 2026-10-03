from llm_gc.engine.generations import LexicalFactRetriever
from llm_gc.models import KnowledgeEntry
from llm_gc.models.knowledge_entry import KnowledgeType


def _entry(topic: str, content: str, turn: int = 1, kind: KnowledgeType = KnowledgeType.FACT) -> KnowledgeEntry:
    return KnowledgeEntry(message_turn=turn, content=content, topic_label=topic, knowledge_type=kind)


DATABASE = _entry("database", "postgres, and we are keeping ninety days of raw events", turn=2)
CACHE = _entry("cache", "redis, sitting in front of the read path", turn=4)
QUEUE = _entry("queue", "rabbitmq for now, though the platform team suggests kafka", turn=10)
ALL = [DATABASE, CACHE, QUEUE]


def _retrieve(entries=None, query=None, limit=5):
    return LexicalFactRetriever().retrieve(entries if entries is not None else ALL, query, limit)


class TestItFindsWhatTheQueryIsAbout:
    def test_the_topic_label_is_matched(self):
        assert _retrieve(query="Which database are we actually targeting?") == [DATABASE]

    def test_content_words_also_match(self):
        """The subject is not always named. "Are we still on rabbitmq?" never says
        "queue", but the fact's content does."""
        assert _retrieve(query="Are we still on rabbitmq?") == [QUEUE]

    def test_a_topic_match_outranks_a_content_match(self):
        """The topic label is the fact's subject, so a query word hitting it is
        stronger evidence than the same word appearing somewhere in prose."""
        mentions_cache_in_content = _entry("ttl", "short, because the cache goes stale", turn=1)

        ranked = _retrieve([mentions_cache_in_content, CACHE], query="what about the cache?")

        assert ranked[0] is CACHE

    def test_unrelated_facts_are_excluded_entirely(self):
        """Not merely ranked last. A prompt full of irrelevant "memory" is worse
        than a shorter one: it spends tokens and invites the model to use something
        that has nothing to do with the turn."""
        assert _retrieve(query="Which database are we targeting?") == [DATABASE]

    def test_nothing_relevant_returns_empty(self):
        assert _retrieve(query="What time is the standup tomorrow morning?") == []


class TestRanking:
    def test_more_overlap_ranks_higher(self):
        ranked = _retrieve(query="the database and the cache")
        assert ranked.index(DATABASE) < 2 and ranked.index(CACHE) < 2

    def test_a_tie_goes_to_the_more_recent_fact(self):
        """Later information about the same subject is usually the one that still
        holds."""
        old = _entry("database", "postgres", turn=2)
        new = _entry("database", "mysql", turn=40)

        assert _retrieve([old, new], query="which database?") == [new, old]

    def test_the_limit_is_respected(self):
        assert len(_retrieve(query="database cache queue", limit=2)) == 2

    def test_a_zero_limit_returns_nothing(self):
        assert _retrieve(query="database", limit=0) == []


class TestWhenThereIsNoQuery:
    def test_it_falls_back_to_the_most_recent_facts(self):
        """A caller with no query has not asked for no memory. Recency is the only
        ordering available, and a defensible one: a fact from turn 90 is more likely
        to still hold than one from turn 3."""
        assert _retrieve(query=None, limit=2) == [QUEUE, CACHE]

    def test_an_all_stopword_query_also_falls_back(self):
        """Ranking against nothing would return an arbitrary order dressed up as
        relevance."""
        assert _retrieve(query="so what about it then?", limit=2) == [QUEUE, CACHE]

    def test_an_empty_candidate_list_returns_empty(self):
        assert _retrieve([], query="database") == []


class TestRawEntriesParticipate:
    def test_a_raw_excerpt_can_be_retrieved(self):
        """A RAW entry is a whole turn kept verbatim because extraction found
        nothing in it. If it matches the query it is the most useful thing
        available - the actual text of a relevant turn."""
        raw = _entry("__raw__", "We walked through the ingest spike numbers last quarter", turn=6,
                     kind=KnowledgeType.RAW)

        assert _retrieve([raw, CACHE], query="what were the ingest spike numbers?") == [raw]
