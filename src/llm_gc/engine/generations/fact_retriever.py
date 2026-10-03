"""Choosing which stored facts are worth putting back into the prompt.

Permanent generation accumulates for the life of a session, so injecting all of it
would recreate the unbounded growth archiving exists to stop. See
doc/design/generational-memory/overview.md.
"""

from typing import TYPE_CHECKING, Protocol

from llm_gc.models import KnowledgeEntry
from llm_gc.utils.text import significant_words

# A topic label is the fact's subject, so matching it beats matching its prose.
_TOPIC_MATCH_WEIGHT = 2.0


class FactRetriever(Protocol):
    """Picks and ranks the stored facts relevant to a query."""

    name: str

    def retrieve(self, entries: list[KnowledgeEntry], query: str | None, limit: int) -> list[KnowledgeEntry]:
        """Return at most `limit` entries, most relevant first.

        Args:
            entries: Active candidates — supersession is already applied.
            query: Text to rank against, normally the turn about to be sent. None
                means there is nothing to rank against; implementations must still
                return something, since a caller with no query has not asked for
                no memory.
            limit: Hard cap. The caller applies a token budget on top.

        Returns:
            A ranked list, possibly empty. Order is the contract: the caller drops
            from the tail when its budget runs out.
        """
        ...


class LexicalFactRetriever:
    """Ranks facts by word overlap with the query.

    No model, no index, no startup cost. Extraction produces single-word topic
    labels, which word overlap matches well. Known miss: paraphrase — "which
    datastore?" will not match a fact stored under `database`.
    """

    name = "lexical"

    def retrieve(self, entries: list[KnowledgeEntry], query: str | None, limit: int) -> list[KnowledgeEntry]:
        """See `FactRetriever.retrieve`."""
        if limit <= 0 or not entries:
            return []

        if query is None:
            return self._most_recent(entries, limit)

        query_words = significant_words(query)
        if not query_words:
            # An all-stopword query ("so what about it?") gives nothing to rank by.
            return self._most_recent(entries, limit)

        scored = [(self._score(entry, query_words), entry) for entry in entries]
        # Zero-overlap facts are excluded, not ranked last: irrelevant "memory"
        # spends tokens and invites the model to use something off-topic.
        relevant = [(score, entry) for score, entry in scored if score > 0]
        # Ties go to the newer fact — later information usually still holds.
        relevant.sort(key=lambda pair: (pair[0], pair[1].message_turn), reverse=True)
        return [entry for _, entry in relevant[:limit]]

    @staticmethod
    def _score(entry: KnowledgeEntry, query_words: set[str]) -> float:
        topic_overlap = len(query_words & significant_words(entry.topic_label))
        content_overlap = len(query_words & significant_words(entry.content))
        return _TOPIC_MATCH_WEIGHT * topic_overlap + content_overlap

    @staticmethod
    def _most_recent(entries: list[KnowledgeEntry], limit: int) -> list[KnowledgeEntry]:
        """Fallback with no query: recency is the only ordering available."""
        return sorted(entries, key=lambda entry: entry.message_turn, reverse=True)[:limit]


if TYPE_CHECKING:
    _conformance: FactRetriever = LexicalFactRetriever()
