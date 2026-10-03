"""Choosing which stored facts are worth putting back into the prompt.

Permanent generation accumulates for the life of a session. Injecting all of it
would recreate, in a new place, the unbounded growth that archiving exists to
stop — so something has to choose, and that choice is what this module is.

Retrieval sits behind a `Protocol` for the same reason providers and compactors
do: ranking quality is a measurable property that will be tuned against a
benchmark, and the component that *uses* the ranking should not have to change
when the ranking does.
"""

from typing import TYPE_CHECKING, Protocol

from llm_gc.models import KnowledgeEntry
from llm_gc.utils.text import significant_words

# A topic label is the fact's subject, so a query word matching it is stronger
# evidence than the same word appearing somewhere in the content. Two, rather than
# a larger number, because this is a tie-breaker and not a filter: a fact whose
# content matches three query words should still outrank one whose label matches
# one.
_TOPIC_MATCH_WEIGHT = 2.0


class FactRetriever(Protocol):
    """Picks and ranks the stored facts relevant to a query.

    A `Protocol`, not an ABC: implementations share no code, only a shape, and a
    library user with their own vector store should be able to supply one without
    inheriting from us. (Contrast `BaseCompactor`, which is an ABC because
    `_format_summary_marker` is real shared behaviour.)
    """

    name: str
    """Short identifier, so a caller holding several can report which ranked."""

    def retrieve(self, entries: list[KnowledgeEntry], query: str | None, limit: int) -> list[KnowledgeEntry]:
        """Return at most `limit` entries, most relevant first.

        Args:
            entries: The candidates — active entries only; supersession has
                already been applied by the caller, so a contradicted fact never
                reaches here.
            query: The text to rank against, normally the turn about to be sent.
                `None` means there is nothing to rank against, and an
                implementation must still return something sensible rather than
                nothing: a caller with no query has not asked for no memory.
            limit: Hard cap on how many to return. The caller applies a token
                budget on top, so returning fewer is always safe.

        Returns:
            A ranked list, possibly empty. Order is the contract — the caller
            drops from the tail when the token budget runs out, so the least
            relevant entry must be last.
        """
        ...


class LexicalFactRetriever:
    """Ranks facts by word overlap with the query.

    No model, no index, no startup cost. That is a deliberate trade and not a
    placeholder: extraction produces single-word topic labels like `database` and
    `cache`, and a query such as "which database are we targeting?" matches those
    exactly. An embedding retriever would additionally catch paraphrase — "which
    datastore?" will not match `database` here — which is a real and known miss.
    It is left for when there is a benchmark to measure the improvement against,
    because the alternative is paying a large dependency on the collect path for a
    gain nobody has measured.
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
            # The query was all stopwords ("so what about it?"). Ranking against
            # nothing would return an arbitrary order dressed up as relevance.
            return self._most_recent(entries, limit)

        scored = [(self._score(entry, query_words), entry) for entry in entries]
        # Only entries with some overlap. Returning zero-score facts would fill
        # the memory block with whatever happened to be stored, and a prompt full
        # of irrelevant "memory" is worse than a shorter one: it spends tokens and
        # invites the model to use something that has nothing to do with the turn.
        relevant = [(score, entry) for score, entry in scored if score > 0]

        # Sort by score descending, then by recency, so a tie goes to the newer
        # fact - later information about the same subject is usually the one that
        # still holds.
        relevant.sort(key=lambda pair: (pair[0], pair[1].message_turn), reverse=True)
        return [entry for _, entry in relevant[:limit]]

    @staticmethod
    def _score(entry: KnowledgeEntry, query_words: set[str]) -> float:
        topic_overlap = len(query_words & significant_words(entry.topic_label))
        content_overlap = len(query_words & significant_words(entry.content))
        return _TOPIC_MATCH_WEIGHT * topic_overlap + content_overlap

    @staticmethod
    def _most_recent(entries: list[KnowledgeEntry], limit: int) -> list[KnowledgeEntry]:
        """Fallback when there is nothing to rank against.

        Recency is the only ordering available without a query, and it is a
        defensible one: a fact extracted from turn 90 is more likely to still hold
        than one from turn 3. Returning an empty list instead would be the wrong
        reading of a missing query — the caller asked for memory and gave no hint,
        not for no memory.
        """
        return sorted(entries, key=lambda entry: entry.message_turn, reverse=True)[:limit]


if TYPE_CHECKING:
    # mypy verifies conformance; the block is erased at runtime. Same enforcement
    # the llm/ implementations carry, and for the same reason: a Protocol has none
    # of its own.
    _conformance: FactRetriever = LexicalFactRetriever()
