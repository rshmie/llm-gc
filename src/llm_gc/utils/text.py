"""Word-level helpers for components that compare English text."""

import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        # Articles, determiners, demonstratives
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "both",
        "all",
        "some",
        "any",
        "each",
        "every",
        # Pronouns and possessives
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "one",
        "ones",
        "something",
        "anything",
        "nothing",
        "everything",
        # Interrogatives and relatives
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        # Conjunctions and discourse glue
        "and",
        "but",
        "or",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "because",
        "also",
        "too",
        "as",
        # Prepositions
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "about",
        "into",
        "over",
        "under",
        # Common verbs carrying no retrieval signal
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "get",
        "got",
        "go",
        "going",
        "went",
        "use",
        "using",
        "used",
        # Filler
        "just",
        "really",
        "very",
        "actually",
        "still",
        "now",
        "here",
        "there",
        "again",
        "not",
        "no",
        "yes",
        "ok",
        "okay",
        "please",
        "thanks",
    }
)
"""Words carrying no signal when comparing two pieces of text.

Retrieval drops them from both sides; extraction uses them as a rejection test
(a topic made only of these names nothing retrievable). Note "no" and "not" are
in here, so this cannot distinguish "we chose Redis" from "we did not choose
Redis" — negation needs a learned extractor.
"""


def tokenize_words(text: str) -> list[str]:
    """Split text into lowercase alphanumeric words, in order."""
    return _WORD.findall(text.lower())


def significant_words(text: str) -> set[str]:
    """The distinct words in `text` that carry comparison signal.

    Single characters are dropped along with stopwords: a stray "x" matches too
    much to be useful.
    """
    return {word for word in tokenize_words(text) if len(word) > 1 and word not in STOPWORDS}
