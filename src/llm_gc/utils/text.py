"""Word-level text helpers shared by components that compare English text.

Pure functions and one constant, no state. Lives in `utils/` because two
unrelated components need the same stopword list, and two copies of a
forty-word list drift.
"""

import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset({
    # Articles, determiners, demonstratives
    "a", "an", "the", "this", "that", "these", "those", "both", "all", "some", "any", "each", "every",
    # Pronouns and possessives
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
    "one", "ones", "something", "anything", "nothing", "everything",
    # Interrogatives and relatives
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    # Conjunctions and discourse glue
    "and", "but", "or", "nor", "so", "if", "then", "than", "because", "also", "too", "as",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "about", "into", "over", "under",
    # Common verbs that carry no retrieval signal on their own
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "must", "shall",
    "get", "got", "go", "going", "went", "use", "using", "used",
    # Filler
    "just", "really", "very", "actually", "still", "now", "here", "there", "again",
    "not", "no", "yes", "ok", "okay", "please", "thanks",
})
"""Words that carry no signal when comparing two pieces of text.

One list, two different uses, and the difference is worth knowing:

- **Retrieval** drops them from both sides of a comparison, because "which
  database are we targeting" and "the database is postgres" should match on
  `database`, not on `we` and `the`.
- **Extraction** uses them as a *rejection* test: a topic made only of these
  words names nothing retrievable later, so the match is discarded.

A stopword list is a blunt instrument. "no" and "not" are in here, which means
this cannot distinguish "we chose Redis" from "we did not choose Redis" — a real
limitation, and the reason negation handling belongs to a learned extractor rather
than to word matching.
"""


def tokenize_words(text: str) -> list[str]:
    """Split text into lowercase alphanumeric words, in order.

    Punctuation and case are dropped, so `"Postgres,"` and `"postgres"` are the
    same word. Order is preserved rather than returning a set, because a caller
    that wants a set can build one and a caller that wants position cannot
    recover it.
    """
    return _WORD.findall(text.lower())


def significant_words(text: str) -> set[str]:
    """The distinct words in `text` that carry comparison signal.

    A set, because overlap is the only question asked of it and repetition does
    not make a word more relevant. Single characters are dropped along with
    stopwords: a stray "x" matches too much to be useful.
    """
    return {word for word in tokenize_words(text) if len(word) > 1 and word not in STOPWORDS}
