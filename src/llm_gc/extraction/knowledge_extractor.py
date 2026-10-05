"""Pull reusable facts out of a turn that is about to be archived.

Regex patterns keyed by `KnowledgeType`, so a new phrasing is a new row rather
than a new branch.

Refusing to label beats labelling wrongly: when nothing matches, the caller stores
the turn verbatim as a `RAW` entry, so a rejected match loses nothing. A
mislabelled one is filed as a confident fact under a topic that is really a
sentence fragment, and since supersession matches on the label it can never be
contradicted. See doc/design/generational-memory/overview.md.
"""

import re
from dataclasses import dataclass

from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import GENERAL_TOPIC, KnowledgeType

# Split on sentence-ending punctuation followed by whitespace, keeping the
# punctuation with the sentence it ends (that is what the lookbehind buys).
#
# Deliberately naive: "e.g." and "Dr. Smith" split wrongly, and no regex fixes
# that without a model or an abbreviation list. The cost of a wrong split is one
# missed or truncated fact, which falls back to a RAW archive. The cost of *not*
# splitting is what this replaces — matching across unrelated clauses and storing
# the result as a fact.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# A topic is one to four words. The old pattern used `[\w\s]+?`, which matches word
# characters *and whitespace*, so a topic could span an unbounded number of words
# and was bounded only by a full stop happening to appear. Counting words is the
# fix: whitespace is a separator here, never part of a word.
_TOPIC = r"(?P<topic>\w+(?:[\s-]+\w+){0,3})"

# Words that cannot be a topic on their own. A "fact" whose subject is `what`, `it`
# or `that` names nothing retrievable later, and its label can never be matched for
# supersession. Rejecting the match sends the turn to a RAW archive instead, which
# keeps the text and claims nothing about it.
_TOPIC_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "ours", "you", "your", "i", "my", "me", "he", "she", "him", "her", "his",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "there", "here", "something", "anything", "nothing", "everything", "one", "ones",
    "and", "but", "or", "so", "if", "then", "also", "both", "all", "some", "any",
})

# Dropped from the front of a topic, so "for the database" and "the database is"
# file under one label: supersession compares labels exactly.
_LEADING_DETERMINERS = frozenset({"the", "a", "an", "our", "my", "your", "their"})

# Shorter than this and the "content" is a fragment, not a fact worth retrieving.
_MIN_CONTENT_CHARS = 3

# What a person puts before a statement without adding to it: content-free openers
# ("OK, the database is postgres") and a short label before a colon ("Decision: the
# database is postgres"). Stripped before matching, because the subject is anchored
# at the sentence start and either would hide it.
_OPENER = re.compile(
    r"^(?:(?:ok(?:ay)?|so|alright|right|well|actually|also|and|but|now|yes|yeah|sure|fine|great|good|cool"
    r"|perfect|thanks|thank you|got it|understood|noted|sounds good)\b[\s,]*)+",
    re.IGNORECASE,
)
_LABEL = re.compile(r"^\w+(?:[\s-]+\w+){0,3}:\s+")

# The sentence splitter keeps the full stop with its sentence, so the end-of-sentence
# anchors in the patterns never met the topic slot before it: "for the database."
# ends in a full stop, not in "database", and every such decision filed as general.
_END_PUNCTUATION = re.compile(r"[.!;:,]+$")

# Separators inside a decision's content. A trailing "for <topic>" names the topic
# of everything before it, which is only safe when that is one thing: in "use
# postgres for the database, and redis for the cache" the topic slot catches "cache".
_COMPOUND = re.compile(r",|;|\band\b|\bfor\b", re.IGNORECASE)


@dataclass(frozen=True)
class Reading:
    """What the extractor made of one turn."""

    entries: list[KnowledgeEntry]
    unread: str   # the sentences that yielded nothing, as written; empty when none did


class KnowledgeExtractor:
    EXTRACTION_PATTERNS = {
        KnowledgeType.FACT: [
            re.compile(rf"^(?:the\s+)?{_TOPIC}\s+(?:is|are)\s+(?P<content>.+)", re.IGNORECASE),
            re.compile(rf"(?:we(?:'re|\s+are)\s+)?using\s+(?P<content>\w+)\s+for\s+{_TOPIC}", re.IGNORECASE),
            re.compile(rf"^{_TOPIC}\s+runs?\s+on\s+(?P<content>.+)", re.IGNORECASE),
        ],
        KnowledgeType.DECISION: [
            re.compile(rf"(?:we\s+)?decided\s+to\s+(?P<content>.+?)(?:\s+for\s+{_TOPIC})?$", re.IGNORECASE),
            re.compile(rf"let(?:'s|\s+us)\s+go\s+with\s+(?P<content>.+?)(?:\s+for\s+{_TOPIC})?$", re.IGNORECASE),
            re.compile(rf"(?:we\s+)?chose\s+(?P<content>.+?)\s+over\s+.+?(?:\s+for\s+{_TOPIC})?$", re.IGNORECASE),
        ],
        KnowledgeType.PREFERENCE: [
            re.compile(rf"(?:i|we)\s+prefer\s+(?P<content>.+?)(?:\s+for\s+{_TOPIC})?$", re.IGNORECASE),
            re.compile(
                rf"(?:i|we)(?:'d|\s+would)\s+rather\s+(?:use\s+)?(?P<content>.+?)(?:\s+for\s+{_TOPIC})?$",
                re.IGNORECASE,
            ),
        ],
    }

    def extract_knowledge(self, message: Message, message_turn: int) -> list[KnowledgeEntry]:
        """Extract whatever facts, decisions and preferences the turn states.

        Returns an empty list when nothing matches cleanly, which the archiving
        caller treats as "store the turn verbatim". That is the intended outcome for
        an ordinary conversational turn, not a failure.
        """
        return self.read(message, message_turn).entries

    def read(self, message: Message, message_turn: int) -> Reading:
        """Extract the turn's entries, and keep the sentences none came from.

        A turn is rarely one statement. Archived as only the facts found in it, a
        turn that stated a database and a latency requirement kept the database
        and lost the requirement, so what extraction cannot read is returned for
        the caller to keep. Content-free sentences ("OK.", "Thanks!") are not.
        """
        entries: list[KnowledgeEntry] = []
        unread: list[str] = []
        for sentence in self._split_sentences(message.content):
            statement = self._strip(sentence)
            if not statement:
                continue
            # A question claims nothing: "the database is postgres, right?" asks for
            # confirmation, and filing it would let it supersede the answer.
            found = [] if sentence.endswith("?") else self._entries_in(statement, message_turn)
            if found:
                entries.extend(found)
            else:
                unread.append(sentence)

        return Reading(entries=entries, unread=" ".join(unread))

    def _entries_in(self, statement: str, message_turn: int) -> list[KnowledgeEntry]:
        entries = []
        for knowledge_type, patterns in self.EXTRACTION_PATTERNS.items():
            for pattern in patterns:
                entry = self._entry_from(pattern, statement, knowledge_type, message_turn)
                if entry is not None:
                    entries.append(entry)
                    # First match per type per sentence wins. The patterns within a
                    # type are alternative phrasings of one idea, so two of them
                    # matching means two descriptions of the same fact, not two
                    # facts — and storing both would make the later one supersede
                    # the earlier one for no reason.
                    break
        return entries

    def _entry_from(
        self, pattern: re.Pattern[str], sentence: str, knowledge_type: KnowledgeType, message_turn: int
    ) -> KnowledgeEntry | None:
        """Build one entry from a match, or None if the match is not worth storing."""
        match = pattern.search(sentence)
        if match is None:
            return None

        content = (match.group("content") or "").strip()
        if len(content) < _MIN_CONTENT_CHARS:
            return None

        topic_label = self._normalise_topic(match.group("topic"))
        if topic_label is None:
            return None
        if match.start("topic") > match.start("content") and _COMPOUND.search(content):
            topic_label = GENERAL_TOPIC   # the trailing topic names only the last of several things

        return KnowledgeEntry(
            message_turn=message_turn,
            content=content,
            topic_label=topic_label,
            knowledge_type=knowledge_type,
        )

    @staticmethod
    def _strip(sentence: str) -> str:
        """The sentence without its openers, leading label or closing punctuation."""
        sentence = _OPENER.sub("", sentence)
        sentence = _OPENER.sub("", _LABEL.sub("", sentence))
        return _END_PUNCTUATION.sub("", sentence).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split a turn into sentences, dropping empties.

        Matching per sentence instead of per message is the core of the fix. The
        old code ran `search` over the whole turn, so an "is" two hundred words in
        would match and the topic capture would reach back toward the start.
        """
        return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]

    @staticmethod
    def _normalise_topic(raw: str | None) -> str | None:
        """Turn a raw topic capture into a label, or reject it.

        Three outcomes:

        - `None` capture -> `GENERAL_TOPIC`; the pattern has no topic slot.
        - A usable capture -> lowercased, whitespace collapsed and a leading
          article or possessive dropped, because supersession compares labels
          exactly.
        - Stopwords only, or empty -> `None`, meaning discard the match so the
          caller archives the turn verbatim instead.
        """
        if raw is None:
            return GENERAL_TOPIC

        words = raw.lower().split()
        while len(words) > 1 and words[0] in _LEADING_DETERMINERS:
            words = words[1:]
        if not words or all(word in _TOPIC_STOPWORDS for word in words):
            return None
        return " ".join(words)
