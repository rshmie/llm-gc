"""Pull reusable facts out of a turn that is about to be archived.

Regex patterns, not a model. The patterns are a dictionary keyed by
`KnowledgeType` so a new phrasing is a new row rather than a new branch — the
Open/Closed Principle, applied to a lookup table.

The guiding constraint is the same one the compactor works under: **refusing to
label is better than labelling wrongly.** When nothing here matches, the archiving
caller stores the turn verbatim as a `RAW` entry, so a rejected match loses
nothing. A *mislabelled* match, by contrast, is stored as a confident fact under a
topic that is really a sentence fragment — and because supersession matches on the
topic label, a mangled label can never be contradicted and sits active forever.
That is a worse outcome than no extraction at all.
"""

import re

from llm_gc.events import Event, EventBus, EventType
from llm_gc.models import KnowledgeEntry, Message
from llm_gc.models.knowledge_entry import KnowledgeType

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

# Shorter than this and the "content" is a fragment, not a fact worth retrieving.
_MIN_CONTENT_CHARS = 3

_GENERAL_TOPIC = "general"
"""Used when a pattern legitimately has no topic slot filled — "we decided to ship
on Friday" states a decision with no subject to name. Distinct from a *rejected*
topic, which discards the match entirely."""


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

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus

    def extract_knowledge(self, message: Message, message_turn: int) -> list[KnowledgeEntry]:
        """Extract whatever facts, decisions and preferences the turn states.

        Returns an empty list when nothing matches cleanly, which the archiving
        caller treats as "store the turn verbatim". That is the intended outcome for
        an ordinary conversational turn, not a failure.
        """
        knowledge_entries: list[KnowledgeEntry] = []

        for sentence in self._split_sentences(message.content):
            for knowledge_type, patterns in self.EXTRACTION_PATTERNS.items():
                for pattern in patterns:
                    entry = self._entry_from(pattern, sentence, knowledge_type, message_turn)
                    if entry is not None:
                        knowledge_entries.append(entry)
                        # First match per type per sentence wins. The patterns within
                        # a type are alternative phrasings of one idea, so two of them
                        # matching means two descriptions of the same fact, not two
                        # facts — and storing both would make the later one supersede
                        # the earlier one for no reason.
                        break

        if knowledge_entries:
            self.event_bus.emit(Event(event_type=EventType.KNOWLEDGE_ENTRIES_EXTRACTED,
                                      data={"entries": knowledge_entries, "message_turn": message_turn}))
        return knowledge_entries

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

        return KnowledgeEntry(
            message_turn=message_turn,
            content=content,
            topic_label=topic_label,
            knowledge_type=knowledge_type,
        )

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

        Three outcomes, and the distinction between the last two matters:

        - `None` capture -> `_GENERAL_TOPIC`. The pattern has no topic slot filled,
          which is legitimate: "we decided to ship on Friday" names no subject.
        - A usable capture -> lowercased, with internal whitespace collapsed.
          Supersession compares topic labels exactly, so "the Database is X" and
          "the database is Y" have to normalise to the same string or the
          contradiction is never detected.
        - Stopwords only, or empty after stripping -> `None`, meaning *discard the
          match*. The caller then stores the turn verbatim instead of filing a fact
          under a label that names nothing.

        The old code was `match.group("topic") or "general"`, which is the Python
        `or`-as-default idiom guarding the wrong thing: it falls back on *empty*,
        not on *meaningless*. `""` is falsy and fell back correctly; `" "` is truthy
        and was stored, which is exactly where the leading space in `' What'` came
        from — `content` was stripped and `topic_label` was not.
        """
        if raw is None:
            return _GENERAL_TOPIC

        collapsed = " ".join(raw.split()).lower()
        if not collapsed:
            return None
        if all(word in _TOPIC_STOPWORDS for word in collapsed.split()):
            return None
        return collapsed
