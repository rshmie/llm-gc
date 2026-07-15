from itertools import cycle

from llm_gc.models import Message
from llm_gc.utils import count_tokens

# Plain filler lines. Deliberately avoid the knowledge-extractor patterns
# ("X is Y", "decided to", "we chose", ...) so they archive as RAW entries.
_FILLER_LINES = [
    "kicked off the sprint planning discussion this morning",
    "walked through the open review comments together",
    "spent a while debugging the flaky integration run",
    "compared notes on the deployment checklist",
    "went over the incident timeline from last week",
    "sketched the onboarding flow on the whiteboard",
    "cleaned up a couple of stale feature branches",
    "paired on the migration script for an hour",
]

# Fact lines the extractor parses (topic: "database" / "cache" / "queue").
# Topics repeat with different values, so once an older fact has been
# archived, its newer contradiction triggers a supersession - the dashboard
# gets real SUPERSEDED transitions just by clicking "Run GC now" repeatedly.
_FACT_LINES = [
    "the database is postgres",
    "the cache is redis",
    "the database is mysql",
    "the cache is memcached",
    "the database is sqlite",
    "the queue is rabbitmq",
]


class DemoConversationFeeder:
    """Produces the next batch of synthetic turns for the dashboard's demo session.

    Each "Run GC now" click appends one batch: a few filler turns plus one
    extractable fact. Owns the turn counter so turn indices stay monotonic
    across GC passes (the conversation itself shrinks and grows as GC rewrites
    it, but a turn index is never reused).
    """

    def __init__(self, turns_per_batch: int = 4) -> None:
        self.turns_per_batch = turns_per_batch
        self._next_turn_index = 0
        self._filler_lines = cycle(_FILLER_LINES)
        self._fact_lines = cycle(_FACT_LINES)

    def next_turns(self) -> list[Message]:
        contents = [next(self._filler_lines) for _ in range(self.turns_per_batch - 1)]
        contents.append(next(self._fact_lines))

        messages = []
        for content in contents:
            role = "user" if self._next_turn_index % 2 == 0 else "assistant"
            messages.append(Message(role=role, content=content, token_count=count_tokens(content),
                                    turn_index=self._next_turn_index))
            self._next_turn_index += 1
        return messages
