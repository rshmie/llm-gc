from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClientOrigin:
    """Which of the client's own sessions a conversation came from, as the client names it.

    One client session can hold several conversations: Claude Code runs its main
    conversation, one per sub-agent, and a fresh one after it compacts, all under one
    session id. Each is its own prompt, so each is its own LLM-GC session; this only
    says which client session it belongs to, so they can be shown together. It never
    decides which conversation a request continues.
    """

    session_id: str
    agent_id: str | None = None  # set for a sub-agent's conversation, None for the main one
