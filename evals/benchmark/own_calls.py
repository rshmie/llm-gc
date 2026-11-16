"""LLM-GC's own model calls, kept by the model that answered each one.

The engine counts what each collection's calls spend as one total, which is enough for the
dashboard but not for a bill. LLM-GC usually summarises with a cheaper model than the
conversation's, and a gateway can answer with a model other than the one named, so the
benchmark wraps the summary provider it hands the engine and keeps every finished call
under the model the provider says answered it.
"""

from llm_gc.llm import LLMProvider, LLMResponse, ModelUsage
from llm_gc.models import Message


class OwnCallRecorder:
    """An `LLMProvider` that passes each call through and keeps what each finished one spent, by model.

    It counts for itself instead of reading the engine's count. That way it sees every call
    the engine makes, inside a collection or not, and a fault in the engine's count shows up
    as a disagreement between the two instead of being copied into the bill. A call that
    raises adds nothing, because the provider reported no usage for it.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.name: str = getattr(inner, "name", type(inner).__name__)
        self._spent: dict[str, ModelUsage] = {}

    async def generate(
        self, *, messages: list[Message], system: str | None = None, max_output_tokens: int, timeout_s: float
    ) -> LLMResponse:
        response = await self.inner.generate(
            messages=messages, system=system, max_output_tokens=max_output_tokens, timeout_s=timeout_s
        )
        # Looked up only now, after the call: a take made while it was running started a
        # new count, and this call belongs in that one.
        spent = self._spent.setdefault(response.model, ModelUsage())
        spent.add(ModelUsage(calls=1, input_tokens=response.input_tokens, output_tokens=response.output_tokens))
        return response

    def take(self) -> dict[str, ModelUsage]:
        """Return what calls have spent since the last take, by model, and start counting afresh.

        The replay takes after each request, once that request's collection has finished, so
        each call is billed to the request it followed. A call still running at a take is
        billed to the next one: late, but never twice and never lost.
        """
        taken, self._spent = self._spent, {}
        return taken
