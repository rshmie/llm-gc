"""Play a scripted conversation through the real proxy, then ask its questions with GC and without.

The proxy runs in-process on the engine `build_gc_service` wires, so what is measured
is what users run. The scripted turns are answered locally; only the questions, and
LLM-GC's own summaries, reach a model.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from anthropic.types import MessageParam, OutputConfigParam

from evals.decision_survival.dataset import Conversation
from evals.decision_survival.grading import Presence, presence, question_message
from llm_gc.config import GCConfig
from llm_gc.engine import build_gc_service
from llm_gc.events import Event, EventType
from llm_gc.llm import FakeProvider, LLMProvider, LLMResponse, StopReason
from llm_gc.proxy import SESSION_HEADER, create_app
from llm_gc.utils import count_tokens

# Room for the model's thinking as well as one short line per question.
QUESTIONS_MAX_TOKENS = 16_000
# Scripted turns are answered locally and the reply discarded.
_SCRIPTED_MAX_TOKENS = 64
# Never resolved: requests to it reach the proxy app in-process.
_PROXY_URL = "http://llm-gc.eval"

ModelTransport = Callable[[], httpx2.AsyncBaseTransport]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class Answer:
    text: str            # the reply's text blocks; thinking is not graded
    input_tokens: int    # as the provider counted them
    stop_reason: str | None


@dataclass(frozen=True)
class Run:
    """One conversation, asked through LLM-GC and sent whole."""

    conversation: Conversation
    with_gc: Answer
    without_gc: Answer
    forwarded: list[dict[str, Any]]   # the messages the model received through the proxy
    cuts: int
    bypassed: int                     # requests the proxy forwarded without GC
    degradations: list[str]


async def run_conversation(conversation: Conversation, *, config: GCConfig, summary_provider: LLMProvider,
                           model_transport: ModelTransport, api_key: str, model: str, effort: Effort) -> Run:
    service = build_gc_service(config, summary_provider=summary_provider)
    degradations: list[str] = []
    service.event_bus.subscribe(EventType.AGING_DEGRADED, lambda event: degradations.append(_reason(event)))
    aging: list[asyncio.Future[None]] = []
    upstream = _ProxyUpstream(model_transport())
    upstream_client = httpx2.AsyncClient(transport=upstream)
    app = create_app(service, http_client=upstream_client, schedule=lambda work: aging.append(_start(work)))
    through_proxy = AsyncAnthropic(api_key=api_key, base_url=_PROXY_URL,
                                   default_headers={SESSION_HEADER: conversation.id},
                                   http_client=DefaultAsyncHttpxClient(transport=httpx2.ASGITransport(app=app)))
    direct = AsyncAnthropic(api_key=api_key, http_client=DefaultAsyncHttpxClient(transport=model_transport()))

    turns = [_message(turn.role, turn.text) for turn in conversation.turns]
    try:
        for end in range(1, len(turns), 2):
            await through_proxy.messages.create(model=model, max_tokens=_SCRIPTED_MAX_TOKENS, messages=turns[:end])
            # Aging runs after every response, as in production; finishing it here
            # means each request sees the state the previous one left.
            await asyncio.gather(*aging)
            aging.clear()
        asked = [*turns, _message("user", question_message(conversation.probes))]
        upstream.answering = True
        with_gc = await _ask(through_proxy, asked, model, effort)
        without_gc = await _ask(direct, asked, model, effort)
    finally:
        # Aging after the questions prepares a request that never comes.
        for work in aging:
            work.cancel()
        await asyncio.gather(*aging, return_exceptions=True)
        await through_proxy.close()
        await direct.close()
        await upstream_client.aclose()

    state = service.session_manager.store.get(conversation.id)
    return Run(conversation=conversation, with_gc=with_gc, without_gc=without_gc,
               forwarded=upstream.forwarded_messages(), cuts=state.snapshot.cut_count if state else 0,
               bypassed=app.state.metrics.gc_bypassed, degradations=degradations)


def rehearsal_model(conversation: Conversation) -> ModelTransport:
    """A stand-in model that answers exactly what it was sent: right when an answer is present."""
    return lambda: httpx2.MockTransport(rehearsal_answer(conversation))


def rehearsal_answer(conversation: Conversation) -> Callable[[httpx2.Request], httpx2.Response]:
    def answer(request: httpx2.Request) -> httpx2.Response:
        messages = json.loads(request.content)["messages"]
        lines = [f"{number}. {probe.accept[0] if presence(probe, messages) is not Presence.ABSENT else 'unknown'}"
                 for number, probe in enumerate(conversation.probes, start=1)]
        return _message_response("\n".join(lines), input_tokens=count_tokens(json.dumps(messages)))

    return answer


def rehearsal_summaries() -> LLMProvider:
    """Summaries that state nothing, so a rehearsal shows what survives without them."""
    text = "Earlier turns continued the design discussion."
    response = LLMResponse(text=text, stop_reason=StopReason.COMPLETE, input_tokens=0,
                           output_tokens=count_tokens(text), model="rehearsal", provider="offline")
    return FakeProvider(responses=[response] * 200)


class _ProxyUpstream(httpx2.AsyncBaseTransport):
    """Where the proxy forwards to: a canned reply while the script plays, then the model."""

    def __init__(self, model: httpx2.AsyncBaseTransport) -> None:
        self.model = model
        self.answering = False
        self._forwarded: dict[str, Any] | None = None

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if not self.answering:
            return _message_response("(scripted turn; the conversation supplies the reply)", input_tokens=0)
        self._forwarded = json.loads(await request.aread())
        return await self.model.handle_async_request(request)

    def forwarded_messages(self) -> list[dict[str, Any]]:
        if self._forwarded is None:
            raise RuntimeError("the questions never reached the model")
        return list(self._forwarded["messages"])

    async def aclose(self) -> None:
        await self.model.aclose()


async def _ask(client: AsyncAnthropic, messages: Sequence[MessageParam], model: str, effort: Effort) -> Answer:
    output_config: OutputConfigParam = {"effort": effort}
    response = await client.messages.create(model=model, max_tokens=QUESTIONS_MAX_TOKENS, messages=messages,
                                            output_config=output_config)
    text = "".join(block.text for block in response.content if block.type == "text")
    return Answer(text=text, input_tokens=response.usage.input_tokens, stop_reason=response.stop_reason)


def _message(role: str, text: str) -> MessageParam:
    return {"role": "user" if role == "user" else "assistant", "content": text}


def _start(work: Awaitable[None]) -> asyncio.Future[None]:
    return asyncio.ensure_future(work)


def _reason(event: Event) -> str:
    degradation = event.data["degradation"]
    return f"{degradation.reason.value}: {degradation.detail}"


def _message_response(text: str, *, input_tokens: int) -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "msg_eval", "type": "message", "role": "assistant", "model": "eval",
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": count_tokens(text)},
    })
