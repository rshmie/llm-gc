"""Sizing each image and document a client sends at what its model bills, and counting it while it is sent."""

import asyncio
import json

import httpx2

from llm_gc.proxy import SESSION_HEADER
from llm_gc.proxy.anthropic_translation import attachment_blocks, attachment_key, extract_messages
from llm_gc.proxy.attachment_sizes import COUNTS_IN_FLIGHT, AttachmentSizes
from llm_gc.utils import count_tokens
from tests.proxy.test_proxy_server import API_KEY, UPSTREAM_OK, _is_front, _proxy

MODEL = "claude-opus-5-5"
BASELINE = 9  # what the stand-in endpoint adds to every message
SCREENSHOT_TOKENS = 3_700


def _image(data: str = "iVBORw0KGgo" + "A" * 9_000 + "end") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


class _Endpoint:
    """A stand-in for the token-counting endpoint: the baseline, plus each image's size."""

    def __init__(self, sizes: dict[str, int] | None = None, fails: bool = False) -> None:
        self.sizes = sizes or {}
        self.fails = fails
        self.asked: list[tuple[str, int]] = []

    async def __call__(self, model: str, content: list[dict]) -> int | None:
        self.asked.append((model, len(content)))
        if self.fails:
            raise TimeoutError("the upstream did not answer")
        images = [block for block in content if block.get("type") == "image"]
        return BASELINE + sum(self.sizes.get(model, SCREENSHOT_TOKENS) for _ in images)


def _sizes(endpoint, **kwargs) -> tuple[AttachmentSizes, list]:
    scheduled: list = []
    return AttachmentSizes(endpoint, scheduled.append, **kwargs), scheduled


def _run(scheduled: list) -> None:
    for coroutine in scheduled:
        asyncio.run(coroutine)
    scheduled.clear()


class TestSizingAnAttachment:
    def test_it_is_unsized_until_its_count_comes_back_and_then_known(self):
        endpoint = _Endpoint()
        sizes, scheduled = _sizes(endpoint)

        assert sizes.size(MODEL, "k", _image()) is None, "never waits on the request path"
        _run(scheduled)

        assert sizes.size(MODEL, "k", _image()) == SCREENSHOT_TOKENS
        assert not scheduled, "counted once"

    def test_what_the_provider_adds_to_every_message_is_taken_off_and_counted_once_per_model(self):
        endpoint = _Endpoint()
        sizes, scheduled = _sizes(endpoint)

        sizes.size(MODEL, "a", _image())
        _run(scheduled)
        sizes.size(MODEL, "b", _image("other"))
        _run(scheduled)

        assert endpoint.asked == [(MODEL, 1), (MODEL, 2), (MODEL, 2)], "the baseline once, then each image"
        assert sizes.size(MODEL, "b", _image("other")) == SCREENSHOT_TOKENS

    def test_a_size_belongs_to_the_model_it_was_counted_on(self):
        """Older models cap an image lower than newer ones, so one image costs them differently."""
        endpoint = _Endpoint(sizes={"claude-haiku-4-5": 1_568})
        sizes, scheduled = _sizes(endpoint)

        sizes.size(MODEL, "k", _image())
        sizes.size("claude-haiku-4-5", "k", _image())
        _run(scheduled)

        assert sizes.size(MODEL, "k", _image()) == SCREENSHOT_TOKENS
        assert sizes.size("claude-haiku-4-5", "k", _image()) == 1_568

    def test_a_count_that_fails_leaves_it_unsized_and_is_not_asked_again(self, caplog):
        sizes, scheduled = _sizes(_Endpoint(fails=True))

        sizes.size(MODEL, "k", _image())
        _run(scheduled)

        assert sizes.size(MODEL, "k", _image()) is None
        assert not scheduled, "a failure is remembered, so every request does not ask again"
        assert "TimeoutError" in caplog.text and "iVBOR" not in caplog.text

    def test_an_upstream_without_the_endpoint_leaves_it_unsized_and_says_so(self, caplog):
        """A gateway that does not offer token counting answers 404, which the client reads as no count."""

        async def no_endpoint(model: str, content: list[dict]) -> int | None:
            return None

        sizes, scheduled = _sizes(no_endpoint)
        sizes.size(MODEL, "k", _image())
        _run(scheduled)

        assert sizes.size(MODEL, "k", _image()) is None
        assert "no token count" in caplog.text

    def test_without_a_credential_nothing_is_asked(self):
        scheduled: list = []
        sizes = AttachmentSizes(None, scheduled.append)

        assert sizes.size(MODEL, "k", _image()) is None
        assert not scheduled

    def test_only_so_many_are_asked_at_once_and_the_rest_on_a_later_request(self):
        sizes, scheduled = _sizes(_Endpoint())

        for index in range(COUNTS_IN_FLIGHT + 3):
            sizes.size(MODEL, f"k{index}", _image(str(index)))
        assert len(scheduled) == COUNTS_IN_FLIGHT
        _run(scheduled)

        sizes.size(MODEL, f"k{COUNTS_IN_FLIGHT}", _image())
        assert len(scheduled) == 1
        _run(scheduled)

    def test_the_attachment_sent_back_least_recently_is_let_go_first(self):
        sizes, scheduled = _sizes(_Endpoint(), capacity=2)
        for key in ("old", "newer"):
            sizes.size(MODEL, key, _image())
        _run(scheduled)
        sizes.size(MODEL, "old", _image())  # sent again, so it is no longer the oldest

        sizes.size(MODEL, "newest", _image())
        _run(scheduled)

        assert sizes.size(MODEL, "old", _image()) == SCREENSHOT_TOKENS
        assert sizes.size(MODEL, "newer", _image()) is None
        assert len(scheduled) == 1, "let go, so asked about again"
        _run(scheduled)


class TestFindingAttachments:
    def test_images_and_documents_are_found_directly_and_inside_tool_results(self):
        document = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVB"}}
        content = [
            {"type": "text", "text": "what is wrong here?"},
            _image(),
            {"type": "tool_result", "tool_use_id": "t1", "content": [document, {"type": "text", "text": "ok"}]},
            {"type": ["image"], "source": {}},
            "image",
        ]

        assert attachment_blocks(content) == [_image(), document]
        assert attachment_blocks("a plain string turn") == []

    def test_the_same_image_has_the_same_key_and_another_image_another(self):
        assert attachment_key(_image()) == attachment_key(_image())
        assert attachment_key(_image()) != attachment_key(_image("iVBORw0KGgo" + "B" * 9_000 + "end"))
        assert attachment_key(_image()) != attachment_key(_image("iVBORw0KGgo" + "A" * 9_001 + "end"))

    def test_an_image_given_by_url_is_known_by_its_url(self):
        def by_url(url: str) -> dict:
            return {"type": "image", "source": {"type": "url", "url": url}}

        assert attachment_key(by_url("https://a.example/1.png")) != attachment_key(by_url("https://a.example/2.png"))

    def test_a_turn_counts_what_its_sized_attachments_add_beyond_their_placeholders(self):
        payload = {
            "messages": [
                {"role": "user", "content": [_image(), _image("unsized"), {"type": "text", "text": "and these?"}]},
            ]
        }
        known = attachment_key(_image())

        [turn] = extract_messages(payload, size_attachment=lambda key, block: 2_500 if key == known else None)

        assert turn.attachment_tokens == 2_500 - count_tokens("[image]")
        assert extract_messages(payload)[0].attachment_tokens == 0


class TestTheProxyCountsAnAttachmentAtItsSize:
    @staticmethod
    def _provider():
        """Reports a prompt equal to LLM-GC's count of the text, so the rate is 1.0 exactly."""

        def upstream(request: httpx2.Request) -> httpx2.Response:
            text = sum(turn.token_count for turn in extract_messages(json.loads(request.content)))
            return httpx2.Response(200, json={**UPSTREAM_OK, "usage": {"input_tokens": text, "output_tokens": 9}})

        return upstream

    def test_from_the_request_after_it_first_appears_until_a_collection_takes_its_turn(self):
        """Live run: four screenshots, about 13,000 provider tokens, counted as four placeholders. The
        provider ÷ LLM-GC ratio stepped at each, and a collection's saving left them out."""
        client, _, _, seen = _proxy(self._provider(), attachment_counter=_Endpoint(), min_cut_tokens=60)
        added = SCREENSHOT_TOKENS - count_tokens("[image]")
        history: list[dict] = []
        collected_away = False
        for index in range(12):
            content = f"question {index}: the storage layer, its retention window and the rollout plan"
            history.append(
                {"role": "user", "content": [_image(), {"type": "text", "text": content}] if index == 1 else content}
            )
            payload = {"model": MODEL, "max_tokens": 64, "messages": list(history)}
            reply = client.post("/v1/messages", json=payload, headers={"x-api-key": API_KEY, SESSION_HEADER: "s"})
            history.append({"role": "assistant", "content": reply.json()["content"]})
            record = client.app.state.metrics.request_log.records()[-1]
            sent = json.loads(seen[-1].content)["messages"]
            text_sent = sum(turn.token_count for turn in extract_messages({"messages": sent}))
            text_from_client = sum(turn.token_count for turn in extract_messages(payload))
            image_sent = any(attachment_blocks(message["content"]) for message in sent)

            sized = index > 1  # sized once counted, which the request that brings it never waits for
            assert record.tokens_from_client == text_from_client + (added if sized else 0), index
            assert record.tokens_forwarded == text_sent + (added if sized and image_sent else 0), index
            collected_away |= index > 1 and _is_front(sent[0]) and not image_sent

        assert collected_away, "a collection took the image's turn, and the image with it"
