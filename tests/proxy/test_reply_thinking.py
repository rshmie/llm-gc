"""Measuring each reply's hidden thinking, and attaching it to the turn when the client sends it back."""

import itertools
import json

import httpx2
import pytest

from llm_gc.proxy import SESSION_HEADER
from llm_gc.models import Message
from llm_gc.proxy.anthropic_translation import (StreamOutlineReader, extract_messages, read_reply,
                                                rewrite_messages, sent_thinking_tokens, thinking_keys)
from llm_gc.proxy.reply_thinking import ReplyThinking
from llm_gc.utils import count_tokens
from tests.proxy.test_proxy_server import API_KEY, UPSTREAM_OK, _body, _is_front, _post, _proxy, _streaming_upstream

MODEL = "claude-opus-5-5"
# Shaped like a real signature: base64 text, long enough to be realistic.
SIGNATURE = "EqQBCkgIARABGAIiQL" + "q7Kd0Zr9fXw2Tb4" * 40
OTHER_SIGNATURE = "EqQBCkgIARABGAIiQL" + "M3nVb8Lp1sYc6Qe" * 40


def _events(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _thinking_stream(signature: str = SIGNATURE, output_tokens: int = 900, text: str = "Postgres.") -> bytes:
    return _events(
        {"type": "message_start", "message": {"usage": {"input_tokens": 300, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": signature}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": output_tokens}},
        {"type": "message_stop"},
    )


def _read(stream: bytes, size: int) -> StreamOutlineReader:
    reader = StreamOutlineReader()
    for start in range(0, len(stream), size):
        reader.feed(stream[start:start + size])
    return reader


def _sent_back(signature: str = SIGNATURE, text: str = "Postgres.") -> list:
    return [{"type": "thinking", "thinking": "", "signature": signature}, {"type": "text", "text": text}]


def _replies(rate: float = 1.5, **kwargs) -> ReplyThinking:
    replies = ReplyThinking(**kwargs)
    replies.observe_rate(MODEL, rate)
    return replies


class TestSizingAReplysThinking:
    def test_it_is_the_output_beyond_what_the_turn_shows_at_the_models_rate(self):
        replies = _replies(rate=1.5)
        replies.remember("k", output_tokens=900, model=MODEL)

        assert replies.tokens(["k"], visible_tokens=40, model=MODEL) == 600 - 40

    def test_a_turn_whose_reply_was_never_measured_carries_none(self):
        assert _replies().tokens(["unknown"], visible_tokens=40, model=MODEL) == 0

    def test_a_reply_is_not_kept_before_its_model_has_a_rate(self):
        replies = ReplyThinking()
        replies.remember("k", output_tokens=900, model=MODEL)

        assert len(replies) == 0

    def test_the_rate_is_fixed_when_the_reply_is_stored(self):
        """Its size must not move from one request to the next as newer rates arrive."""
        replies = _replies(rate=1.5)
        replies.remember("k", output_tokens=900, model=MODEL)
        replies.observe_rate(MODEL, 3.0)

        assert replies.tokens(["k"], visible_tokens=40, model=MODEL) == 600 - 40

    def test_it_is_counted_only_for_the_model_that_wrote_it(self):
        """Another model may be unable to read it, and the provider does not bill thinking it drops."""
        replies = _replies()
        replies.remember("k", output_tokens=900, model=MODEL)

        assert replies.tokens(["k"], visible_tokens=40, model="claude-sonnet-5-5") == 0

    def test_an_estimate_below_what_the_turn_shows_is_none_not_negative(self):
        replies = _replies()
        replies.remember("k", output_tokens=50, model=MODEL)

        assert replies.tokens(["k"], visible_tokens=40, model=MODEL) == 0

    def test_a_turn_holding_two_replies_takes_off_what_it_shows_once(self):
        replies = _replies()
        replies.remember("a", output_tokens=300, model=MODEL)
        replies.remember("b", output_tokens=600, model=MODEL)

        assert replies.tokens(["a", "b", "not a reply's first block"], visible_tokens=40, model=MODEL) == 600 - 40

    def test_the_reply_sent_back_least_recently_is_let_go_first(self):
        """Looking a reply up marks it as recently used, so active conversations keep their entries."""
        replies = _replies(rate=1.0, capacity=2)
        replies.remember("old", output_tokens=900, model=MODEL)
        replies.remember("newer", output_tokens=900, model=MODEL)
        replies.tokens(["old"], visible_tokens=0, model=MODEL)     # used again, so it is no longer the oldest

        replies.remember("newest", output_tokens=900, model=MODEL)

        assert len(replies) == 2
        assert replies.tokens(["newer"], visible_tokens=0, model=MODEL) == 0
        assert replies.tokens(["old"], visible_tokens=0, model=MODEL) == 900


class TestKnowingAThinkingBlock:
    def test_a_reply_and_the_turn_that_sends_it_back_give_the_same_key(self):
        """If the two sides computed different keys, every turn would silently get 0 thinking."""
        answered = json.dumps({"type": "message", "stop_reason": "end_turn", "content": _sent_back()}).encode()
        [key] = thinking_keys(_sent_back())

        assert read_reply(200, answered)[0].thinking_key == key
        assert _read(_thinking_stream(), size=len(_thinking_stream())).outline().thinking_key == key

    def test_a_redacted_block_is_known_by_its_data(self):
        redacted = [{"type": "redacted_thinking", "data": SIGNATURE}]
        stream = _events({"type": "content_block_start", "index": 0, "content_block": redacted[0]},
                         {"type": "message_stop"})

        assert _read(stream, size=64).outline().thinking_key == thinking_keys(redacted)[0]

    def test_blocks_that_are_not_thinking_have_no_key(self):
        assert thinking_keys([{"type": "text", "text": SIGNATURE}, {"type": "thinking", "signature": ""},
                              {"type": ["thinking"], "signature": SIGNATURE}, "text"]) == ()
        assert thinking_keys("a plain string turn") == ()

    def test_distinct_blocks_have_distinct_keys_and_no_key_holds_the_thinking(self):
        [key], [other] = thinking_keys(_sent_back()), thinking_keys(_sent_back(OTHER_SIGNATURE))

        assert key != other
        assert SIGNATURE[:16] not in key


class TestReadingAStreamsThinking:
    def test_a_stream_split_anywhere_gives_the_first_blocks_key(self):
        [key] = thinking_keys(_sent_back())
        stream = _thinking_stream()

        for size in (1, 7, 64, len(stream)):
            assert _read(stream, size).outline().thinking_key == key, size

    def test_a_signature_too_long_to_hold_is_known_from_the_head_of_its_line(self):
        """A long signature arrives as one line bigger than the reader keeps; the key comes from its start."""
        signature = SIGNATURE * 200
        [key] = thinking_keys(_sent_back(signature))

        for size in (4096, 65_536):
            assert _read(_thinking_stream(signature), size).outline().thinking_key == key, size

    def test_only_the_first_thinking_block_names_the_reply(self):
        stream = _events(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": SIGNATURE}},
            {"type": "content_block_delta", "index": 2,
             "delta": {"type": "signature_delta", "signature": OTHER_SIGNATURE}},
            {"type": "message_stop"})

        assert _read(stream, size=64).outline().thinking_key == thinking_keys(_sent_back())[0]

    def test_text_that_names_the_event_is_not_a_signature(self):
        stream = _events({"type": "content_block_delta", "index": 0,
                          "delta": {"type": "text_delta", "text": '"signature_delta" "signature": "EqQB' + "x" * 70_000}},
                         {"type": "message_stop"})

        for size in (64, 4096):
            assert _read(stream, size).outline().thinking_key is None, size


class TestATurnCarriesItsReplysThinking:
    def test_only_an_assistant_turn_with_thinking_is_sized(self):
        asked = []

        def size(keys, visible_tokens):
            asked.append((keys, visible_tokens))
            return 500

        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": "which database?"}]},
                                {"role": "assistant", "content": _sent_back()},
                                {"role": "assistant", "content": "No thinking here."},
                                {"role": "user", "content": "and the retention?"}]}

        turns = extract_messages(payload, size_thinking=size)

        assert [turn.thinking_tokens for turn in turns] == [0, 500, 0, 0]
        assert asked == [(thinking_keys(_sent_back()), count_tokens("Postgres."))]
        assert [turn.thinking_tokens for turn in extract_messages(payload)] == [0, 0, 0, 0]

    @pytest.mark.parametrize("next_model, carried", [(MODEL, True), ("claude-sonnet-5-5", False)])
    def test_the_proxy_measures_a_streamed_reply_and_sizes_it_when_sent_back(self, next_model, carried):
        client, service, _, seen = _proxy(_streaming_upstream(_thinking_stream(output_tokens=900)))
        first = _body(("user", "which database should hold the events?"), stream=True, model=MODEL)
        _post(client, first)
        forwarded = client.get("/stats").json()["tokens_forwarded"]

        later = {**first, "model": next_model, "messages": [*first["messages"],
                                                            {"role": "assistant", "content": _sent_back()},
                                                            {"role": "user", "content": "and the retention?"}]}
        _post(client, later, headers={SESSION_HEADER: "s"})

        reply = service.session_manager.store.get("s").snapshot.young[1]
        rate = 300 / forwarded
        assert reply.thinking_tokens == (round(900 / rate) - count_tokens("Postgres.") if carried else 0)
        assert len(seen) == 2

    def test_a_conversation_uses_its_models_rate_until_it_measures_its_own(self):
        """Live run: the first request is clean but comes before the conversation is known, and every later
        one carries thinking, so the conversation never had a rate and its window went unconverted."""
        client, service, _, _ = _proxy(_streaming_upstream(_thinking_stream(output_tokens=900)))
        first = _body(("user", "which database should hold the events?"), stream=True, model=MODEL)
        _post(client, first)
        forwarded = client.get("/stats").json()["tokens_forwarded"]

        later = {**first, "messages": [*first["messages"], {"role": "assistant", "content": _sent_back()},
                                       {"role": "user", "content": "and the retention?"}]}
        _post(client, later, headers={SESSION_HEADER: "s"})

        assert service.session_manager.store.get("s").token_rate == 300 / forwarded

    def test_an_unfinished_or_refused_reply_is_not_measured(self):
        cut_short = _thinking_stream().split(b"event: message_stop")[0]
        client, _, _, _ = _proxy(_streaming_upstream(cut_short))
        _post(client, _body(("user", "which database?"), stream=True, model=MODEL))

        refused, _, _, _ = _proxy(lambda request: httpx2.Response(529, json={"type": "error", "error": {}}))
        _post(refused, _body(("user", "which database?"), stream=True, model=MODEL))

        assert len(client.app.state.reply_thinking) == 0
        assert len(refused.app.state.reply_thinking) == 0


class TestThinkingIsCountedAsSent:
    def test_the_count_and_the_rewrite_agree_on_which_turns_keep_their_thinking(self):
        turns = [Message(role="user", content="recap", turn_index=-1)] + [
            Message(role="assistant", content="Postgres.", turn_index=index, thinking_tokens=100 * (index + 1),
                    raw_content=_sent_back(f"Eq{index:04d}{SIGNATURE}")) for index in range(4)]

        for strip_before in range(5):
            rendered = rewrite_messages({}, turns, strip_thinking_before=strip_before)["messages"]
            kept = sum(turn.thinking_tokens for turn, sent in zip(turns, rendered, strict=True)
                       if isinstance(sent["content"], list) and any(b["type"] == "thinking" for b in sent["content"]))
            assert sent_thinking_tokens(turns, strip_thinking_before=strip_before) == kept, strip_before

    @staticmethod
    def _thinking_provider():
        """A provider whose replies think, reporting a prompt equal to LLM-GC's text count. Like an older
        model, it does not bill old thinking, so any thinking LLM-GC adds to its rate would show."""
        replies = itertools.count()

        def upstream(request: httpx2.Request) -> httpx2.Response:
            text = sum(turn.token_count for turn in extract_messages(json.loads(request.content)))
            return httpx2.Response(200, json={**UPSTREAM_OK, "content": _sent_back(f"Eq{next(replies):04d}{SIGNATURE}"),
                                              "usage": {"input_tokens": text, "output_tokens": 400}})
        return upstream

    @staticmethod
    def _counted(client, body: dict) -> list:
        store = client.app.state.reply_thinking
        return extract_messages(body, size_thinking=lambda keys, visible: store.tokens(keys, visible, "m"))

    def _converse(self, client, seen, turns: int):
        """Yield (payload, forwarded body, request record) for each request of one conversation."""
        history: list[dict] = []
        for index in range(turns):
            history.append({"role": "user", "content": f"question {index}: the storage layer, retention, rollout"})
            payload = {"model": "m", "max_tokens": 64, "messages": list(history)}
            reply = client.post("/v1/messages", json=payload, headers={"x-api-key": API_KEY, SESSION_HEADER: "s"})
            yield payload, json.loads(seen[-1].content), client.app.state.metrics.request_log.records()[-1]
            history.append({"role": "assistant", "content": reply.json()["content"]})

    def test_both_counts_match_the_bytes_before_and_after_a_collection(self):
        """Each request's counts must equal an independent reading of what was sent: the client's request
        holds every turn's thinking; the forwarded one holds only what survived the strip."""
        client, _, _, seen = _proxy(self._thinking_provider())

        def total(body: dict) -> int:
            return sum(turn.token_count + turn.thinking_tokens for turn in self._counted(client, body))

        stripped = kept_after_front = 0
        for index, (payload, sent, record) in enumerate(self._converse(client, seen, 10)):
            assert record.tokens_from_client == total(payload), index
            assert record.tokens_forwarded == total(sent), index
            if _is_front(sent["messages"][0]):
                stripped += len(thinking_keys_in(sent)) < len(thinking_keys_in(payload))
                kept_after_front += bool(thinking_keys_in(sent))

        assert stripped and kept_after_front, "a collection stripped old thinking, and newer thinking was kept"

    def test_an_estimate_never_feeds_its_own_rate(self):
        """With the rate measured on requests that carried thinking, each reply's estimate grew with the
        last, without limit, whenever the provider did not bill the thinking."""
        client, _, _, seen = _proxy(self._thinking_provider(), min_cut_tokens=10_000)
        *_, (payload, _, _) = self._converse(client, seen, 8)

        sizes = [turn.thinking_tokens for turn in self._counted(client, payload) if turn.role == "assistant"]
        assert sizes == [400 - count_tokens("Postgres.")] * 7


def thinking_keys_in(body: dict) -> list[str]:
    return [key for message in body["messages"] for key in thinking_keys(message["content"])]


class TestAModelSwitch:
    def test_the_conversation_continues_when_the_client_drops_earlier_thinking(self):
        """Live run: moving to another model, Claude Code sent the history without the earlier thinking."""
        replies = itertools.count()

        def upstream(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json={**UPSTREAM_OK, "content": _sent_back(f"Eq{next(replies):04d}{SIGNATURE}")})

        client, service, _, _ = _proxy(upstream)
        history: list[dict] = []
        for index, model in enumerate([MODEL, MODEL, "claude-sonnet-5-5"]):
            if model != MODEL:
                history = [{**turn, "content": [b for b in turn["content"] if b["type"] != "thinking"]}
                           if turn["role"] == "assistant" else turn for turn in history]
            history.append({"role": "user", "content": f"question {index}: the storage layer and its retention"})
            reply = client.post("/v1/messages", json={"model": model, "max_tokens": 64, "messages": history},
                                headers={"x-api-key": API_KEY})
            history.append({"role": "assistant", "content": reply.json()["content"]})

        assert service.session_manager.store.count() == 1, "one conversation, not one per model"
        assert client.get("/stats").json()["sessions_seeded"] == 1
