"""Each HTTP request gets one ID, held while it runs, its streamed reply included."""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from llm_gc.web.request_id import REQUEST_ID_CHARS, RequestIdMiddleware, current_request_id


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/id")
    def read_id() -> dict[str, str | None]:
        return {"request_id": current_request_id()}

    @app.get("/stream")
    def stream() -> StreamingResponse:
        def chunks():
            # Runs after the route has returned, while the reply is relayed.
            yield (current_request_id() or "none").encode()

        return StreamingResponse(chunks(), media_type="text/plain")

    return app


def test_each_request_gets_an_id_of_its_own():
    client = TestClient(_app())

    first, second = (client.get("/id").json()["request_id"] for _ in range(2))

    assert first is not None and len(first) == REQUEST_ID_CHARS
    assert second is not None and second != first


def test_the_id_is_still_set_while_a_streamed_reply_is_relayed():
    """A dropped stream is logged after the route returned; it must still name its request."""
    client = TestClient(_app())

    streamed = client.get("/stream").text

    assert streamed != "none" and len(streamed) == REQUEST_ID_CHARS


def test_there_is_no_id_outside_a_request():
    TestClient(_app()).get("/id")

    assert current_request_id() is None
