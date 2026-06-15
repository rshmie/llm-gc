import pytest
from llm_gc.models.message import Message
from pydantic import ValidationError

def test_invalid_role_raises_error():
    with pytest.raises(ValidationError):
        Message(role="doctor", content="hi")

def test_valid_message_creation():
    msg = Message(role="user", content="Hello", token_count=5)
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.token_count == 5

def test_default_token_count():
    msg = Message(role="assistant", content="Hi")
    assert msg.token_count == 0

def test_preview_content():
    msg = Message(role="user", content="Hello, how are you?")
    assert msg.preview() == "Hello, how are you?"


def test_preview_content_with_truncation():
    msg = Message(role="system", content="This is a test message for previewing." * 10, token_count=100)
    assert len(msg.preview(max_length = 50)) == 53
    assert msg.preview().endswith("...")