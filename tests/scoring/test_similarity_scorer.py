import numpy
import pytest

from llm_gc.models import Message
from llm_gc.scoring.similarity_scorer import SimilarityScorer

class FakeEmbeddingModel:
    """A fake SentenceTransformer that returns predictable vectors based on text content."""

    def __init__(self, embedding_map: dict[str, list[float]]):
        self.embedding_map = embedding_map

    def encode(self, texts: list[str]) -> numpy.ndarray:
        return numpy.array([self.embedding_map[text] for text in texts])

@pytest.fixture
def identical_model():
    """Model where two texts get the same vector - cosine similarity = 1.0."""
    return FakeEmbeddingModel({
        "How do I hash passwords?": [1.0, 0.0, 0.0],
        "Use bcrypt for password hashing": [1.0, 0.0, 0.0],
    })

@pytest.fixture
def orthogonal_model():
    """Model where two texts get perpendicular vectors - cosine similarity = 0.0."""
    return FakeEmbeddingModel({
        "How do I hash passwords?": [1.0, 0.0, 0.0],
        "The weather is nice today": [0.0, 1.0, 0.0],
    })

@pytest.fixture
def opposite_model():
    """Model where two texts get opposite vectors - cosine similarity = -1.0."""
    return FakeEmbeddingModel({
        "How do I hash passwords?": [1.0, 0.0, 0.0],
        "Something totally opposite": [-1.0, 0.0, 0.0],
    })

@pytest.fixture
def partial_similarity_model():
    """Model where two texts have partial similarity - cosine between 0 and 1."""
    return FakeEmbeddingModel({
        "How do I hash passwords?": [1.0, 0.0, 0.0],
        "Security best practices for web apps": [0.7, 0.7, 0.0],
    })

def test_message_is_query_returns_one(identical_model):
    scorer = SimilarityScorer(identical_model)
    msg = Message(role="user", content="How do I hash passwords?")
    conversation = [msg]
    result = scorer.score(msg, conversation)
    assert result.score == 1.0
    assert result.reason == "Message is the same as the query message, score 1.0"

def test_identical_vectors_score_one(identical_model):
    scorer = SimilarityScorer(identical_model)
    msg = Message(role="user", content="Use bcrypt for password hashing")
    query = Message(role="user", content="How do I hash passwords?")
    conversation = [msg, query]
    result = scorer.score(msg, conversation)
    assert result.score == pytest.approx(1.0)

def test_orthogonal_vectors_score_zero(orthogonal_model):
    scorer = SimilarityScorer(orthogonal_model)
    msg = Message(role="user", content="The weather is nice today")
    query = Message(role="user", content="How do I hash passwords?")
    conversation = [msg, query]
    result = scorer.score(msg, conversation)
    assert result.score == pytest.approx(0.0)

def test_negative_similarity_clamped_to_zero(opposite_model):
    scorer = SimilarityScorer(opposite_model)
    msg = Message(role="user", content="Something totally opposite")
    query = Message(role="user", content="How do I hash passwords?")
    conversation = [msg, query]
    result = scorer.score(msg, conversation)
    assert result.score == 0.0
    assert result.signals["cosine_similarity"] < 0

def test_partial_similarity_between_zero_and_one(partial_similarity_model):
    scorer = SimilarityScorer(partial_similarity_model)
    msg = Message(role="user", content="Security best practices for web apps")
    query = Message(role="user", content="How do I hash passwords?")
    conversation = [msg, query]
    result = scorer.score(msg, conversation)
    assert 0.0 < result.score < 1.0

def test_scorer_name_is_similarity(identical_model):
    scorer = SimilarityScorer(identical_model)
    msg = Message(role="user", content="How do I hash passwords?")
    conversation = [msg]
    result = scorer.score(msg, conversation)
    assert result.scorer_name.value == "similarity"

def test_signals_contain_expected_keys(orthogonal_model):
    scorer = SimilarityScorer(orthogonal_model)
    msg = Message(role="user", content="The weather is nice today")
    query = Message(role="user", content="How do I hash passwords?")
    conversation = [msg, query]
    result = scorer.score(msg, conversation)
    assert "cosine_similarity" in result.signals
    assert "similarity_score" in result.signals

def test_query_is_last_message_regardless_of_role(identical_model):
    scorer = SimilarityScorer(identical_model)
    user_msg = Message(role="user", content="How do I hash passwords?")
    assistant_msg = Message(role="assistant", content="Use bcrypt for password hashing")
    conversation = [user_msg, assistant_msg]
    result = scorer.score(user_msg, conversation)
    assert result.score == pytest.approx(1.0)

def test_score_with_empty_content_message():
    model = FakeEmbeddingModel({
        "": [0.1, 0.1, 0.1],
        "How do I hash passwords?": [1.0, 0.0, 0.0],
    })
    scorer = SimilarityScorer(model)
    msg = Message(role="user", content="")
    query = Message(role="user", content="How do I hash passwords?")
    conversation = [msg, query]
    result = scorer.score(msg, conversation)
    assert 0.0 <= result.score <= 1.0