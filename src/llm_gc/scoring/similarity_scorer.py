import numpy
from sentence_transformers import SentenceTransformer

from llm_gc.models import Message
from llm_gc.scoring import BaseScorer, ScorerResult, RelevanceType

class SimilarityScorer(BaseScorer):
    """Score messages by semantic similarity to the current query using sentence embeddings and cosine similarity.

    Compares the target message's meaning against conversation[-1] (the current topic).
    Uses a SentenceTransformer model (injected via DI) to encode text into 384-dim vectors,
    then measures directional closeness via cosine similarity, clamped to [0.0, 1.0].
    """

    def __init__(self, sentence_transformer_model: SentenceTransformer):
        super().__init__(scorer_name=RelevanceType.SIMILARITY)
        self.sentence_transformer_model = sentence_transformer_model

    def score(self, message: Message, conversation: list[Message]) -> ScorerResult:
        query_message = conversation[-1]
        if message is query_message:
            return ScorerResult(score=1.0, scorer_name=self.scorer_name,
                                reason="Message is the same as the query message, score 1.0",
                                signals={"cosine_similarity": 1.0, "similarity_score": 1.0})
        embeddings = self.sentence_transformer_model.encode([message.content, query_message.content])
        cosine_similarity = numpy.dot(embeddings[0], embeddings[1]) / (numpy.linalg.norm(embeddings[0]) * numpy.linalg.norm(embeddings[1]))
        similarity_score = max(0.0, cosine_similarity)
        return ScorerResult(score=similarity_score, scorer_name=self.scorer_name,
                            reason=f"Message has cosine similarity {cosine_similarity:.4f} with query message, score {similarity_score:.4f}",
                            signals={"cosine_similarity": cosine_similarity, "similarity_score": similarity_score})