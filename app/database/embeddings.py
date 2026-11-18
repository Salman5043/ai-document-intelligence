
from functools import lru_cache

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


@lru_cache(maxsize=1)
def get_embedding_model():
    return TextEmbedding(model_name=MODEL_NAME)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple text chunks."""
    if not texts:
        return []

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All embedding inputs must be non-empty strings.")

    model = get_embedding_model()

    embeddings = list(model.embed(texts))

    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding model returned an unexpected result count.")

    results = [embedding.tolist() for embedding in embeddings]

    if any(len(vector) != EMBEDDING_DIMENSIONS for vector in results):
        raise RuntimeError("Unexpected embedding dimensions.")

    return results


def generate_embedding(text: str) -> list[float]:
    """Generate one embedding, preserving the existing API."""
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    return generate_embeddings([text])[0]
