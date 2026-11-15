
from functools import lru_cache

from fastembed import TextEmbedding


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    return TextEmbedding(model_name=MODEL_NAME)


def generate_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    model = get_embedding_model()

    embeddings = list(model.embed([text]))

    return embeddings[0].tolist()
