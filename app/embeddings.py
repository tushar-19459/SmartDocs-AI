import torch
from sentence_transformers import SentenceTransformer
from .config import (
    EMBEDDING_MODEL768,
    # EMBEDDING_MODEL384,
    # EMBEDDING_MODEL1024
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"Using device: {device}")

model = SentenceTransformer(
    EMBEDDING_MODEL768,
    device=device
)


def get_embeddings(texts, batch_size=512):
    """
    Generate embeddings for a list of texts.
    """
    print(f"Using device: {device}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings.tolist()


def get_embedding(text):
    """
    Backward compatibility for single text.
    """

    return get_embeddings([text])[0]