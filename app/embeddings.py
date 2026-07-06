from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL768,EMBEDDING_MODEL1024,EMBEDDING_MODEL384 

# model = SentenceTransformer(EMBEDDING_MODEL384)
model = SentenceTransformer(EMBEDDING_MODEL768)
# model = SentenceTransformer(EMBEDDING_MODEL1024)


def get_embedding(text: str):
    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()