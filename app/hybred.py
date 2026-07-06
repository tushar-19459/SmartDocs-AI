from embeddings import get_embedding
from vector_store import HybridSearch

question = "my car is not starting"

embedding = get_embedding(question)

results = HybridSearch(
    question,
    embedding,
    k=5
)

for chunk_id, item in results:

    print("-" * 60)

    print("Fusion Score :", round(item["score"], 4))
    print("Chunk ID     :", chunk_id)
    print("Metadata     :", item["metadata"])

    print(item["document"])