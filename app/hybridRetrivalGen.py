from embeddings import get_embedding
from vector_store import HybridSearch
from query_rewriter import rewrite_query
from reranker import rerank
from generator import generate_answer
from ingest import ingest

from pathlib import Path

DATA_DIR = Path("../data")

pdfs = list(DATA_DIR.glob("*.pdf"))

if len(pdfs) == 0:
    raise FileNotFoundError("No PDF found.")

if len(pdfs) > 1:
    raise RuntimeError(
        f"Expected exactly one PDF, found {len(pdfs)}."
    )

ingest(str(pdfs[0]))

profile_path = "../knowledge_base/customer_support_profile.json"

questions = [
    "what is the car name mention in the document",
    # "my car is not starting",
    # "the car is overheating",
    # "Vehicle will not restart",
    # "Low voltage battery alert",
    # "Power cycling vehicle",
    # "Jump starting low voltage battery"
]

for question in questions:

    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)

    # ---------------------------------------------------
    # Rewrite Query
    # ---------------------------------------------------
    queries = rewrite_query(question, profile_path)

    print("\nRewritten Queries:")
    for q in queries:
        print(f" - {q}")

    retrieved_chunks = []
    seen = set()

    # ---------------------------------------------------
    # Hybrid Retrieval
    # ---------------------------------------------------
    for query in queries:

        embedding = get_embedding(query)

        results = HybridSearch(
            query=query,
            query_embedding=embedding,
            k=10,          # keep top 10 after fusion
            fetch_k=20     # retrieve 20 from BM25 and 20 from semantic
        )

        for chunk_id, item in results:

            key = (item["metadata"]["page"], item["document"])

            if key in seen:
                continue

            seen.add(key)

            retrieved_chunks.append(
                {
                    "document": item["document"],
                    "metadata": item["metadata"],
                    "fusion_score": item["fusion_score"],
                    "bm25_rank": item["bm25_rank"],
                    "semantic_rank": item["semantic_rank"]
                }
            )

    print(f"\nRetrieved Chunks: {len(retrieved_chunks)}")

    # ---------------------------------------------------
    # Rerank
    # ---------------------------------------------------
    reranked_chunks = rerank(
        question,
        retrieved_chunks,
        top_k=5
    )

    print(f"\nAfter Reranking: {len(reranked_chunks)}")

    print("\nTop Retrieved Chunks:\n")

    for i, chunk in enumerate(reranked_chunks, start=1):

        print("-" * 80)
        print(f"Rank           : {i}")
        print(f"Rerank Score   : {chunk['score']:.4f}")
        print(f"Fusion Score   : {chunk['fusion_score']:.4f}")
        print(f"BM25 Rank      : {chunk['bm25_rank']}")
        print(f"Semantic Rank  : {chunk['semantic_rank']}")
        print(chunk["metadata"])
        print(chunk["document"][:400])

    # ---------------------------------------------------
    # Generate Final Answer
    # ---------------------------------------------------
    answer = generate_answer(
        question,
        reranked_chunks
    )

    print("\nAnswer:\n")
    print(answer)

    print("\n" + "-" * 100)