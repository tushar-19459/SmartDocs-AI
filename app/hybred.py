from embeddings import get_embedding
from vector_store import search
from query_rewriter import rewrite_query
from reranker import rerank
from generator import generate_answer

profile_path = "../knowledge_base/customer_support_profile.json"

questions = [
    "my car is not starting",
    "the car is overheating",
    "Vehicle will not restart",
    "Low voltage battery alert",
    "Power cycling vehicle",
    "Jump starting low voltage battery"
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
    # Retrieve documents for every rewritten query
    # ---------------------------------------------------
    for query in queries:

        embedding = get_embedding(query)

        results = search(
            embedding,
            k=5
        )

        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        ):

            key = (meta["page"], doc)

            if key in seen:
                continue

            seen.add(key)

            retrieved_chunks.append(
                {
                    "document": doc,
                    "metadata": meta
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

    print(f"After Reranking: {len(reranked_chunks)}")

    print("\nTop Retrieved Chunks:\n")

    for i, chunk in enumerate(reranked_chunks, start=1):

        print("-" * 80)
        print(f"Rank : {i}")
        print(f"Score: {chunk['score']:.4f}")
        print(chunk["metadata"])
        print(chunk["document"][:400])

    # ---------------------------------------------------
    # Generate Answer
    # ---------------------------------------------------
    answer = generate_answer(
        question,
        reranked_chunks
    )

    print("\nAnswer:\n")
    print(answer)

    print("\n" + "-" * 100)