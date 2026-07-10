from embeddings import get_embedding
from query_rewriter import rewrite_query
from vector_store import HybridSearch
from reranker import rerank
from generator import generate_answer


PROFILE_PATH = "../knowledge_base/customer_support_profile.json"


def ask_question(question):
    """
    Complete RAG pipeline.

    Returns:
        answer
        rewritten_queries
        reranked_chunks
    """

    # --------------------------------------------------
    # Rewrite Query
    # --------------------------------------------------

    rewritten_queries = rewrite_query(
        question,
        PROFILE_PATH
    )

    # Always include original question

    if question not in rewritten_queries:
        rewritten_queries.insert(0, question)

    # --------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------

    retrieved_chunks = []

    seen = set()

    for query in rewritten_queries:

        embedding = get_embedding(query)

        results = HybridSearch(

            query=query,

            query_embedding=embedding,

            k=10,

            fetch_k=20
        )

        for chunk_id, info in results:

            key = (
                info["metadata"]["page"],
                info["document"]
            )

            if key in seen:
                continue

            seen.add(key)

            retrieved_chunks.append(

                {
                    "document": info["document"],

                    "metadata": info["metadata"],

                    "score": info["fusion_score"],

                    "bm25_rank": info["bm25_rank"],

                    "semantic_rank": info["semantic_rank"]
                }

            )

    # --------------------------------------------------
    # Rerank
    # --------------------------------------------------

    reranked_chunks = rerank(

        question,

        retrieved_chunks,

        top_k=5
    )

    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    answer = generate_answer(

        question,

        reranked_chunks
    )

    return (

        answer,

        rewritten_queries,

        reranked_chunks
    )