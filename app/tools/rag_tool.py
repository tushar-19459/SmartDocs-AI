from embeddings import get_embedding
from vector_store import HybridSearch
from query_rewriter import rewrite_query
from reranker import rerank
from generator import generate_answer
from langchain_core.messages import HumanMessage, AIMessage
PROFILE_PATH = "../knowledge_base/customer_support_profile.json"


def rag_tool(question: str, messages=None):
    """
    Complete RAG pipeline.

    Returns:
        {
            "question": str,
            "answer": str,
            "sources": list,
            "queries": list
        }
    """

    # ---------------------------------------------------
    # Rewrite Query
    # ---------------------------------------------------
    history = format_chat_history(messages or [])
    
    queries = rewrite_query(
        question=question,
        profile_path=PROFILE_PATH,
        history=history
    )

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
            k=10,
            fetch_k=20
        )

        for chunk_id, item in results:

            key = (
                item["metadata"]["page"],
                item["document"]
            )

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

    # ---------------------------------------------------
    # Rerank
    # ---------------------------------------------------
    reranked_chunks = rerank(
        question,
        retrieved_chunks,
        top_k=5
    )

    # ---------------------------------------------------
    # Generate Answer
    # ---------------------------------------------------
    answer = generate_answer(
        question,
        reranked_chunks
    )

    return {
        "question": question,
        "queries": queries,
        "answer": answer,
        "sources": reranked_chunks
    }


def format_chat_history(messages, max_turns=4):
    """
    Convert the last few conversation turns into text.
    Keeps the prompt small while preserving context.
    """

    if not messages:
        return ""

    # Only keep the last few messages
    recent_messages = messages[-max_turns:]

    history = []

    for message in recent_messages:

        if isinstance(message, HumanMessage):
            history.append(f"User: {message.content}")

        elif isinstance(message, AIMessage):
            history.append(f"Assistant: {message.content}")

    return "\n".join(history)