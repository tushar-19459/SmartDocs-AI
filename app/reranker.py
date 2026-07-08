from sentence_transformers import CrossEncoder

# Excellent free reranker
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(question, retrieved_chunks, top_k=5):
    """
    Rerank retrieved chunks using a Cross Encoder.

    Parameters
    ----------
    question : str

    retrieved_chunks : list
        [
            {
                "document": "...",
                "metadata": {...}
            }
        ]

    top_k : int

    Returns
    -------
    Top reranked chunks
    """

    if len(retrieved_chunks) == 0:
        return []

    pairs = [
        (question, chunk["document"])
        for chunk in retrieved_chunks
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, retrieved_chunks),
        key=lambda x: x[0],
        reverse=True
    )

    reranked = []

    for score, chunk in ranked[:top_k]:

        chunk["score"] = float(score)

        reranked.append(chunk)

    return reranked