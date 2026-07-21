import math


def recall_at_k(retrieved_pages, relevant_pages, k):
    """
    Recall@K

    retrieved_pages : list[int]
    relevant_pages  : list[int]
    """

    retrieved = set(retrieved_pages[:k])
    relevant = set(relevant_pages)

    if len(relevant) == 0:
        return 0.0

    hits = len(retrieved & relevant)

    return hits / len(relevant)


def precision_at_k(retrieved_pages, relevant_pages, k):
    """
    Precision@K
    """

    retrieved = retrieved_pages[:k]

    if len(retrieved) == 0:
        return 0.0

    hits = sum(
        1
        for page in retrieved
        if page in relevant_pages
    )

    return hits / len(retrieved)


def reciprocal_rank(retrieved_pages, relevant_pages):
    """
    Reciprocal Rank
    """

    for rank, page in enumerate(retrieved_pages, start=1):

        if page in relevant_pages:
            return 1 / rank

    return 0.0


import math


def dcg(retrieved_pages, relevant_pages, k):
    """
    Discounted Cumulative Gain
    """

    score = 0.0

    for i, page in enumerate(retrieved_pages[:k]):

        if page in relevant_pages:
            score += 1 / math.log2(i + 2)

    return score


def ndcg_at_k(retrieved_pages, relevant_pages, k):
    """
    Normalized Discounted Cumulative Gain

    Returns a value between 0 and 1.
    """

    actual_dcg = dcg(
        retrieved_pages,
        relevant_pages,
        k
    )

    # Ideal DCG assumes all relevant documents are ranked first
    ideal_dcg = 0.0

    num_relevant = min(len(relevant_pages), k)

    for i in range(num_relevant):
        ideal_dcg += 1 / math.log2(i + 2)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg