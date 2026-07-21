from retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    ndcg_at_k
)

retrieved = [
    10,
    5,
    2,
    7,
    9
]

relevant = [
    2,
    7
]

print("Recall@5")
print(
    recall_at_k(
        retrieved,
        relevant,
        5
    )
)

print()

print("Precision@5")
print(
    precision_at_k(
        retrieved,
        relevant,
        5
    )
)

print()

print("MRR")
print(
    reciprocal_rank(
        retrieved,
        relevant
    )
)

print()

print("nDCG")
print(
    ndcg_at_k(
        retrieved,
        relevant,
        5
    )
)