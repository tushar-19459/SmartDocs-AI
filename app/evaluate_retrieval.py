import json

from tools.rag_tool import rag_tool

from evaluation.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    ndcg_at_k
)

with open("evaluation/test_set.json", "r") as f:
    dataset = json.load(f)

for sample in dataset:

    question = sample["question"]
    relevant_pages = sample["relevant_pages"]

    result = rag_tool(question)

    retrieved_pages = []
    seen = set()    

    for chunk in result["sources"]:
        page = chunk["metadata"]["page"]

        if page not in seen:
            seen.add(page)
            retrieved_pages.append(page)

    print("=" * 70)
    print(question)
    print()

    print("Retrieved:", retrieved_pages)
    print("Relevant :", relevant_pages)

    print()

    print("Recall@5:",
          recall_at_k(retrieved_pages, relevant_pages, 5))

    print("Precision@5:",
          precision_at_k(retrieved_pages, relevant_pages, 5))

    print("MRR:",
          reciprocal_rank(retrieved_pages, relevant_pages))

    print("nDCG:",
          ndcg_at_k(retrieved_pages, relevant_pages, 5))