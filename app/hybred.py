from embeddings import get_embedding
from vector_store import search
from query_rewriter import rewrite_query

question = "my car is not starting"
profile_path="../knowledge_base/customer_support_profile.json"

# Generate rewritten queries
queries = rewrite_query(question,profile_path)

for i, query in enumerate(queries, start=1):

    print("=" * 80)
    print(f"Query {i}: {query}")
    print("=" * 80)

    embedding = get_embedding(query)

    results = search(
        embedding,
        k=5
    )

    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        print("-" * 50)
        print(meta)
        print(doc)