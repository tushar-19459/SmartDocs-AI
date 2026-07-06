from vector_store import BM25Search

question = "How do I return my product?"

print(question)

results = BM25Search(question,5)

for score, chunk_id, document, metadata in results:

    print("-" * 60)
    print("Score :", score)
    print("ID    :", chunk_id)
    print("Meta  :", metadata)
    print(document)