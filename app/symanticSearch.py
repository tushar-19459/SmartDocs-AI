from embeddings import get_embedding
from vector_store import search
from query_rewriter import rewrite_query

question = "my car is not starting"

# Generate rewritten queries
queries = rewrite_query(question)

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
        
# import fitz
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer


# data = fitz.open("../data/tesla.pdf")
# for page_No , page in enumerate(data):
#     if page_No==1:
#         text = page.get_text()
#         break

# splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)

# print(text)
# splits = splitter.split_text(text)

# for i in splits:
#     print(i)
#     print()

# print("no of splits ", len(splits))

# mode = SentenceTransformer("BAAI/bge-small-en-v1.5")

# embeddings = []
# for i in splits:
#     embeddings.append(mode.encode(i,normalize_embeddings=True))

# print(embeddings[0])
# print(len(embeddings[0]))
# print(embeddings[0].shape)
