import chromadb

from config import (
    CHROMA_PATH,
    COLLECTION_NAME
)

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Customer Support Knowledge Base"
    }
)


def add_chunks(ids, documents, embeddings, metadatas):

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

from rank_bm25 import BM25Okapi
import re
import nltk

from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download only once
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text):
    """
    Lowercase, remove punctuation,
    remove stop words, and stem words.
    """

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize
    tokens = text.split()

    # Remove stop words + stem
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return tokens


def BM25Search(query, k=5):

    # Retrieve all documents
    results = collection.get()

    documents = results["documents"]
    metadatas = results["metadatas"]
    ids = results["ids"]

    # Preprocess all documents
    tokenized_docs = [
        preprocess(doc)
        for doc in documents
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)

    # Preprocess query
    query_tokens = preprocess(query)

    # Compute BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Rank documents
    ranked = sorted(
        zip(scores, ids, documents, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:k]

def search(query_embedding, k=5):

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

def HybridSearch(query, query_embedding, k=5, fetch_k=20):
    """
    Hybrid Retrieval using BM25 + Semantic Search
    with Reciprocal Rank Fusion (RRF).
    """

    # -----------------------------
    # Retrieve more candidates
    # -----------------------------
    bm25_results = BM25Search(query, fetch_k)

    semantic_results = search(
        query_embedding,
        fetch_k
    )

    fusion = {}

    RRF_K = 60

    # ===================================================
    # BM25 Contribution
    # ===================================================
    for rank, (bm_score, chunk_id, document, metadata) in enumerate(bm25_results):

        fusion[chunk_id] = {
            "fusion_score": 1 / (RRF_K + rank + 1),
            "bm25_rank": rank + 1,
            "semantic_rank": None,
            "bm25_score": bm_score,
            "document": document,
            "metadata": metadata
        }

    # ===================================================
    # Semantic Contribution
    # ===================================================
    for rank, (
        chunk_id,
        document,
        metadata
    ) in enumerate(
        zip(
            semantic_results["ids"][0],
            semantic_results["documents"][0],
            semantic_results["metadatas"][0]
        )
    ):

        if chunk_id not in fusion:

            fusion[chunk_id] = {
                "fusion_score": 0,
                "bm25_rank": None,
                "semantic_rank": rank + 1,
                "bm25_score": None,
                "document": document,
                "metadata": metadata
            }

        else:
            fusion[chunk_id]["semantic_rank"] = rank + 1

        fusion[chunk_id]["fusion_score"] += 1 / (RRF_K + rank + 1)

    # ===================================================
    # Final Ranking
    # ===================================================
    ranked = sorted(
        fusion.items(),
        key=lambda x: x[1]["fusion_score"],
        reverse=True
    )

    return ranked[:k]