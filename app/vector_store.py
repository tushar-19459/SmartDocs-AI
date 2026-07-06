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

def HybridSearch(query, query_embedding, k=5):
    """
    Hybrid Search using:
    1. BM25 (keyword search)
    2. ChromaDB (semantic search)

    Results are fused using Reciprocal Rank Fusion (RRF).
    """

    # -------------------------------
    # BM25 Results
    # -------------------------------
    bm25_results = BM25Search(query, k)

    # -------------------------------
    # Semantic Results
    # -------------------------------
    semantic_results = search(
        query_embedding,
        k
    )

    fusion = {}

    RRF_K = 60

    # -------------------------------------------------
    # BM25 contribution
    # -------------------------------------------------
    for rank, (score, chunk_id, document, metadata) in enumerate(bm25_results):

        fusion[chunk_id] = {
            "score": 1 / (RRF_K + rank + 1),
            "document": document,
            "metadata": metadata
        }

    # -------------------------------------------------
    # Semantic contribution
    # -------------------------------------------------
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
                "score": 0,
                "document": document,
                "metadata": metadata
            }

        fusion[chunk_id]["score"] += 1 / (RRF_K + rank + 1)

    # -------------------------------------------------
    # Final Ranking
    # -------------------------------------------------
    ranked = sorted(
        fusion.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    return ranked[:k]