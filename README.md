# Hybrid Retrieval-Augmented Generation (RAG) System

A production-style Retrieval-Augmented Generation (RAG) pipeline that combines keyword search, semantic search, query rewriting, reranking, and LLM-based answer generation to provide accurate responses from large PDF documents.

---

## Features

- PDF ingestion and intelligent text chunking
- Dense vector embeddings using Sentence Transformers
- ChromaDB vector database for semantic retrieval
- BM25 keyword retrieval
- Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
- LLM-powered query rewriting using Groq (Llama 3.3 70B)
- Cross-Encoder reranking for relevance optimization
- Context-aware answer generation using Groq LLM
- Automatic document knowledge profile generation

---

## Pipeline

```
                    PDF Document
                         │
                         ▼
                 PDF Text Extraction
                         │
                         ▼
                  Document Chunking
                         │
                         ▼
              Generate Dense Embeddings
                         │
                         ▼
            Store in Chroma Vector Database
                         │
         ┌───────────────┴────────────────┐
         │                                │
         ▼                                ▼
     BM25 Index                    Vector Database
         │                                │
         └───────────────┬────────────────┘
                         │

=============================================================

                    User Question
                         │
                         ▼
                Query Rewriting (Groq)
                         │
          Multiple Optimized Search Queries
                         │
                         ▼
              Hybrid Retrieval (BM25 + Vector)
                         │
                         ▼
          Reciprocal Rank Fusion (RRF)
                         │
                         ▼
             Candidate Document Chunks
                         │
                         ▼
           Cross-Encoder Reranking
                         │
                         ▼
               Top Relevant Chunks
                         │
                         ▼
          Answer Generation (Groq LLM)
                         │
                         ▼
                  Final Response
```

---

## Technologies Used

- Python
- ChromaDB
- Sentence Transformers
- Rank-BM25
- Cross-Encoder (MS MARCO)
- LangChain Text Splitter
- PyMuPDF
- Groq API (Llama 3.3 70B)

---

## Project Structure

```
app/
│
├── ingest.py
├── vector_store.py
├── embeddings.py
├── bm25.py
├── hybrid.py
├── reranker.py
├── query_rewriter.py
├── generator.py
├── profile_builder.py
├── config.py
└── main.py

knowledge_base/
└── customer_support_profile.json

data/
└── tesla.pdf
```

---

## Retrieval Pipeline

### 1. Query Rewriting

The user's query is rewritten into multiple semantically related search queries using Groq Llama 3.3.

Example:

```
User:
my car is not starting

↓

my car is not starting
Vehicle Not Starting
Tesla Not Turning On
Vehicle Start Failure
Car Won't Start Error
Troubleshooting Start Issue
```

This improves recall by searching the document using multiple formulations of the same intent.

---

### 2. Hybrid Retrieval

For each rewritten query:

- Retrieve Top 20 documents using BM25
- Retrieve Top 20 documents using Semantic Search
- Merge both rankings using Reciprocal Rank Fusion (RRF)
- Return the Top 10 fused candidates

Hybrid retrieval combines the strengths of lexical matching and semantic understanding.

---

### 3. Reciprocal Rank Fusion (RRF)

BM25 and Semantic Search produce independent ranked lists.

RRF combines these rankings using:

```
RRF Score = Σ 1 / (k + rank)
```

where:

- **rank** = document position in a retrieval list
- **k = 60** (standard smoothing constant)

RRF rewards documents that consistently rank highly across multiple retrieval methods without requiring score normalization.

---

### 4. Cross-Encoder Reranking

Candidate chunks are reranked using the **cross-encoder/ms-marco-MiniLM-L-6-v2** model.

Unlike embedding similarity, the Cross-Encoder jointly processes the question and document, producing a relevance score for each pair.

Only the highest-ranked chunks are passed to the LLM.

---

### 5. Answer Generation

The reranked document chunks are provided to the Groq-hosted Llama 3.3 model, which generates an answer grounded only in the retrieved context.

---

## Document Profiling

During ingestion, the system automatically creates a knowledge profile containing:

- Document title
- Domain
- Summary
- Topics
- Technical terminology

The profile guides query rewriting, allowing the LLM to generate search queries using terminology specific to the knowledge base.

---

## Future Improvements

- Hybrid retrieval with weighted RRF
- Parent-Child Retrieval
- Multi-query parallel retrieval
- Metadata filtering
- Multi-document support
- Citation-aware responses
- Streaming responses
- Conversation memory
- Evaluation pipeline (Recall@K, MRR, nDCG)

---

## Example Applications

- Customer Support Assistants
- Technical Documentation Search
- Enterprise Knowledge Bases
- Internal Company Documentation
- Product Manuals
- IT Help Desk Systems

---

## Key Concepts

- Retrieval-Augmented Generation (RAG)
- Hybrid Search
- Semantic Search
- BM25
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Query Rewriting
- Dense Vector Embeddings
- ChromaDB
- Large Language Models (LLMs)