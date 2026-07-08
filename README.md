# Retrieval-Augmented Generation (RAG) Customer Support System

A Retrieval-Augmented Generation (RAG) pipeline built using ChromaDB, Sentence Transformers, BM25, Groq LLMs, and Cross-Encoder reranking to answer customer support questions from PDF documentation.

The system combines semantic search, keyword search, LLM-powered query rewriting, reranking, and answer generation to improve retrieval quality over traditional vector search.

---

## Features

- PDF document ingestion
- Recursive document chunking
- Dense embeddings using BAAI BGE models
- ChromaDB vector database
- BM25 keyword retrieval
- Hybrid Retrieval (Semantic + BM25)
- Reciprocal Rank Fusion (RRF)
- LLM-based Query Rewriting
- Knowledge Base Profile Generation
- Cross-Encoder Reranking
- LLM-based Answer Generation
- Metadata preservation (page number & source)

---

## Tech Stack

- Python
- ChromaDB
- Sentence Transformers
- BAAI/bge-base-en-v1.5
- LangChain Text Splitters
- BM25 (rank_bm25)
- Groq API
- Llama 3.3 70B
- Cross Encoder (BAAI/bge-reranker-base)

---

## Project Structure

```
rag-project/
│
├── app/
│   ├── ingest.py
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── query_rewriter.py
│   ├── reranker.py
│   ├── generator.py
│   ├── search.py
│   ├── hybrid.py
│   └── config.py
│
├── chroma_db/
│
├── data/
│   └── tesla.pdf
│
├── knowledge_base/
│   └── customer_support_profile.json
│
├── requirements.txt
└── README.md
```

---

# Pipeline

```
                PDF Document
                      │
                      ▼
              PDF Text Extraction
                      │
                      ▼
              Recursive Chunking
                      │
                      ▼
              Generate Embeddings
                      │
                      ▼
               Store in ChromaDB
                      │
                      │
────────────────────────────────────────────────────────

                 User Question
                      │
                      ▼
           Query Rewriting (Groq LLM)
                      │
                      ▼
          Multiple Search Queries
                      │
                      ▼
        Dense Semantic Retrieval
             (ChromaDB Search)
                      │
                      ▼
           Collect Candidate Chunks
                      │
                      ▼
        Cross Encoder Reranking
                      │
                      ▼
         Top Ranked Relevant Chunks
                      │
                      ▼
      Answer Generation (Groq LLM)
                      │
                      ▼
              Final Response
```

---

# Knowledge Base Profiling

During ingestion, the first few document chunks are analyzed using an LLM to automatically create a knowledge profile containing:

- Document title
- Domain
- Summary
- Topics
- Technical terminology

This profile is later used to rewrite user queries into domain-specific search queries, improving retrieval performance.

Example:

```json
{
    "title": "Model 3 Owner's Manual",
    "domain": "Electric Vehicles",
    "summary": "...",
    "topics": [
        "Vehicle Operation",
        "Safety",
        "Charging"
    ],
    "terminology": [
        "Supercharger",
        "Autopilot",
        "Low Voltage Battery",
        "Thermal Management",
        "Regenerative Braking"
    ]
}
```

---

# Retrieval Pipeline

Instead of searching only the original user query:

```
"My car is not starting"
```

the system first rewrites it into multiple search queries:

```
My car is not starting
Vehicle Not Starting
Tesla Not Turning On
Car Won't Start Error
Troubleshooting Start Issue
Battery Not Charging
```

Each rewritten query retrieves the top semantic matches from ChromaDB.

Duplicate chunks are removed before reranking.

---

# Cross Encoder Reranking

The retrieved chunks are reranked using the BAAI Cross Encoder:

```
Question + Chunk
        │
        ▼
Relevance Score
```

Unlike embedding similarity, the Cross Encoder jointly processes the query and document, producing a more accurate relevance score.

Only the highest-ranked chunks are passed to the language model.

---

# Answer Generation

The final response is generated using Groq's Llama 3.3 70B model.

The model is instructed to:

- Answer only using the retrieved context
- Cite page numbers when appropriate
- Avoid hallucinations
- State when the answer is unavailable in the provided documentation

---

# Example Workflow

User Question:

```
How do I jump start the low voltage battery?
```

Pipeline:

```
Question
    ↓
Rewrite Query
    ↓
Semantic Retrieval
    ↓
Deduplicate Chunks
    ↓
Cross Encoder Reranking
    ↓
Top Relevant Chunks
    ↓
Groq LLM
    ↓
Final Answer
```

---

# Future Improvements

- Hybrid Search (BM25 + Semantic Retrieval)
- Reciprocal Rank Fusion (RRF)
- Multi-query Retrieval
- Metadata Filtering
- Parent-Child Retrieval
- Context Compression
- Multi-document Support
- Streaming Responses
- Conversation Memory
- Evaluation using RAGAS
- FAISS Backend Support
- REST API with FastAPI
- Docker Deployment

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd rag-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

# Run

Ingest documents:

```bash
python ingest.py
```

Run retrieval:

```bash
python search.py
```

Run the complete RAG pipeline:

```bash
python main.py
```

---

# Learning Outcomes

This project demonstrates practical implementation of modern Retrieval-Augmented Generation techniques, including:

- Dense Vector Retrieval
- Sentence Embeddings
- ChromaDB
- BM25
- Reciprocal Rank Fusion
- Query Expansion using LLMs
- Cross Encoder Reranking
- Prompt Engineering
- Knowledge Base Profiling
- Large Language Model Integration
- End-to-End RAG Pipeline Design