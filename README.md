# SmartDocs-AI: Agentic Hybrid Retrieval-Augmented Generation (RAG)

A production-style **Agentic Retrieval-Augmented Generation (RAG)** system that combines hybrid retrieval, intelligent query rewriting, reranking, and LangGraph-based workflow orchestration to deliver accurate, context-aware responses from large PDF documents.

The system automatically detects when the source PDF changes using **SHA-256 hashing**, rebuilds the knowledge base only when necessary, and serves answers using an intelligent retrieval pipeline.

---

# Features

## Intelligent Document Ingestion

- Automatic PDF text extraction
- Recursive document chunking
- Dense embedding generation using Sentence Transformers
- Automatic SHA-256 document change detection
- Incremental ingestion (only rebuilds when the PDF changes)
- Automatic ChromaDB refresh
- Automatic Knowledge Profile generation
- Metadata preservation

---

## Retrieval Pipeline

- Dense Vector Search (Semantic Search)
- BM25 Keyword Search
- Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
- LLM-powered Query Rewriting (Groq Llama 3.3 70B)
- Cross-Encoder Reranking
- Context-aware Answer Generation

---

## Agentic Workflow

- LangGraph workflow orchestration
- Intelligent LLM Router
- Shared Graph State
- Modular Tool Architecture
- Conversation Thread Support
- Retry-aware Retrieval
- Easily extensible for additional tools

---

# System Architecture

```
                  PDF Document
                       │
                       ▼
             SHA-256 Change Detection
                       │
        ┌──────────────┴──────────────┐
        │                             │
   Unchanged                    Modified
        │                             │
        ▼                             ▼
 Skip Ingestion             Rebuild Knowledge Base
                                    │
                                    ▼
                          PDF Text Extraction
                                    │
                                    ▼
                            Recursive Chunking
                                    │
                                    ▼
                         Generate Dense Embeddings
                                    │
                                    ▼
                         Store in Chroma Vector DB
                                    │
             ┌──────────────────────┴──────────────────────┐
             │                                             │
             ▼                                             ▼
         BM25 Index                                Vector Database
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                                    ▼
                          Hybrid Retrieval (RRF)

==============================================================

                    User Question
                          │
                          ▼
                    LangGraph Router
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
     Direct Reply      RAG Tool       Web Search*
                            │
                            ▼
                    Query Rewriting
                            │
                            ▼
                     Hybrid Retrieval
                            │
                            ▼
                   Reciprocal Rank Fusion
                            │
                            ▼
                 Cross-Encoder Reranking
                            │
                            ▼
                    Answer Generation
                            │
                            ▼
                     Final Response

* Planned
```

---

# Technologies Used

- Python
- LangGraph
- LangChain
- ChromaDB
- Sentence Transformers
- Cross Encoder
- Rank-BM25
- PyMuPDF
- Groq API
- HuggingFace Transformers
- PyTorch

---

# Project Structure

```
app/
│
├── graph/
│   ├── graph.py
│   ├── nodes.py
│   ├── router.py
│   └── state.py
│
├── tools/
│   └── rag_tool.py
│
├── ingest.py
├── embeddings.py
├── vector_store.py
├── reranker.py
├── generator.py
├── query_rewriter.py
├── document_profiler.py
├── document_state.py
├── hash.py
├── llm.py
├── config.py
└── main.py

knowledge_base/
├── customer_support_profile.json
└── document_state.json

data/
└── your_document.pdf

chroma_db/
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd SmartDocs-AI
```

---

## 2. Create a Conda Environment

Python **3.11** is recommended.

```bash
conda create -n smartdocs python=3.11
conda activate smartdocs
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# GPU Installation (NVIDIA RTX)

If your machine has an NVIDIA GPU (RTX series), install the CUDA-enabled version of PyTorch.

First remove the CPU build if installed:

```bash
pip uninstall torch torchvision torchaudio
```

Install CUDA 12.1 version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU support:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```
True
```

Check the detected GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Example:

```
NVIDIA GeForce RTX 4050 Laptop GPU
```

The embedding model automatically uses CUDA if available:

```
Using device: cuda
```

Otherwise it falls back to CPU.

---

# Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=YOUR_GROQ_API_KEY
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
```

---

# Automatic Document Change Detection

Every time the application starts:

1. Reads the PDF inside the `data/` directory.
2. Computes its SHA-256 hash.
3. Compares it against the previously stored hash.
4. If unchanged:
   - Skips ingestion.
   - Reuses the existing ChromaDB and knowledge profile.
5. If modified:
   - Clears the existing vector database.
   - Rebuilds embeddings.
   - Regenerates the knowledge profile.
   - Updates the stored document hash.

Example:

```
Checking document...

Document unchanged.

Skipping ingestion.
```

or

```
Checking document...

New or modified document detected.

Clearing vector database...
Loading PDF...
Creating chunks...
Generating embeddings...
Knowledge profile saved.
Document state updated.
```

This avoids unnecessary embedding generation and significantly reduces startup time.

---

# Agent Workflow

## Routing

Every user query first passes through the LangGraph Router.

Possible routes:

- Direct
- RAG
- Web (planned)

Example:

```
Hi
```

↓

```
Direct
```

---

```
My Tesla won't start
```

↓

```
RAG
```

---

```
Latest Tesla news
```

↓

```
Web
```

---

# Retrieval Pipeline

## 1. Query Rewriting

The user's question is expanded into multiple semantically related queries using Groq Llama 3.3.

Example:

```
My car is not starting
```

↓

```
My car is not starting
Vehicle not starting
Tesla won't turn on
Vehicle start failure
Low voltage battery issue
```

This improves retrieval recall.

---

## 2. Hybrid Retrieval

For every rewritten query:

- Top 20 BM25 results
- Top 20 Semantic Search results
- Reciprocal Rank Fusion
- Top fused candidates

This combines lexical matching with semantic similarity.

---

## 3. Reciprocal Rank Fusion (RRF)

BM25 and semantic rankings are merged using:

```
RRF Score = Σ 1 / (k + rank)
```

where:

- rank = document position
- k = 60

Chunks that rank highly across both retrieval methods receive higher fusion scores.

---

## 4. Cross-Encoder Reranking

Candidate chunks are reranked using:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike embedding similarity, the Cross Encoder jointly processes the query and candidate chunk, producing more accurate relevance scores.

Only the top-ranked chunks are passed to the LLM.

---

## 5. Answer Generation

The reranked context is provided to the Groq-hosted Llama 3.3 model, which generates grounded answers using only the retrieved information.

---

# LangGraph Components

## Shared State

The graph maintains:

- User question
- Conversation history
- Rewritten queries
- Selected route
- Retrieved sources
- Final answer
- Retry count

Each node reads and updates the shared state.

---

## Nodes

Current graph nodes:

- Router Node
- Direct Response Node
- RAG Node
- Web Node (placeholder)

---

## Conditional Graph

```
START
   │
   ▼
Router
 ├─────────────┬──────────────┐
 ▼             ▼              ▼
Direct        RAG            Web
 │             │              │
 └─────────────┴──────────────┘
               │
               ▼
              END
```

---

# Knowledge Profiling

During ingestion, the system automatically generates a document profile containing:

- Document title
- Domain
- Summary
- Topics
- Technical terminology

The profile is later used during query rewriting to improve retrieval quality.

---

# Future Roadmap

## In Progress

- Conversation Memory
- History-aware Query Rewriting
- Retrieval Retry Loop

## Planned

- Web Search Tool
- Reflection Agent
- Citation-aware Responses
- Parent-Child Retrieval
- Multi-document Support
- Metadata Filtering
- Streaming Responses
- Faithfulness Evaluation
- Recall@K
- MRR
- nDCG
- Hallucination Detection

---

# Example Applications

- Enterprise Knowledge Assistants
- Customer Support Bots
- Technical Documentation Search
- Product Manuals
- Internal Knowledge Bases
- IT Help Desk Systems
- Conversational AI Assistants

---

# Key Concepts

- Retrieval-Augmented Generation (RAG)
- Agentic RAG
- LangGraph
- Hybrid Search
- Semantic Search
- BM25
- Reciprocal Rank Fusion
- Cross-Encoder Reranking
- Query Rewriting
- ChromaDB
- Vector Databases
- Dense Embeddings
- LLM Routing
- Workflow Orchestration
- Automatic Document Version Detection
- Incremental Knowledge Base Updates