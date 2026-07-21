# Agentic Hybrid Retrieval-Augmented Generation (RAG) System

A production-style **Agentic Retrieval-Augmented Generation (RAG)** system that combines hybrid retrieval, intelligent query rewriting, reranking, and LangGraph-based workflow orchestration to deliver accurate, context-aware responses from large PDF documents.

The system uses an **agentic architecture** where a routing agent determines whether a user query should be answered directly, retrieved from the knowledge base, or (future) searched on the web.

---

# Features

### Retrieval Pipeline

* PDF ingestion and intelligent text chunking
* Dense vector embeddings using Sentence Transformers
* ChromaDB vector database
* BM25 keyword retrieval
* Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
* LLM-powered query rewriting (Groq Llama 3.3 70B)
* Cross-Encoder reranking
* Context-aware answer generation
* Automatic document knowledge profile generation

### Agentic Workflow

* LangGraph workflow orchestration
* Modular RAG Tool abstraction
* Shared graph state management
* Intelligent LLM-based routing
* Conditional graph execution
* Conversation thread support (LangGraph Checkpointer)
* Extensible tool-based architecture

---

# System Architecture

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

=============================================================

                    User Question
                         │
                         ▼
                 LangGraph Router
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
   Direct Answer     RAG Pipeline     Web Search*
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

* Web Search planned
```

---

# Technologies Used

* Python
* LangGraph
* LangChain
* ChromaDB
* Sentence Transformers
* Rank-BM25
* Cross Encoder (MS MARCO)
* PyMuPDF
* Groq API (Llama 3.3 70B)

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
├── vector_store.py
├── embeddings.py
├── bm25.py
├── hybrid.py
├── reranker.py
├── query_rewriter.py
├── generator.py
├── profile_builder.py
├── llm.py
├── config.py
└── main.py

knowledge_base/
└── customer_support_profile.json

data/
└── tesla.pdf
```

---

# Agent Workflow

## 1. Routing

Every user question first passes through a LangGraph Router.

The router classifies the request into one of three categories:

* **Direct** – greetings and general conversation
* **RAG** – questions answerable from the uploaded PDF
* **Web** – questions requiring external or real-time information (planned)

Example:

```
User:
Hi

↓

Route:
Direct
```

```
User:
My Tesla won't start

↓

Route:
RAG
```

```
User:
Latest Tesla software update

↓

Route:
Web
```

---

# Retrieval Pipeline

## 1. Query Rewriting

The original user query is expanded into multiple semantically related search queries using Groq Llama 3.3.

Example:

```
User:
My car is not starting

↓

My car is not starting
Vehicle not starting
Tesla won't turn on
Vehicle start failure
Low voltage battery issue
```

This increases retrieval recall by searching with multiple formulations of the same intent.

---

## 2. Hybrid Retrieval

For each rewritten query:

* Top 20 BM25 results
* Top 20 Semantic Search results
* Reciprocal Rank Fusion
* Top 10 fused candidates

This combines lexical matching with semantic similarity.

---

## 3. Reciprocal Rank Fusion (RRF)

The independent BM25 and vector rankings are merged using:

```
RRF Score = Σ 1 / (k + rank)
```

where:

* rank = document position
* k = 60

RRF rewards chunks that consistently rank highly across multiple retrieval strategies.

---

## 4. Cross Encoder Reranking

Candidate chunks are reranked using:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike embedding similarity, the Cross Encoder jointly processes the query and candidate chunk, producing a more accurate relevance score.

Only the highest-ranked chunks are passed to the LLM.

---

## 5. Answer Generation

The final reranked context is provided to the Groq-hosted Llama 3.3 model, which generates a grounded response using only the retrieved information.

---

# LangGraph Components

## State

The graph maintains a shared state containing:

* User question
* Selected route
* Retrieved sources
* Generated answer
* Conversation messages
* Rewritten queries

Each node reads from and updates this shared state.

---

## Nodes

Current graph nodes include:

* Router Node
* Direct Response Node
* RAG Node
* Web Node (placeholder)

Each node performs a single responsibility and returns updates to the graph state.

---

## Conditional Routing

Instead of executing a fixed pipeline, LangGraph dynamically selects the next node based on the router's decision.

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

# Document Profiling

During ingestion, the system automatically creates a knowledge profile containing:

* Document title
* Domain
* Summary
* Topics
* Technical terminology

The profile guides query rewriting by encouraging terminology specific to the knowledge base.

---

# Roadmap

## In Progress

* Conversation Memory
* History-aware Query Rewriting

## Planned

* Web Search Tool Integration
* Reflection & Self-Correction
* Retrieval Retry Loop
* Citation-aware Responses
* Multi-document Support
* Metadata Filtering
* Parent-Child Retrieval
* Streaming Responses
* Evaluation Pipeline
* Faithfulness Evaluation
* Recall@K
* MRR
* nDCG
* Hallucination Detection

---

# Example Applications

* Enterprise Knowledge Assistants
* Customer Support Agents
* Technical Documentation Search
* Product Manuals
* Internal Knowledge Bases
* IT Help Desk Systems
* Conversational AI Assistants

---

# Key Concepts

* Retrieval-Augmented Generation (RAG)
* Agentic RAG
* LangGraph
* Hybrid Search
* Semantic Search
* BM25
* Reciprocal Rank Fusion
* Cross-Encoder Reranking
* Query Rewriting
* Vector Databases
* ChromaDB
* Dense Embeddings
* LLM Routing
* Workflow Orchestration
