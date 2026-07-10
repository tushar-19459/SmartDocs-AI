# Intelligent Document Question Answering (Hybrid RAG)

A production-style **Hybrid Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents, build a searchable knowledge base, and ask natural language questions grounded in the document content.

The system combines **semantic search**, **keyword search**, **query rewriting**, **reranking**, and **LLM-powered answer generation** to deliver accurate and context-aware responses.

---

# Features

- 📄 PDF upload through Streamlit
- 📑 Automatic PDF text extraction
- ✂ Intelligent document chunking
- ⚡ GPU-accelerated embedding generation using Sentence Transformers
- 🧠 Semantic search using ChromaDB
- 🔍 BM25 keyword retrieval
- 🔀 Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
- ✨ LLM-based query rewriting
- 🎯 Cross-Encoder reranking for improved retrieval quality
- 🤖 Context-aware answer generation using Groq/Gemini
- 📚 Automatic knowledge profile generation
- 💬 Interactive chat interface with conversation history
- 📖 Source chunk visualization

---

# System Architecture

```
                      PDF Document
                           │
                           ▼
                  PDF Text Extraction
                           │
                           ▼
                  Intelligent Chunking
                           │
                           ▼
           GPU SentenceTransformer Embeddings
                           │
                           ▼
                Chroma Vector Database
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
   BM25 Retriever                    Semantic Search
        │                                     │
        └──────────────────┬──────────────────┘
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
                 Most Relevant Context
                           │
                           ▼
              Query Rewriting (LLM)
                           │
                           ▼
          Groq / Gemini Answer Generation
                           │
                           ▼
                   Final Response
```

---

# Technologies

## Frontend

- Streamlit

## Retrieval

- ChromaDB
- Sentence Transformers
- Rank-BM25
- LangChain Text Splitter

## LLM

- Groq
- Google Gemini

## Document Processing

- PyMuPDF
- PyPDF

## NLP

- Cross Encoder (MS MARCO)
- NLTK

## Utilities

- NumPy
- Pandas
- tqdm

---

# Project Structure

```
project/
│
├── app.py                     # Streamlit application
├── ingest.py                  # PDF ingestion pipeline
├── embeddings.py              # GPU embedding generation
├── vector_store.py            # ChromaDB operations
├── hybrid.py                  # Hybrid retrieval
├── bm25.py                    # BM25 indexing
├── reranker.py                # Cross encoder reranker
├── generator.py               # LLM answer generation
├── query_rewriter.py          # Query rewriting
├── document_profiler.py       # Knowledge profile generation
├── chat.py                    # QA pipeline
├── config.py
│
├── uploads/
│
├── knowledge_base/
│   └── customer_support_profile.json
│
└── chroma_db/
```

---

# Installation

## Create Conda Environment

```bash
conda create -n rag python=3.11 -y
conda activate rag
```

---

## Install PyTorch (GPU)

For CUDA 12.6

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU

```bash
python -c "import torch;print(torch.cuda.is_available());print(torch.cuda.get_device_name(0))"
```

Expected output

```
True
NVIDIA RTX ...
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will be available at

```
http://localhost:8501
```

---

# Retrieval Pipeline

## 1. Document Ingestion

The uploaded PDF is

- parsed using PyMuPDF
- split into overlapping chunks
- converted into dense embeddings on the GPU
- stored inside ChromaDB

During ingestion, BM25 indexes are also built for lexical retrieval.

---

## 2. Document Profiling

The system automatically generates a knowledge profile containing

- Document title
- Domain
- Summary
- Topics
- Keywords

This profile improves query rewriting by providing domain-specific context.

---

## 3. Query Rewriting

User questions are expanded into multiple semantically related queries.

Example

```
User

Battery issue

↓

Battery issue
Battery not charging
Charging problems
Power system fault
Charging troubleshooting
```

This improves retrieval recall.

---

## 4. Hybrid Retrieval

Each rewritten query searches

- BM25 Index
- Chroma Vector Store

The retrieved candidates are merged using

**Reciprocal Rank Fusion (RRF)**

which combines lexical and semantic retrieval strengths.

---

## 5. Cross Encoder Reranking

Candidate chunks are reranked using

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike embedding similarity, the Cross Encoder jointly evaluates

```
Question + Document Chunk
```

producing a more accurate relevance score.

---

## 6. Answer Generation

The highest-ranked chunks are passed to the configured LLM

Supported providers include

- Groq
- Google Gemini

The model generates answers grounded entirely in the retrieved context.

---

# GPU Acceleration

Embedding generation uses **Sentence Transformers** with CUDA when available.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

SentenceTransformer(
    MODEL_NAME,
    device=device
)
```

Embeddings are generated in batches for significantly faster indexing of large documents.

---

# Streamlit Interface

The application provides

- PDF upload
- One-click knowledge base creation
- Interactive chat
- Conversation history
- Rewritten query inspection
- Source document visualization

---

# Future Improvements

- Multi-document collections
- Incremental indexing
- Metadata filtering
- Streaming LLM responses
- Parent-child retrieval
- Query caching
- Evaluation metrics (Recall@K, MRR, nDCG)
- Multi-modal document support
- OCR support for scanned PDFs
- Docker deployment

---

# Applications

- Enterprise Knowledge Bases
- Customer Support
- Technical Documentation
- Internal Company Wikis
- Product Manuals
- Research Papers
- Standard Operating Procedures
- Policy Documents

---

# Key Concepts

- Retrieval-Augmented Generation (RAG)
- Hybrid Search
- Semantic Search
- BM25
- ChromaDB
- Dense Embeddings
- Sentence Transformers
- GPU Embedding Generation
- Reciprocal Rank Fusion (RRF)
- Cross Encoder Reranking
- Query Rewriting
- Large Language Models
- Streamlit