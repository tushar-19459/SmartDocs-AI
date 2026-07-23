import uuid
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from embeddings import get_embeddings

from vector_store import add_chunks

from document_profiler import (
    build_document_profile,
    save_profile
)

import os
from datetime import datetime

from metadata.hashUtils import calculate_file_hash
from metadata.metadata_manager import (
    document_changed,
    load_metadata,
    save_metadata,
)

def load_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_num, page in enumerate(doc):

        text = page.get_text()

        pages.append(
            {
                "page": page_num + 1,
                "text": text
            }
        )

    doc.close()

    return pages


def chunk_document(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for page in pages:

        splits = splitter.split_text(page["text"])

        for split in splits:

            chunks.append(
                {
                    "text": split,
                    "page": page["page"]
                }
            )

    return chunks


def ingest(pdf_path):
    filename = os.path.basename(pdf_path)
    current_hash = calculate_file_hash(pdf_path)

    if not document_changed(filename, current_hash):
        print(f"{filename} has not changed. Skipping ingestion.")
        return
    
    print("Loading PDF...")

    pages = load_pdf(pdf_path)

    print(f"Loaded {len(pages)} pages")

    print("Creating chunks...")

    chunks = chunk_document(pages)

    print(f"Created {len(chunks)} chunks")

    print("Building knowledge profile...")

    profile = build_document_profile(
        chunks,
        sample_size=30
    )

    save_profile(
        profile,
        "../knowledge_base/customer_support_profile.json"
    )

    print("Knowledge profile saved.")

    print("Preparing data...")

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": pdf_path,
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    print("Generating embeddings on GPU...")

    embeddings = get_embeddings(
        documents,
        batch_size=64
    )

    print("Storing vectors in ChromaDB...")

    add_chunks(
        ids,
        documents,
        embeddings,
        metadatas
    )

    print(f"\nSuccessfully stored {len(chunks)} chunks.")
    metadata = load_metadata()

    metadata[filename] = {
        "hash": current_hash,
        "chunks": len(chunks),
        "last_updated": datetime.now().isoformat()
    }

    save_metadata(metadata)

    print("Metadata updated.")