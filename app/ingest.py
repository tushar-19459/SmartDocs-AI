import uuid
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from embeddings import get_embeddings
from vector_store import (
    add_chunks,
    reset_collection      # we'll create this
)

from document_profiler import (
    build_document_profile,
    save_profile
)

from hash import get_file_hash
from document_state import (
    load_state,
    save_state
)


def load_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_num, page in enumerate(doc):

        pages.append(
            {
                "page": page_num + 1,
                "text": page.get_text()
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

    print("=" * 60)
    print("Checking document...")
    print("=" * 60)

    current_hash = get_file_hash(pdf_path)

    state = load_state()

    # --------------------------------------------------
    # Skip ingestion if PDF is unchanged
    # --------------------------------------------------

    if state is not None:

        if state["sha256"] == current_hash:

            print("Document unchanged.")
            print("Skipping ingestion.")

            return

    print("New or modified document detected.")
    print()

    # --------------------------------------------------
    # Remove old vectors
    # --------------------------------------------------

    print("Clearing existing vector database...")

    reset_collection()

    # --------------------------------------------------
    # Load PDF
    # --------------------------------------------------

    print("Loading PDF...")

    pages = load_pdf(pdf_path)

    print(f"Loaded {len(pages)} pages")

    # --------------------------------------------------
    # Chunk document
    # --------------------------------------------------

    print("Creating chunks...")

    chunks = chunk_document(pages)

    print(f"Created {len(chunks)} chunks")

    # --------------------------------------------------
    # Build document profile
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    print("Generating embeddings...")

    print("Generating embeddings...")

# --------------------------------------
# Prepare data
# --------------------------------------

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

    # --------------------------------------
    # Generate embeddings in batches (GPU optimized)
    # --------------------------------------

    embeddings = get_embeddings(
        documents,
        batch_size=256   # Try 512 if you have enough GPU memory
    )
    
    # --------------------------------------------------
    # Store vectors
    # --------------------------------------------------

    print("Storing vectors...")

    add_chunks(
        ids,
        documents,
        embeddings,
        metadatas
    )

    # --------------------------------------------------
    # Save new hash
    # --------------------------------------------------

    save_state(
        pdf_path,
        current_hash
    )

    print()
    print(f"Stored {len(chunks)} chunks.")
    print("Document state updated.")