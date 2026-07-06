import uuid
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from embeddings import get_embedding

from vector_store import add_chunks

from document_profiler import (
    build_document_profile,
    save_profile
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

    print("Loading PDF...")

    pages = load_pdf(pdf_path)

    print(f"Loaded {len(pages)} pages")

    # --------------------------------------
    # Chunk the document
    # --------------------------------------

    print("Creating chunks...")

    chunks = chunk_document(pages)

    print(f"Created {len(chunks)} chunks")

    # --------------------------------------
    # Build Knowledge Profile
    # --------------------------------------

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

    # --------------------------------------
    # Generate embeddings
    # --------------------------------------

    print("Generating embeddings...")

    ids = []

    documents = []

    embeddings = []

    metadatas = []

    for chunk in chunks:

        ids.append(str(uuid.uuid4()))

        documents.append(chunk["text"])

        embeddings.append(
            get_embedding(chunk["text"])
        )

        metadatas.append(
            {
                "source": pdf_path,
                "page": chunk["page"]
            }
        )

    # --------------------------------------
    # Store in ChromaDB
    # --------------------------------------

    print("Storing vectors in ChromaDB...")

    add_chunks(
        ids,
        documents,
        embeddings,
        metadatas
    )

    print(f"\nSuccessfully stored {len(chunks)} chunks.")