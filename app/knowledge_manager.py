import os

from file_hash import get_file_hash
from metadata import load_metadata, save_metadata
from ingest import ingest
from vector_store import (
    is_collection_empty,
    clear_collection,
)

PDF_PATH = "../data/tesla.pdf"
PROFILE_PATH = "../knowledge_base/customer_support_profile.json"


def ensure_knowledge_base():
    """
    Ensures the knowledge base is ready before answering questions.

    Rebuilds the knowledge base if:
    - ChromaDB is empty
    - Document profile is missing
    - metadata.json is missing or outdated
    """

    rebuild = False

    print("\n========== Knowledge Base Check ==========")

    # ----------------------------------------
    # Check 1: ChromaDB
    # ----------------------------------------

    if is_collection_empty():
        print("✓ ChromaDB is empty.")
        rebuild = True
    else:
        print("✓ ChromaDB found.")

    # ----------------------------------------
    # Check 2: Document Profile
    # ----------------------------------------

    if not os.path.exists(PROFILE_PATH):
        print("✓ Knowledge profile not found.")
        rebuild = True
    else:
        print("✓ Knowledge profile found.")

    # ----------------------------------------
    # Check 3: Document Hash
    # ----------------------------------------

    metadata = load_metadata()

    current_hash = get_file_hash(PDF_PATH)
    stored_hash = metadata.get("pdf_hash")

    if stored_hash is None:
        print("✓ metadata.json not found.")
        rebuild = True

    elif current_hash != stored_hash:
        print("✓ Document has changed.")
        rebuild = True

    else:
        print("✓ Document is unchanged.")

    # ----------------------------------------
    # Rebuild if required
    # ----------------------------------------

    if rebuild:

        print("\nRebuilding knowledge base...\n")

        # Remove old vectors (safe even if empty)
        clear_collection()

        # Rebuild profile + embeddings
        ingest(PDF_PATH)

        # Save new hash
        save_metadata(current_hash)

        print("\nKnowledge base successfully updated.")
    else:
        print("\nKnowledge base is already up to date.")

    print("==========================================\n")