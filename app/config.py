from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "customer_support"

EMBEDDING_MODEL384 = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL768 = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL1024 = "BAAI/bge-large-en-v1.5"


CHUNK_SIZE = 500

CHUNK_OVERLAP = 100