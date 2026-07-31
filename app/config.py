from huggingface_hub import login
from pathlib import Path
from dotenv import load_dotenv
import os


BASE_DIR = Path(__file__).resolve().parent.parent

# Load app/.env
load_dotenv(Path(__file__).resolve().parent / ".env")

CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "customer_support"

EMBEDDING_MODEL384 = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL768 = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL1024 = "BAAI/bge-large-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


if HF_TOKEN:
    login(token=HF_TOKEN)