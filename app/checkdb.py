import chromadb

from config import (
    CHROMA_PATH,
    COLLECTION_NAME
)

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Customer Support Knowledge Base"
    }
)
results = collection.get()

print("No. of IDs:", len(results["ids"]))
print("No. of Documents:", len(results["documents"]))
print("No. of Metadata:", len(results["metadatas"]))