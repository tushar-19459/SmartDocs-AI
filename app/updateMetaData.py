from app.metadata.metadata_manager import load_metadata, save_metadata
from metadata.hashUtils import calculate_file_hash
from datetime import datetime
import os

file_path = "uploads/tesla.pdf"

filename = os.path.basename(file_path)
file_hash = calculate_file_hash(file_path)

metadata = load_metadata()

metadata[filename] = {
    "hash": file_hash,
    "last_updated": datetime.now().isoformat()
}

save_metadata(metadata)