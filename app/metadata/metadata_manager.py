import json
import os

METADATA_FILE = "metadata/documents.json"


def load_metadata():
    """
    Load document metadata from JSON file.
    """
    if not os.path.exists(METADATA_FILE):
        return {}

    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    """
    Save document metadata to JSON file.
    """
    os.makedirs("metadata", exist_ok=True)

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def document_changed(filename, current_hash):
    metadata = load_metadata()

    if filename not in metadata:
        return True

    return metadata[filename]["hash"] != current_hash