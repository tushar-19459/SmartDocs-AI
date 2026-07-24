import json
import os

METADATA_FILE = "../knowledge_base/metadata.json"


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}

    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_metadata(pdf_hash):
    with open(METADATA_FILE, "w") as f:
        json.dump(
            {"pdf_hash": pdf_hash},
            f,
            indent=4
        )