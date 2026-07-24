import json
import os
from datetime import datetime

STATE_FILE = "../knowledge_base/document_state.json"


def load_state():

    if not os.path.exists(STATE_FILE):
        return None

    with open(STATE_FILE, "r") as f:

        return json.load(f)


def save_state(pdf_path, sha256):

    state = {

        "file": pdf_path,
        "sha256": sha256,
        "last_updated": datetime.now().isoformat()

    }

    with open(STATE_FILE, "w") as f:

        json.dump(state, f, indent=4)