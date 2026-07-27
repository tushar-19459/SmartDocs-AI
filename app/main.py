from pathlib import Path

from graph.graph import graph
from langchain_core.messages import HumanMessage

from ingest import ingest

DATA_DIR = Path("../data")

config = {
    "configurable": {
        "thread_id": "tesla_chat"
    }
}


def check_document():

    pdfs = list(DATA_DIR.glob("*.pdf"))

    if len(pdfs) == 0:
        raise FileNotFoundError(
            "No PDF found in ../data"
        )

    if len(pdfs) > 1:
        raise RuntimeError(
            f"Expected exactly one PDF, found {len(pdfs)}."
        )

    # ingest() already checks the SHA256 hash.
    # It only rebuilds the knowledge base if needed.
    ingest(str(pdfs[0]))


while True:

    # ----------------------------------------
    # Check whether the PDF changed
    # ----------------------------------------
    check_document()

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    result = graph.invoke(
        {
            "question": question,
            "messages": [
                HumanMessage(content=question)
            ],
            "retry_count": 1,
            "max_retries": 2
        },
        config=config
    )

    print("\nAssistant:\n")
    print(result["answer"])