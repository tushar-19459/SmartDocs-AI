from pathlib import Path

from graph.graph import graph
from langchain_core.messages import HumanMessage

from ingest import ingest

DATA_DIR = Path("../data")


def check_document():
    pdfs = list(DATA_DIR.glob("*.pdf"))

    if len(pdfs) == 0:
        raise FileNotFoundError("No PDF found in ../data")

    if len(pdfs) > 1:
        raise RuntimeError(
            f"Expected exactly one PDF, found {len(pdfs)}."
        )

    # ingest() decides whether a rebuild is needed
    ingest(str(pdfs[0]))


def ask_question(question: str, thread_id: str = "tesla_chat"):
    """
    Ask a question to the Agentic RAG system.
    Can be used by both the CLI and Streamlit UI.
    """

    # Check if the document has changed before every query
    check_document()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

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

    return result["answer"]


if __name__ == "__main__":

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\nAssistant:\n")
        print(answer)