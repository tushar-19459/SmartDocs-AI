from langchain_core.messages import AIMessage

from tools.rag_tool import rag_tool


def rag_node(state):

    question = state["question"]

    result = rag_tool(question)

    return {
        "answer": result["answer"],
        "queries": result["queries"],
        "sources": result["sources"],
        "messages": [
            AIMessage(content=result["answer"])
        ]
    }