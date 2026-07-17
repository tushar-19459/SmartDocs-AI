from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from config import GROQ_API_KEY
from tools.rag_tool import rag_tool


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0
)


def rag_node(state):
    result = rag_tool(state["question"])

    return {
        "answer": result["answer"],
        "queries": result["queries"],
        "sources": result["sources"],
        "messages": [
            AIMessage(content=result["answer"])
        ]
    }


def direct_node(state):
    response = llm.invoke(
        [
            HumanMessage(content=state["question"])
        ]
    )

    return {
        "answer": response.content,
        "messages": [
            AIMessage(content=response.content)
        ]
    }


def web_node(state):
    return {
        "answer": "Web search not implemented yet.",
        "messages": [
            AIMessage(content="Web search not implemented yet.")
        ]
    }