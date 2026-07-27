from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from tools.direct_web_tool import direct_web_tool
from tools.web_tool import web_tool
from config import GROQ_API_KEY
from tools.rag_tool import rag_tool


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0
)

def rag_node(state):
    """
    Executes the RAG pipeline.
    """

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
    """
    Uses the conversation history for normal chat.
    """

    # Use the accumulated conversation history
    response = llm.invoke(state["messages"])

    return {
        "answer": response.content,
        "messages": [
            AIMessage(content=response.content)
        ]
    }



def web_node(state):

    result = direct_web_tool(state["question"])

    answer = (
        "[Fallback: Web Search]\n\n"
        + result["answer"]
    )

    return {
        "answer": answer,
        "sources": result["sources"],
        "search_results": result["search_results"],
        "messages": [
            AIMessage(content=answer)
        ]
    }

def refelect_web_node(state):
    result = web_tool(state["question"])

    answer = (
        "[Fallback: Web Search]\n\n"
        + result["answer"]
    )

    return {
        "answer": answer,
        "sources": result["sources"],
        "search_results": result["search_results"],
        "messages": [
            AIMessage(content=answer)
        ]
    }