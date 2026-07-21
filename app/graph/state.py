from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # User input
    question: str

    # Router
    route: str

    # RAG/Web output
    answer: str
    queries: list
    sources: list
    search_results: dict

    # Reflection
    reflection: str

    # Retry control
    retry_count: int
    max_retries: int