from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import AgentState
from graph.nodes import rag_node

builder = StateGraph(AgentState)

builder.add_node("rag", rag_node)

builder.add_edge(START, "rag")
builder.add_edge("rag", END)

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)