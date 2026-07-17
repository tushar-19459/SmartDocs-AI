from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import AgentState
from graph.router import route_question
from graph.nodes import (
    rag_node,
    direct_node,
    web_node,
)

# ----------------------------------------------------
# Build Graph
# ----------------------------------------------------

builder = StateGraph(AgentState)

# Nodes
builder.add_node("router", route_question)
builder.add_node("direct", direct_node)
builder.add_node("rag", rag_node)
builder.add_node("web", web_node)

# ----------------------------------------------------
# Graph Flow
# ----------------------------------------------------

builder.add_edge(START, "router")


def route(state: AgentState):
    """
    Returns the next node to execute.
    """
    return state["route"]


builder.add_conditional_edges(
    "router",
    route,
    {
        "direct": "direct",
        "rag": "rag",
        "web": "web",
    },
)

builder.add_edge("direct", END)
builder.add_edge("rag", END)
builder.add_edge("web", END)

# ----------------------------------------------------
# Memory
# ----------------------------------------------------

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)