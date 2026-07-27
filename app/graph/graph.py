from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import AgentState
from graph.router import route_question
from graph.nodes import (
    rag_node,
    direct_node,
    web_node,
    refelect_web_node
)
from graph.reflection import reflection_node

# ----------------------------------------------------
# Build Graph
# ----------------------------------------------------

builder = StateGraph(AgentState)

# Nodes
builder.add_node("router", route_question)
builder.add_node("direct", direct_node)
builder.add_node("rag", rag_node)
builder.add_node("web", web_node)
builder.add_node("refelect_web", refelect_web_node)
builder.add_node("reflection", reflection_node)

# ----------------------------------------------------
# Graph Flow
# ----------------------------------------------------

builder.add_edge(START, "router")


def route(state: AgentState):
    """
    Decide which node to execute after routing.
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

# ----------------------------------------------------
# Direct and Web finish immediately
# ----------------------------------------------------

builder.add_edge("direct", END)
builder.add_edge("web", END)

# ----------------------------------------------------
# RAG -> Reflection
# ----------------------------------------------------

builder.add_edge("rag", "reflection")


def reflection_route(state):
    return state["reflection"]


builder.add_conditional_edges(
    "reflection",
    reflection_route,
    {
        "good": END,
        "retry": "rag",
        "web": "refelect_web",
    },
)

# ----------------------------------------------------
# Memory
# ----------------------------------------------------

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)