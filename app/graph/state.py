from typing import TypedDict


class AgentState(TypedDict):
    question: str
    answer: str
    queries: list
    sources: list