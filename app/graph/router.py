from langchain_core.messages import HumanMessage
from llm import llm


def route_question(state):
    """
    Decide which tool should answer the user's question.

    Returns one of:
    - direct
    - rag
    - web
    """

    question = state["question"]

    prompt = f"""
            You are a routing agent for a Retrieval-Augmented Generation (RAG) assistant.

            Your job is to decide how the assistant should answer the user's question.

            Choose exactly ONE route.

            Route: direct
            Use when:
            - greetings
            - casual conversation
            - thank you messages
            - simple questions that need no retrieval

            Route: rag
            Use when the question is about:
            - Tesla manuals
            - troubleshooting
            - vehicle documentation
            - customer support
            - information likely contained in the uploaded PDF

            Route: web
            Use when the question requires:
            - current events
            - latest news
            - recent software versions
            - live information
            - anything not likely contained in the PDF

            Question:
            {question}

            Return ONLY one word.

            direct
            rag
            web
            """

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    route = response.content.strip().lower()

    # Validate output
    if "rag" in route:
        route = "rag"
    elif "web" in route:
        route = "web"
    else:
        route = "direct"

    print("\n========== Router ==========")
    print(f"Question : {question}")
    print(f"Decision : {route}")
    print("============================\n")

    return {
        "route": route
    }