from langchain_core.messages import HumanMessage
from llm import llm


FAILURE_PHRASES = [
    "i couldn't find",
    "i could not find",
    "not found",
    "not mentioned",
    "insufficient information",
    "provided documentation",
]


def reflection_node(state):

    answer = state["answer"].lower()
    question = state["question"]

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    print("\n========== Reflection ==========")
    print(f"Retry Count : {retry_count}")
    print(f"Max Retries : {max_retries}")

    # ----------------------------------------------------
    # Deterministic failure detection
    # ----------------------------------------------------

    failed = any(
        phrase in answer
        for phrase in FAILURE_PHRASES
    )

    if failed:

        if retry_count < max_retries:

            print("Decision    : retry")
            print("================================\n")

            return {
                "reflection": "retry",
                "retry_count": retry_count + 1,
            }

        print("Decision    : web")
        print("================================\n")

        return {
            "reflection": "web"
        }

    # ----------------------------------------------------
    # LLM Judge
    # ----------------------------------------------------

    prompt = f"""
Question:
{question}

Answer:
{answer}

Is this answer sufficiently supported by the retrieved context?

Return ONLY one word:

good

or

retry
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    decision = response.content.strip().lower()

    # ----------------------------------------------------
    # Respect max retries
    # ----------------------------------------------------

    if "retry" in decision:

        if retry_count < max_retries:

            print("Decision    : retry")
            print("================================\n")

            return {
                "reflection": "retry",
                "retry_count": retry_count + 1,
            }

        print("Decision    : web")
        print("================================\n")

        return {
            "reflection": "web"
        }

    print("Decision    : good")
    print("================================\n")

    return {
        "reflection": "good"
    }