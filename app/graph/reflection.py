from langchain_core.messages import HumanMessage
from llm import llm


def reflection_node(state):
    """
    Decide whether the answer is good enough
    or if another retrieval attempt is needed.
    """

    answer = state["answer"]
    question = state["question"]
    sources = state.get("sources", [])

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # ----------------------------------------
    # Stop retrying after max attempts
    # ----------------------------------------
    if retry_count >= max_retries:

        print("\n========== Reflection ==========")
        print("Maximum retries reached.")
        print("Decision : END")
        print("================================\n")

        return {
            "reflection": "good"
        }

    # ----------------------------------------
    # No retrieved documents
    # ----------------------------------------
    if len(sources) == 0:

        print("\n========== Reflection ==========")
        print("No retrieved sources.")
        print("Decision : RETRY")
        print("================================\n")

        return {
            "reflection": "retry",
            "retry_count": retry_count + 1
        }

    # ----------------------------------------
    # Ask the LLM to judge
    # ----------------------------------------

    prompt = f"""
You are evaluating the quality of a RAG answer.

Question:
{question}

Answer:
{answer}

Number of retrieved sources:
{len(sources)}

Should the system retry retrieval?

Return ONLY one word.

good

or

retry
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    decision = response.content.strip().lower()

    if "retry" in decision:
        decision = "retry"
    else:
        decision = "good"

    print("\n========== Reflection ==========")
    print(f"Retry Count : {retry_count}")
    print(f"Decision    : {decision}")
    print("================================\n")

    if decision == "retry":
        retry_count += 1

    return {
        "reflection": decision,
        "retry_count": retry_count
    }