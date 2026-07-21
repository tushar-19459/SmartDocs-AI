from graph.graph import graph
from langchain_core.messages import HumanMessage

config = {
    "configurable": {
        "thread_id": "tesla_chat"
    }
}

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    result = graph.invoke(
    {
        "question": question,
        "messages": [
            HumanMessage(content=question)
        ],
        "retry_count": 0,
        "max_retries": 2
    },
    config=config
    )

    print("\nAssistant:\n")
    print(result["answer"])