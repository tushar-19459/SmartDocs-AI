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
            ]
        },
        config=config
    )

    print("\nAssistant:\n")
    print(result["answer"])