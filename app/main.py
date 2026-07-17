from graph.graph import graph

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
            "question": question
        },
        config=config
    )

    print("\nAssistant:\n")
    print(result["answer"])