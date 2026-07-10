from chat import ask_question

answer, queries, sources = ask_question(
    "my car is not starting"
)

print(answer)

print()

print(queries)

print()

print(len(sources))