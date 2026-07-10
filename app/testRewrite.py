from query_rewriter import rewrite_query

question = "my car is not starting"
profile_path="../knowledge_base/customer_support_profile.json"

queries = rewrite_query(question,profile_path)

print("Original Question:")
print(question)

print("\nRewritten Queries:\n")

for q in queries:
    print(q)