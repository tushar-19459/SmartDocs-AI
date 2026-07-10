import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are an expert search query rewriting assistant for Retrieval-Augmented Generation (RAG).

You are given a knowledge profile extracted from an entire document.

Your objective is to translate a user's natural language question into search queries that are most likely to retrieve relevant document chunks.

Instructions:

- Generate EXACTLY 5 search queries.
- Preserve the user's intent.
- Prefer terminology from the document profile.
- Use troubleshooting phrases whenever appropriate.
- Use user intents as examples of how users ask similar questions.
- Convert informal language into document terminology.
- Keep each query concise (3-8 words).
- Do NOT answer the question.
- Return ONLY the queries.
- One query per line.
"""


def rewrite_query(question, profile_path="../knowledge_base/customer_support_profile.json"):

    # -----------------------------
    # Load profile
    # -----------------------------
    with open(profile_path, "r", encoding="utf8") as f:
        profile = json.load(f)

    profile_text = f"""
        Title:
        {profile["title"]}

        Domain:
        {profile["domain"]}

        Summary:
        {profile["summary"]}

        Topics:
        {", ".join(profile["topics"])}

        Technical Terminology:
        {", ".join(profile["terminology"])}

        Common Troubleshooting Terms:
        {", ".join(profile["troubleshooting_terms"])}

        Common User Intents:
        {", ".join(profile["user_intents"])}
        """

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0.2,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Knowledge Base Profile:

{profile_text}

User Question:

{question}
"""
            }
        ]
    )

    response_text = response.choices[0].message.content

    queries = [
        q.strip("-•123456789. ").strip()
        for q in response_text.split("\n")
        if q.strip()
    ]

    # Remove duplicates
    queries = list(dict.fromkeys(queries))

    # Always include the original question
    if question not in queries:
        queries.insert(0, question)

    return queries