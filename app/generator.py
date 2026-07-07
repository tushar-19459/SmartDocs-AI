import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are a customer support assistant.

Answer ONLY using the provided context.

Rules:

- If the answer exists in the context, answer clearly.
- Combine information from multiple context chunks when needed.
- Do not make up information.
- If the answer cannot be found, say:

"I couldn't find this information in the provided documentation."

- Quote important warnings when present.
- Keep answers concise.
"""


def generate_answer(question, retrieved_chunks):

    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context += f"""
                    Document {i}
                    Page: {chunk["metadata"]["page"]}

                    {chunk["document"]}

                    ------------------------------------
                    """

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
                            Question:

                            {question}

                            Context:

                            {context}
                            """
            }
        ]
    )

    return response.choices[0].message.content