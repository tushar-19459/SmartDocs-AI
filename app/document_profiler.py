import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are an expert knowledge engineer.

You will receive representative chunks sampled from an entire document.

Analyze the COMPLETE document and generate a knowledge profile.

Return ONLY valid JSON.

Schema:

{
    "title":"",
    "domain":"",
    "summary":"",
    "topics":[],
    "terminology":[],
    "troubleshooting_terms":[],
    "user_intents":[]
}

Rules

- summary <=100 words
- topics: 5-10 broad topics
- terminology: 20-30 technical terms
- troubleshooting_terms: 15-20 common warning/error phrases
- user_intents: 15-20 common ways users might ask questions
- Use ONLY terminology present in the document.
"""


def sample_chunks(chunks, sample_size=30):
    """
    Uniformly sample chunks across the entire document.
    """

    if len(chunks) <= sample_size:
        return chunks

    step = len(chunks) / sample_size

    sampled = []

    for i in range(sample_size):

        index = int(i * step)

        sampled.append(chunks[index])

    return sampled


def build_document_profile(chunks, sample_size=30):

    sampled_chunks = sample_chunks(
        chunks,
        sample_size
    )

    context = ""

    for i, chunk in enumerate(sampled_chunks):

        context += (
            f"\n========== CHUNK {i+1} ==========\n"
            f"{chunk['text']}\n"
        )

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0,

        response_format={
            "type": "json_object"
        },

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": context
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )


def save_profile(profile, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            profile,
            f,
            indent=4
        )