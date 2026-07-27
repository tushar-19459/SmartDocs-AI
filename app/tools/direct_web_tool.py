import json

from tavily import TavilyClient

from config import TAVILY_API_KEY
from llm import llm

client = TavilyClient(api_key=TAVILY_API_KEY)



def direct_web_tool(question: str):
    """
    Search the web using the document profile as context.
    """

    # --------------------------------------------
    # Generate a better web search query
    # --------------------------------------------

    query_prompt = f"""
    You are generating a web search query.

    User Question:
    {question}

    Generate ONE concise web search query that stays relevant
    to the uploaded document.

    Return ONLY the search query.
    """

    search_query = llm.invoke(query_prompt).content.strip()

    print(f"\nWeb Search Query: {search_query}")

    search = client.search(
        query=search_query,
        search_depth="advanced",
        max_results=5,
    )

    context = "\n\n".join(
        result["content"]
        for result in search["results"]
    )

    answer_prompt = f"""
    User Question:
    {question}

    Web Search Results:
    {context}

    Use the document context together with the web results to answer the user's question.

    If the uploaded document does not explicitly contain the requested information,
    say so clearly instead of making assumptions.

    Provide a concise and accurate answer.
    """

    response = llm.invoke(answer_prompt)

    return {
        "answer": response.content,
        "sources": search["results"],
        "search_results": search,
        "search_query": search_query
    }