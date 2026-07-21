from tavily import TavilyClient
from config import TAVILY_API_KEY
from llm import llm

client = TavilyClient(api_key=TAVILY_API_KEY)


def web_tool(question: str):
    """
    Search the web and summarize the results.
    """

    search = client.search(
        query=question,
        search_depth="advanced",
        max_results=5,
    )

    context = "\n\n".join(
        result["content"]
        for result in search["results"]
    )

    prompt = f"""
            Use the web search results below to answer the user's question.

            Question:
            {question}

            Web Results:
            {context}

            Provide a concise and accurate answer.
            """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": search["results"],
        "search_results": search
    }