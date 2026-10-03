import os

from crewai.tools import tool
from groq import Groq


GROQ_MODEL = "openai/gpt-oss-120b"


@tool("Web Research Tool")
def web_research_tool(query: str) -> str:
    """
    Search the live web for current research information.

    Use this tool when the research question requires:
    - current information
    - recent studies
    - recent statistics
    - recent developments
    - authoritative sources
    - factual evidence from the web
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "ERROR: GROQ_API_KEY is not configured."

    if not query or not query.strip():
        return "ERROR: Search query cannot be empty."

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a web research tool. "
                        "Search the live web and return factual research findings. "
                        "Prefer authoritative, primary, academic, government, "
                        "institutional, and reputable sources. "
                        "Include source names and URLs when available. "
                        "Do not invent sources."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            tools=[
                {
                    "type": "browser_search"
                }
            ],
            tool_choice="required",
            temperature=1,
            max_completion_tokens=4096,
            reasoning_effort="low",
        )

        message = response.choices[0].message

        if not message.content:
            return "No useful web research result was returned."

        return message.content

    except Exception as exc:
        return f"Web research failed: {exc}"
