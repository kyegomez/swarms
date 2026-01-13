import os

import httpx
from loguru import logger

from swarms.structs.agent import Agent
from swarms.utils.any_to_str import any_to_str

from dotenv import load_dotenv

load_dotenv()


def exa_search(
    query: str,
) -> str:
    """
    Exa Web Search Tool

    This function provides advanced, natural language web search capabilities
    using the Exa.ai API. It is designed for use by research agents and
    subagents to retrieve up-to-date, relevant information from the web,
    including documentation, technical articles, and general knowledge sources.

    Features:
    - Accepts natural language queries (e.g., "Find the latest PyTorch 2.2.0 documentation on quantization APIs")
    - Returns structured, summarized results suitable for automated research workflows
    - Supports parallel execution for multiple subagents
    - Can be used to search for:
        * Official documentation (e.g., Python, PyTorch, TensorFlow, API docs)
        * Research papers and technical blogs
        * News, regulatory updates, and more

    Args:
        query (str): The natural language search query. Can be a question, a request for documentation, or a technical prompt.

    Returns:
        str: JSON-formatted string containing the search results, including summaries and key insights.

    Example usage:
        exa_search("Show me the latest Python 3.12 documentation on dataclasses")
        exa_search("Recent research on transformer architectures for vision tasks")

    Notes:
        - This tool is ideal for agents that need to quickly gather authoritative information from the web, especially official docs.
        - The Exa API is capable of extracting and summarizing content from a wide range of sources, including documentation sites, arXiv, blogs, and more.
        - For best results when searching for documentation, include the technology/library name and the specific topic or API in your query.

    """
    api_key = os.getenv("EXA_API_KEY")

    characters = 50
    sources = 2

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
    }

    # Payload format for Exa API (see https://docs.exa.ai/reference/search)
    payload = {
        "query": query,
        "type": "auto",
        "numResults": sources,
        "contents": {
            "text": True,
            "summary": {
                "schema": {
                    "type": "object",
                    "required": ["answer"],
                    "additionalProperties": False,
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": (
                                "Key insights and findings from the search result"
                            ),
                        }
                    },
                }
            },
            "context": {"maxCharacters": characters},
        },
    }

    try:

        response = httpx.post(
            "https://api.exa.ai/search",
            json=payload,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()
        json_data = response.json()

        return any_to_str(json_data)

    except Exception as e:
        logger.error(f"Exa search failed: {e}")
        return f"Search failed: {str(e)}. Please try again."


# System prompt for agent persona and guidelines
quant_report_system_prompt = (
    "You are an expert in quantitative trading and financial analysis. "
    "Your role is to generate accurate, insightful, and well-structured reports for finance professionals. "
    "Always base your analysis on recent and reliable data, and ensure your communication is clear and concise."
)

# Task-specific prompt (the particular assignment or question)
quant_report_task_prompt = (
    "Create a comprehensive, data-driven report on the most undervalued and high potential "
    "publicly traded energy stocks as of today.\n"
    "Create a file for your report.\n"
    "Complete this in 2 steps."
)

# Initialize the agent with the system prompt
agent = Agent(
    model_name="gpt-4.1",
    max_loops="auto",
    tools=[exa_search],
    marketplace_prompt_id="1191250b-9fb3-42e0-b0e9-25ec83260ab2",
    top_p=None,
    system_prompt=quant_report_system_prompt,
)

print(
    f"Agent Name: {agent.agent_name} and Description: {agent.agent_description}"
)
print(f"System Prompt: {agent.system_prompt}")
# print(agent.short_memory.return_history_as_string())
# out = agent.run(quant_report_task_prompt)

# print(
#     f"Total tokens: {count_tokens(agent.short_memory.return_history_as_string())}"
# )
