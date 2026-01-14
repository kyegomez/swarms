"""
Autonomous Agent with Handoffs Example

This example demonstrates how to combine autonomous agent capabilities
with handoff functionality to create a multi-agent system that can:
1. Use tools (like web search) for autonomous research
2. Delegate tasks to specialized agents based on their expertise
3. Coordinate complex workflows across multiple agents

The example creates:
- Specialized agents for research, code generation, and report writing
- A coordinator agent that can delegate tasks and use tools
- Demonstrates both autonomous tool usage and agent-to-agent handoffs
"""

import os
from dotenv import load_dotenv
import httpx
from loguru import logger

from swarms.structs.agent import Agent
from swarms.utils.any_to_str import any_to_str

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

    if not api_key:
        return "Error: EXA_API_KEY not found in environment variables. Please set it to use the Exa search tool."

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


# Create specialized research agent with web search capability
research_agent = Agent(
    agent_name="ResearchAgent",
    agent_description=(
        "Specializes in researching topics and providing detailed, "
        "factual information. Has access to web search tools for "
        "finding up-to-date information."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
    tools=[exa_search],
    system_prompt=(
        "You are a research specialist. Provide detailed, well-researched "
        "information about any topic, citing sources when possible. "
        "Use the exa_search tool to find current and authoritative information."
    ),
    dynamic_temperature_enabled=True,
    dynamic_context_window=True,
)

# Create specialized code generation agent
code_agent = Agent(
    agent_name="CodeExpertAgent",
    agent_description=(
        "Expert in writing, reviewing, and explaining code across "
        "multiple programming languages. Specializes in clean, "
        "maintainable, and well-documented code."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt=(
        "You are a coding expert. Write, review, and explain code with "
        "a focus on best practices and clean code principles. Always "
        "include proper documentation and error handling."
    ),
    dynamic_temperature_enabled=True,
    dynamic_context_window=True,
)

# Create specialized writing/report agent
writing_agent = Agent(
    agent_name="WritingAgent",
    agent_description=(
        "Skilled in creative and technical writing, content creation, "
        "and editing. Specializes in creating well-structured reports, "
        "documentation, and written content."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt=(
        "You are a writing specialist. Create, edit, and improve written "
        "content while maintaining appropriate tone and style. Focus on "
        "clarity, structure, and professional presentation."
    ),
    dynamic_temperature_enabled=True,
    dynamic_context_window=True,
)

# Create coordinator agent with both tools and handoffs
coordinator = Agent(
    agent_name="CoordinatorAgent",
    agent_description=(
        "Coordinates complex tasks by delegating to specialized agents "
        "and using tools when needed. Can research topics, generate code, "
        "and create reports by coordinating with specialized agents."
    ),
    model_name="anthropic/claude-sonnet-4-5",
    max_loops="auto",
    handoffs=[research_agent, code_agent, writing_agent],
    system_prompt=(
        "You are a coordinator agent with access to both tools and "
        "specialized agents. Analyze tasks and determine the best approach:\n"
        "- Use the exa_search tool for quick web searches\n"
        "- Delegate to ResearchAgent for in-depth research tasks\n"
        "- Delegate to CodeExpertAgent for coding tasks\n"
        "- Delegate to WritingAgent for writing and report creation\n"
        "- You can delegate to multiple agents if a task requires "
        "different expertise\n"
        "- Break down complex tasks into subtasks and coordinate "
        "multiple agents as needed"
    ),
    output_type="all",
    dynamic_temperature_enabled=True,
    dynamic_context_window=True,
    tool_call_summary=False,
    top_p=None,
)

task = "Call the writing agent and ask it to write a report about the latest developments in AI agent frameworks. Only have 2 steps in your plan"

result = coordinator.run(task=task)
print(result)
