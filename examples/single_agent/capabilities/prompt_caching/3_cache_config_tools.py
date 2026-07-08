"""
Cache the tool definitions with cache_config.

Tool schemas render before the system prompt and are large + stable, so caching
them helps tool-heavy agents. `cache_tools` is True by default; shown here
explicitly alongside a tool.

    export ANTHROPIC_API_KEY="sk-ant-..."
    python 3_cache_config_tools.py
"""

from swarms import Agent

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Ground every claim in fundamentals, "
    "state assumptions explicitly, and provide analysis only — never "
    "personalized investment advice. " * 200
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get a stock quote for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    }
]

agent = Agent(
    agent_name="Analyst",
    system_prompt=SYSTEM_PROMPT,
    model_name="claude-opus-4-8",
    prompt_caching=True,
    cache_config={"cache_tools": True},  # cache the tool block too
    tools_list_dictionary=TOOLS,
    max_loops=1,
    temperature=None,
)

print(agent.run("What is the quote for AAPL?"))
