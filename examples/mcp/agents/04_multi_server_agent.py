"""
Example 4 — Multiple MCP servers on one agent (all free, no API key).

Pass a list to ``mcp_urls`` and the agent loads the tools from *every*
server and can use them together in a single run. Here we combine two free,
no-auth servers so the agent can cross-reference a GitHub project (DeepWiki)
with official Microsoft documentation (Microsoft Learn).

    Servers : https://mcp.deepwiki.com/mcp
              https://learn.microsoft.com/api/mcp
    Auth    : none

The model sees the union of both toolsets and decides which to call.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    python 4_multi_server_agent.py
"""

from swarms import Agent

# Any LiteLLM model works; gpt-4o-mini is cheap and good at tool use.
MODEL = "gpt-4o-mini"

agent = Agent(
    agent_name="Multi-MCP-Agent",
    agent_description=(
        "Research agent with tools from several free MCP servers."
    ),
    model_name=MODEL,
    mcp_urls=[
        "https://mcp.deepwiki.com/mcp",  # GitHub repo Q&A
        "https://learn.microsoft.com/api/mcp",  # Microsoft docs
    ],
    max_loops=2,  # give the model room to call tools on both servers
)

if __name__ == "__main__":
    result = agent.run(
        "First, use DeepWiki to describe what the modelcontextprotocol/"
        "python-sdk repository does. Then use Microsoft Learn to find how "
        "Azure Functions supports Python. Give one combined summary."
    )
    print(result)
