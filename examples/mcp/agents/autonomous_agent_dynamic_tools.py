"""
Autonomous agent + MCP server, with tools loaded on demand.

The agent connects to a public MCP server, defers every tool it exposes, and
loads only the ones a task actually needs. Nothing else is sent on each
request, which matters because one server can expose dozens of tools and every
schema is re-sent with every call.

    python3 mcp_auto_agent_example.py

Needs OPENAI_API_KEY (read from .env). The MCP server used here, DeepWiki,
needs no key of its own.

Requires mcp>=1.28.1,<2.0.0. On mcp 2.x the connection fails with
`cannot import name 'streamablehttp_client'`, because 2.0 renamed it:
    pip install 'mcp>=1.28.1,<2.0.0'
"""

import json

from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

MCP_SERVER = "https://mcp.deepwiki.com/mcp"

agent = Agent(
    agent_name="RepoResearcher",
    model_name="gpt-4o-mini",
    max_loops="auto",
    mcp_url=MCP_SERVER,
    mcp_timeout=120,  # read_wiki_contents returns a lot; 30s is not enough
    dynamic_tools=True,  # <- MCP tools go into a searchable catalog
    print_on=False,
)

# Building the LLM is what pulls the server's tools into the catalog.
agent.llm = agent.llm_handling()

catalog = (
    agent.tool_loader.deferred_names if agent.tool_loader else []
)
exposed = [t["function"]["name"] for t in agent.tools_list_dictionary]

print(f"MCP server:        {MCP_SERVER}")
print(f"tools in catalog:  {len(catalog)}  {catalog}")
print(f"tools sent per request: {len(exposed)}  {exposed}")
print(
    f"schema bytes sent: {len(json.dumps(agent.tools_list_dictionary)):,}"
)

if not catalog:
    print(
        "\nNo MCP tools were loaded - the server could not be reached. "
        "Check the mcp version note in this file's docstring."
    )
    raise SystemExit(1)

result = agent.run(
    "What is the kyegomez/swarms repository for? Search for a tool that can "
    "answer questions about a GitHub repository, use it, and summarise the "
    "answer in three sentences."
)

print(f"\nloaded during the run: {agent.tool_loader.loaded_names}")
print(
    f"still deferred:        {len(agent.tool_loader.deferred_names)}"
)
print("\n--- answer ---")
print(str(result).split("Subtask Breakdown:")[0].strip()[-700:])
