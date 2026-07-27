"""
Example 2 — GitMCP (free, no API key).

GitMCP turns *any* public GitHub repository into its own MCP documentation
server. You point the URL at a specific ``owner/repo`` and the agent gets
tools to search and read that repo's code and docs. Free, **no auth**.

    Server : https://gitmcp.io/<owner>/<repo>
    Auth   : none
    Tools  : fetch_<repo>_documentation, search_<repo>_code, ...

Because the server is scoped to one repo, this is a clean pattern for
building a documentation assistant for a single project.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    python 2_gitmcp_repo_docs.py
"""

from swarms import Agent

# Any LiteLLM model works; gpt-4o-mini is cheap and good at tool use.
MODEL = "gpt-4o-mini"

# Point GitMCP at whichever repo you want the agent to be an expert on.
OWNER, REPO = "kyegomez", "swarms"

agent = Agent(
    agent_name="GitMCP-Docs-Agent",
    agent_description=(
        f"Documentation expert for the {OWNER}/{REPO} repo via GitMCP."
    ),
    model_name=MODEL,
    mcp_url=f"https://gitmcp.io/{OWNER}/{REPO}",
    max_loops=1,
)

if __name__ == "__main__":
    result = agent.run(
        "Using your tools, show a minimal code example of creating an "
        "Agent with a tool, and cite the file you found it in."
    )
    print(result)
