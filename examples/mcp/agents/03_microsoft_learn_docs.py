"""
Example 3 — Microsoft Learn MCP (free, no API key).

Microsoft's official Learn MCP server provides tools to search and fetch
trusted, up-to-date content from Microsoft Learn (Azure, .NET, C#, etc.).
Free and **no authentication**.

    Server : https://learn.microsoft.com/api/mcp
    Auth   : none
    Tools  : microsoft_docs_search, microsoft_docs_fetch

This is a good example of grounding an agent in an authoritative external
knowledge base instead of relying on the model's training data.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    python 3_microsoft_learn_docs.py
"""

from swarms import Agent

# Any LiteLLM model works; gpt-4o-mini is cheap and good at tool use.
MODEL = "gpt-4o-mini"

agent = Agent(
    agent_name="MS-Learn-Agent",
    agent_description=(
        "Answers Microsoft/Azure/.NET questions from official Learn docs."
    ),
    model_name=MODEL,
    mcp_url="https://learn.microsoft.com/api/mcp",
    max_loops=1,
)

if __name__ == "__main__":
    result = agent.run(
        "Search Microsoft Learn and summarize how to authenticate a "
        "Python app to Azure using DefaultAzureCredential. Cite the docs."
    )
    print(result)
