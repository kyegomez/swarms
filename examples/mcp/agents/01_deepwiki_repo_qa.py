"""
Example 1 — DeepWiki MCP (free, no API key).

DeepWiki (by Cognition) exposes an MCP server that can read and answer
questions about the documentation of any public GitHub repository. It is
completely free and requires **no authentication**, so this example runs
out of the box.

    Server : https://mcp.deepwiki.com/mcp
    Auth   : none
    Tools  : read_wiki_structure, read_wiki_contents, ask_question

The Agent auto-detects the transport from the URL scheme (an ``https://``
URL uses streamable-HTTP), fetches the server's tools on startup, and the
model calls them as needed to answer the task.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    python 1_deepwiki_repo_qa.py
"""

from swarms import Agent

# Any LiteLLM model works; gpt-4o-mini is cheap and good at tool use.
MODEL = "gpt-5.4"

DEEPWIKI_SYSTEM_PROMPT = (
    "You are a repository research specialist who uses the DeepWiki MCP "
    "server to answer questions about public GitHub repositories. Inspect the "
    "repository's wiki structure and relevant documentation before responding, "
    "then provide a clear, technically accurate explanation grounded only in "
    "the retrieved material. Cite relevant files, modules, or documentation "
    "sections when available, distinguish verified details from reasonable "
    "inferences, and state clearly when DeepWiki does not provide enough "
    "information to answer a question."
)

agent = Agent(
    agent_name="DeepWiki-Agent",
    agent_description="Answers questions about GitHub repos via DeepWiki MCP.",
    system_prompt=DEEPWIKI_SYSTEM_PROMPT,
    model_name=MODEL,
    mcp_url="https://mcp.deepwiki.com/mcp",
    max_loops=1,
    reasoning_effort=None,
)

if __name__ == "__main__":
    result = agent.run(
        "Use your DeepWiki tools to explain what the kyegomez/swarms "
        "repository is for and list its main multi-agent structures."
    )
    print(result)
