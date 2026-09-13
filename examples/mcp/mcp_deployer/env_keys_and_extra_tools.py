import datetime

from swarms import Agent, MCPDeployer


def utc_now() -> str:
    """The current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


editor = Agent(
    agent_name="Editor",
    agent_description="Tightens prose without changing its meaning.",
    system_prompt="Rewrite the text to be shorter and clearer. Keep the meaning.",
    model_name="claude-haiku-4-5-20251001",
    max_loops=1,
    print_on=False,
)

deployer = MCPDeployer(
    editor,
    api_key_env="MCP_DEPLOYER_KEYS",  # e.g. MCP_DEPLOYER_KEYS="key-a,key-b"
    extra_tools=[utc_now, word_count],
    port=8004,
)

if __name__ == "__main__":
    # Clients see three tools: editor, utc_now and word_count.
    deployer.run()
