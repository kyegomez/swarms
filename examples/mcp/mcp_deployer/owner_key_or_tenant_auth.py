import os

from swarms import Agent, MCPDeployer

researcher = Agent(
    agent_name="Researcher",
    agent_description="Answers research questions with a short, sourced summary.",
    system_prompt="You are a careful researcher. Answer in a short paragraph.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)


def is_allowed(credential, headers):
    """Custom auth: accept the key from MCP_DEPLOYER_KEY, or any tenant on the allow list."""
    if credential and credential == os.getenv("MCP_DEPLOYER_KEY"):
        return {"subject": "owner", "scopes": ["run"]}
    tenant = headers.get("x-tenant")
    if tenant in {"acme", "globex"}:
        return {"subject": tenant, "scopes": ["run"]}
    return False


deployer = MCPDeployer(
    researcher,
    port=8000,
    auth=is_allowed,
    timeout=120,
    verbose=True,
)

if __name__ == "__main__":
    deployer.run()
