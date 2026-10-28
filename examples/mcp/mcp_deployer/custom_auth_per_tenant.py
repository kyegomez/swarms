import os

from swarms import Agent, MCPDeployer

TENANT_KEYS = {
    "acme": os.getenv("ACME_MCP_KEY", "acme-secret"),
    "globex": os.getenv("GLOBEX_MCP_KEY", "globex-secret"),
}


async def tenant_auth(credential, headers):
    """Admit a tenant only when its key matches; attach the tenant as claims."""
    tenant = headers.get("x-tenant")
    if tenant not in TENANT_KEYS:
        return False
    if credential != TENANT_KEYS[tenant]:
        return False
    return {"subject": tenant, "scopes": ["run"], "tenant": tenant}


analyst = Agent(
    agent_name="Support-Analyst",
    agent_description="Classifies a support ticket and drafts a first reply.",
    system_prompt="Classify the ticket (billing, bug, how-to) and draft a two-sentence reply.",
    model_name="openrouter/moonshotai/kimi-k3",
    max_loops=1,
    print_on=False,
)

deployer = MCPDeployer(
    analyst,
    auth=tenant_auth,
    port=8002,
    verbose=True,
)

if __name__ == "__main__":
    deployer.run()
