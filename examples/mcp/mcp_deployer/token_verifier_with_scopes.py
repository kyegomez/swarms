import time

from mcp.server.auth.provider import AccessToken

from swarms import Agent, MCPDeployer

ISSUED_TOKENS = {
    "tok-admin": AccessToken(
        token="tok-admin",
        client_id="ops-team",
        scopes=["agent:run", "agent:admin"],
        expires_at=int(time.time()) + 3600,
    ),
    "tok-reader": AccessToken(
        token="tok-reader",
        client_id="dashboard",
        scopes=["agent:read"],  # missing agent:run, so it is refused
    ),
}


class StaticTokenVerifier:
    async def verify_token(self, token: str):
        return ISSUED_TOKENS.get(token)


summariser = Agent(
    agent_name="Summariser",
    agent_description="Summarises any text into three bullet points.",
    system_prompt="Summarise the input into exactly three bullet points.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

deployer = MCPDeployer(
    summariser,
    token_verifier=StaticTokenVerifier(),
    required_scopes=["agent:run"],
    port=8003,
)

if __name__ == "__main__":
    # `Authorization: Bearer tok-admin` is admitted; tok-reader gets a 401.
    deployer.run()
