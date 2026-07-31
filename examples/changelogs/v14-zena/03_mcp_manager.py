"""Zena: unified MCP manager with OAuth.

The old mcp_client_tools module is gone. Connections, tool discovery, and
auth now live behind MCPManager, reachable through Agent parameters.

The MCP server serves streamable HTTP rather than stdio, so it works over a
network instead of only as a subprocess.
"""

from swarms import Agent

# --- A single server -------------------------------------------------
agent = Agent(
    agent_name="MCP-Agent",
    model_name="gpt-5.4",
    mcp_url="http://localhost:8000/mcp",
    max_loops=1,
)

# --- Several servers, with auth, headers, and a timeout ---------------
multi = Agent(
    agent_name="Multi-MCP-Agent",
    model_name="gpt-5.4",
    mcp_urls=[
        "http://localhost:8000/mcp",
        "https://tools.internal.example.com/mcp",
    ],
    mcp_authorization_token="Bearer ...",
    mcp_headers={"X-Tenant-Id": "acme"},
    mcp_timeout=30,
    max_loops="auto",
)

# --- OAuth, with the token cache written atomically at 0600 -----------
# from swarms.tools.mcp_manager import MCPOAuthConfig, MCPFileTokenStorage
#
# oauth_agent = Agent(
#     agent_name="OAuth-MCP-Agent",
#     model_name="gpt-5.4",
#     mcp_url="https://api.partner.example.com/mcp",
#     mcp_oauth=MCPOAuthConfig(
#         client_id="...",
#         client_secret="...",
#         storage=MCPFileTokenStorage("~/.swarms/mcp_tokens.json"),
#     ),
# )

if __name__ == "__main__":
    print(
        multi.run("Use the available tools to reconcile the ledger.")
    )
