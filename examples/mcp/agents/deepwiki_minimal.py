from swarms import Agent

agent = Agent(
    agent_name="MCP-Agent",
    model_name="claude-sonnet-5",
    mcp_url="https://mcp.deepwiki.com/mcp",
    max_loops=1,
    temperature=None,
    max_tokens=16_000,
    reasoning_effort=None,
)

print(
    agent.run(
        "Use your tools to explain what the kyegomez/swarms repository does."
    )
)


# ----------------------------------------------------------------------
# Authenticated MCP servers
# ----------------------------------------------------------------------
#
# API key / bearer token — the token is sent as "Authorization: Bearer <key>".
# Use "env:VAR_NAME" to keep the secret out of your source.
#
#   agent = Agent(
#       agent_name="MCP-Agent",
#       model_name="gpt-5.4",
#       mcp_url="https://api.githubcopilot.com/mcp/",
#       mcp_api_key="env:GITHUB_TOKEN",
#   )
#
# A server that wants its key in a custom header instead:
#
#   from swarms.schemas.mcp_schemas import MCPConnection
#
#   agent = Agent(
#       agent_name="MCP-Agent",
#       model_name="gpt-5.4",
#       mcp_config=MCPConnection(
#           url="https://api.example.com/mcp",
#           api_key="env:EXAMPLE_API_KEY",
#           api_key_header="X-API-Key",
#           api_key_prefix=None,      # send the raw key, no "Bearer " prefix
#       ),
#   )
#
# OAuth — opens your browser once, then caches the tokens in
# ~/.swarms/mcp_auth/ so later runs are silent:
#
#   from swarms.schemas.mcp_schemas import MCPOAuthConfig
#
#   agent = Agent(
#       agent_name="MCP-Agent",
#       model_name="gpt-5.4",
#       mcp_url="https://api.example.com/mcp",
#       mcp_oauth=MCPOAuthConfig(scopes=["mcp:tools", "offline_access"]),
#   )
#
# Headless OAuth (no browser), for servers issuing machine tokens:
#
#   mcp_oauth=MCPOAuthConfig(
#       grant_type="client_credentials",
#       client_id="env:MCP_CLIENT_ID",
#       client_secret="env:MCP_CLIENT_SECRET",
#   )
#
# Several servers at once — tools are merged and each call is routed back to
# the server that owns it:
#
#   agent = Agent(
#       agent_name="MCP-Agent",
#       model_name="gpt-5.4",
#       mcp_urls=[
#           "https://mcp.deepwiki.com/mcp",
#           "https://api.example.com/mcp",
#       ],
#   )
