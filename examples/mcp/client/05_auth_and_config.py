"""
Example 5 — Authentication and connection configuration.

A bare URL is the shorthand. For anything else — API keys, bearer tokens, custom
headers, a forced transport, longer timeouts, OAuth — pass an ``MCPConnection``
(or a plain dict, or the equivalent keyword arguments). ``MCPManager`` accepts
all of these interchangeably.

Nothing here connects to a server; it only shows how connections are described.

    python examples/mcp/client/05_auth_and_config.py
"""

import json

from swarms.schemas.mcp_schemas import MCPConnection, MCPOAuthConfig
from swarms.tools.mcp_manager import MCPManager


def api_key_via_kwargs() -> None:
    """Shortest path for a key-protected server."""
    manager = MCPManager(
        mcp_url="https://api.example.com/mcp",
        api_key="sk-example-123",
        timeout=15,
    )
    show("api key via kwargs", manager)


def bearer_token() -> None:
    """A pre-issued bearer token."""
    manager = MCPManager(
        mcp_url="https://api.example.com/mcp",
        authorization_token="ey-example-token",
    )
    show("bearer token", manager)


def connection_object() -> None:
    """Full control over one server via MCPConnection."""
    connection = MCPConnection(
        url="https://api.example.com/mcp",
        headers={"X-Tenant": "acme"},
        transport="streamable_http",
        timeout=20,
        tool_timeout=60,
    )
    show("MCPConnection", MCPManager(mcp_config=connection))


def secrets_from_env() -> None:
    """Keys can be indirected through the environment instead of hard-coded.

    ``env:NAME`` and ``${NAME}`` are both resolved at connection time.
    """
    connection = MCPConnection(
        url="https://api.example.com/mcp",
        api_key="env:EXAMPLE_MCP_KEY",
    )
    show("secret from env", MCPManager(mcp_config=connection))


def oauth() -> None:
    """Headless OAuth 2.1 via client credentials."""
    connection = MCPConnection(
        url="https://api.example.com/mcp",
        oauth=MCPOAuthConfig(
            grant_type="client_credentials",
            client_id="example-client",
            client_secret="env:EXAMPLE_CLIENT_SECRET",
            token_url="https://api.example.com/oauth/token",
            scopes=["tools.read"],
        ),
    )
    show(
        "oauth client credentials", MCPManager(mcp_config=connection)
    )


def mixed_servers() -> None:
    """Different auth per server, all on one manager."""
    manager = MCPManager(
        mcp_urls=[
            "http://localhost:8000/mcp",  # local, no auth
            MCPConnection(
                url="https://api.example.com/mcp",
                api_key="sk-example-123",
            ),
        ]
    )
    show("mixed auth", manager)


def show(label: str, manager: MCPManager) -> None:
    """Print the manager's config — secrets are redacted by to_dict()."""
    print(f"\n── {label}")
    print(json.dumps(manager.to_dict(), indent=2))


if __name__ == "__main__":
    api_key_via_kwargs()
    bearer_token()
    connection_object()
    secrets_from_env()
    oauth()
    mixed_servers()
