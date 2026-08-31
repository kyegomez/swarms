from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

MCPAuthType = Literal[
    "none",
    "api_key",
    "bearer",
    "oauth",
    "custom",
]

MCPTransportType = Literal[
    "streamable_http",
    "sse",
    "stdio",
    "auto",
]


class MCPOAuthConfig(BaseModel):
    """
    OAuth 2.1 configuration for an MCP server.

    Three flavours are supported:

    1. ``grant_type="authorization_code"`` (default) — the interactive browser
       flow described by the MCP authorization spec. PKCE and RFC 7591 dynamic
       client registration are handled by the MCP SDK, so ``client_id`` is
       optional. Tokens are cached on disk so the browser prompt only happens
       once.
    2. ``grant_type="client_credentials"`` — a headless machine-to-machine
       flow. Requires ``client_id``/``client_secret``. The token endpoint is
       discovered from the server's ``/.well-known/oauth-authorization-server``
       metadata unless ``token_url`` is given.
    3. ``access_token=...`` — a token you already obtained elsewhere. No flow
       is run; the token is simply sent as a bearer credential.

    Any string field may use ``"env:MY_VAR"`` or ``"${MY_VAR}"`` to read the
    value from the environment instead of hardcoding a secret.
    """

    grant_type: Literal[
        "authorization_code", "client_credentials"
    ] = Field(
        default="authorization_code",
        description="OAuth grant to use when no static access_token is supplied.",
    )
    client_id: Optional[str] = Field(
        default=None,
        description="OAuth client id. Optional for authorization_code when the server supports dynamic client registration.",
    )
    client_secret: Optional[str] = Field(
        default=None,
        description="OAuth client secret. Required for client_credentials.",
    )
    scopes: Optional[List[str]] = Field(
        default=None,
        description="Scopes to request, e.g. ['mcp:tools', 'offline_access'].",
    )
    redirect_uri: str = Field(
        default="http://127.0.0.1:8765/callback",
        description="Loopback redirect URI used to capture the authorization code.",
    )
    client_name: str = Field(
        default="Swarms Agent",
        description="Client name sent during dynamic client registration.",
    )
    client_uri: Optional[str] = Field(
        default=None,
        description="Client homepage sent during dynamic client registration.",
    )
    authorization_url: Optional[str] = Field(
        default=None,
        description="Explicit authorization endpoint. Discovered automatically when omitted.",
    )
    token_url: Optional[str] = Field(
        default=None,
        description="Explicit token endpoint. Discovered automatically when omitted.",
    )
    access_token: Optional[str] = Field(
        default=None,
        description="Pre-obtained access token. When set, no OAuth flow is performed.",
    )
    refresh_token: Optional[str] = Field(
        default=None,
        description="Pre-obtained refresh token, paired with access_token.",
    )
    token_storage_path: Optional[str] = Field(
        default=None,
        description="File used to cache OAuth tokens. Defaults to ~/.swarms/mcp_auth/<server>.json.",
    )
    use_token_cache: bool = Field(
        default=True,
        description="Persist tokens to disk so the browser flow is only run once.",
    )
    open_browser: bool = Field(
        default=True,
        description="Open the system browser for the authorization step. When False the URL is logged instead.",
    )
    callback_timeout: int = Field(
        default=300,
        description="Seconds to wait for the user to complete the browser flow.",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class MCPConnection(BaseModel):
    type: Optional[str] = Field(
        default="mcp",
        description="The type of connection, defaults to 'mcp'",
    )
    url: Optional[str] = Field(
        default="http://localhost:8000/mcp",
        description="The URL endpoint for the MCP server",
    )
    name: Optional[str] = Field(
        default=None,
        description="Human readable name for the server, used in logs and tool routing",
    )
    tool_configurations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Dictionary containing configuration settings for MCP tools",
    )
    authorization_token: Optional[str] = Field(
        default=None,
        description="Bearer token for accessing the MCP server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the MCP server. Sent using api_key_header/api_key_prefix.",
    )
    api_key_header: str = Field(
        default="Authorization",
        description="Header used to send the API key, e.g. 'Authorization' or 'X-API-Key'.",
    )
    api_key_prefix: Optional[str] = Field(
        default="Bearer",
        description="Prefix prepended to the API key value. Set to None/'' for raw keys.",
    )
    auth_type: Optional[MCPAuthType] = Field(
        default=None,
        description="Explicit auth mode. Inferred from the other fields when omitted.",
    )
    oauth: Optional[MCPOAuthConfig] = Field(
        default=None,
        description="OAuth 2.1 configuration for this server.",
    )
    transport: Optional[str] = Field(
        default="streamable_http",
        description="Transport protocol: 'streamable_http', 'sse', 'stdio', or 'auto'",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="Headers to send to the MCP server"
    )
    timeout: Optional[int] = Field(
        default=30, description="Request timeout for the MCP server"
    )
    sse_read_timeout: Optional[int] = Field(
        default=300,
        description="How long to wait for streamed events before giving up",
    )
    tool_timeout: Optional[int] = Field(
        default=120,
        description="How long a single tool call may run before timing out. Separate from `timeout`, which bounds HTTP requests.",
    )
    command: Optional[str] = Field(
        default=None,
        description="Executable to launch for the 'stdio' transport",
    )
    args: Optional[List[str]] = Field(
        default=None,
        description="Arguments passed to the 'stdio' command",
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description="Environment variables for the 'stdio' command",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
