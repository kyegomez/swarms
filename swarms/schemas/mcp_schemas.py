from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class MCPConnection(BaseModel):
    type: Optional[str] = Field(
        default="mcp",
        description="The type of connection, defaults to 'mcp'",
    )
    url: Optional[str] = Field(
        default="localhost:8000/sse",
        description="The URL endpoint for the MCP server",
    )
    tool_configurations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Dictionary containing configuration settings for MCP tools",
    )
    authorization_token: Optional[str] = Field(
        default=None,
        description="Authentication token for accessing the MCP server",
    )
    transport: Optional[str] = Field(
        default="sse",
        description="The transport protocol to use for the MCP server",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="Headers to send to the MCP server"
    )
    timeout: Optional[int] = Field(
        default=5, description="Timeout for the MCP server"
    )

    class Config:
        arbitrary_types_allowed = True


class MultipleMCPConnections(BaseModel):
    connections: List[MCPConnection] = Field(
        default=[], description="List of MCP connections"
    )

    class Config:
        arbitrary_types_allowed = True
