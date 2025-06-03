from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Callable
from swarms.schemas.mcp_schemas import MCPConnection


class AgentToolTypes(BaseModel):
    tool_schema: List[Dict[str, Any]]
    mcp_connection: MCPConnection
    tool_model: Optional[BaseModel]
    tool_functions: Optional[List[Callable]]

    class Config:
        arbitrary_types_allowed = True
