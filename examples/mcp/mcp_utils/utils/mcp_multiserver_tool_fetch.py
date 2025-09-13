from swarms.tools.mcp_client_tools import (
    get_tools_for_multiple_mcp_servers,
)
from swarms.schemas.mcp_schemas import MCPConnection


mcp_config = MCPConnection(
    url="http://0.0.0.0:8000/mcp",
    # headers={"Authorization": "Bearer 1234567890"},
    timeout=5,
)

urls = ["http://0.0.0.0:8001/mcp", "http://0.0.0.0:8001/mcp"]

out = get_tools_for_multiple_mcp_servers(
    urls=urls,
    # connections=[mcp_config],
)

print(out)
