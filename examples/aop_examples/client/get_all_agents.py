import json
from swarms.tools.mcp_client_tools import get_mcp_tools_sync


print(
    json.dumps(
        get_mcp_tools_sync(server_path="http://0.0.0.0:8000/mcp"),
        indent=4,
    )
)
