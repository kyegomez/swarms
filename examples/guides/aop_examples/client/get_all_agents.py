import json
from swarms.tools.mcp_manager import MCPManager


print(
    json.dumps(
        MCPManager(mcp_url="http://0.0.0.0:8000/mcp").get_tools(),
        indent=4,
    )
)
