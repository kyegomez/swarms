from swarms.tools.mcp_client_tools import (
    get_tools_for_multiple_mcp_servers,
)


print(
    get_tools_for_multiple_mcp_servers(
        urls=["http://0.0.0.0:5932/mcp"]
    )
)
