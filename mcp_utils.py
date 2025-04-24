from swarms.tools.mcp_client import (
    list_tools_for_multiple_urls,
)


print(
    list_tools_for_multiple_urls(
        ["http://0.0.0.0:8000/sse"], output_type="json"
    )
)
