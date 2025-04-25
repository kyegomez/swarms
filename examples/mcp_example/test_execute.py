from swarms.tools.mcp_client import execute_mcp_tool

print(
    execute_mcp_tool(
        "http://0.0.0.0:8000/sse",
        parameters={"name": "add", "a": 1, "b": 2},
    )
)
