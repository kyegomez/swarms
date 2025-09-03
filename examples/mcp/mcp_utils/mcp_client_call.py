from swarms.tools.mcp_client_tools import (
    get_mcp_tools_sync,
    execute_tool_call_simple,
)

tools = get_mcp_tools_sync()

print(tools)

result = execute_tool_call_simple(tools[0], "Hello, world!")

print(result)
