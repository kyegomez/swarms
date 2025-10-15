import json
from swarms.tools.mcp_client_tools import (
    get_mcp_tools_sync,
    execute_tool_call_simple,
)


def get_available_tools(
    server_path: str = "http://localhost:8000/mcp",
) -> list:
    """
    Get all available MCP tools from the server.

    Args:
        server_path: URL of the MCP server

    Returns:
        List of available tools
    """
    tools = get_mcp_tools_sync(server_path=server_path)
    return tools


def call_agent_tool(
    tool_name: str,
    task: str,
    server_path: str = "http://localhost:8000/mcp",
) -> dict:
    """
    Call a specific agent tool with a task.

    Args:
        tool_name: Name of the agent tool to call
        task: Task or prompt to send to the agent
        server_path: URL of the MCP server

    Returns:
        Response from the agent tool
    """
    call = {
        "function": {
            "name": tool_name,
            "arguments": {"task": task},
        }
    }

    try:
        import asyncio

        result = asyncio.run(
            execute_tool_call_simple(
                response=call, server_path=server_path
            )
        )
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    """
    Main function to demonstrate MCP tool usage.
    """
    server_path = "http://localhost:8000/mcp"

    # Step 1: Get available tools
    tools = get_available_tools(server_path)

    if not tools:
        return {
            "error": "No tools available. Make sure the MCP server is running."
        }

    # Step 2: Find an agent tool to call
    agent_tools = [
        tool
        for tool in tools
        if "agent" in tool.get("function", {}).get("name", "").lower()
    ]

    if not agent_tools:
        return {
            "error": "No agent tools found",
            "available_tools": [
                tool.get("function", {}).get("name", "Unknown")
                for tool in tools
            ],
        }

    # Step 3: Call the first available agent tool
    agent_tool = agent_tools[0]
    tool_name = agent_tool.get("function", {}).get("name")

    # Example task
    task = "Hello! Can you help me understand what you do and what capabilities you have?"

    # Call the agent
    result = call_agent_tool(tool_name, task, server_path)

    return result


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=4))
