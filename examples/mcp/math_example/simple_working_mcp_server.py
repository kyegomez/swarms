#!/usr/bin/env python3
"""
Simple working MCP server for testing.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent


def mock_list_tools() -> List[Dict[str, Any]]:
    """Mock function to list available tools."""
    return [
        {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    ]


def mock_call_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Mock function to call a tool."""
    if name == "get_weather":
        location = arguments.get("location", "Unknown")
        return f"Weather in {location}: Sunny, 72°F, Humidity: 45%"
    elif name == "calculate":
        expression = arguments.get("expression", "0")
        try:
            result = eval(expression)
            return f"Result of {expression} = {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    else:
        return f"Unknown tool: {name}"


async def main():
    """Main function to run the MCP server."""
    server = Server("simple-working-mcp-server")
    
    @server.list_tools()
    async def list_tools() -> List[Dict[str, Any]]:
        """List available tools."""
        return mock_list_tools()
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call a tool with the given name and arguments."""
        try:
            result = mock_call_tool(name, arguments)
            return CallToolResult(
                content=[TextContent(type="text", text=result)]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")]
            )
    
    # Run the server with proper stdio handling
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options={}
            )
    except KeyboardInterrupt:
        print("Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1) 
