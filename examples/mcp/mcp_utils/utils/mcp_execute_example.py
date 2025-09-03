from swarms.schemas.mcp_schemas import MCPConnection
from swarms.tools.mcp_client_tools import (
    execute_tool_call_simple,
)
import asyncio

# Example 1: Create a new markdown file
response = {
    "function": {
        "name": "get_crypto_price",
        "arguments": {"coin_id": "bitcoin"},
    }
}

connection = MCPConnection(
    url="http://0.0.0.0:8000/sse",
    headers={"Authorization": "Bearer 1234567890"},
    timeout=10,
)

url = "http://0.0.0.0:8000/sse"

if __name__ == "__main__":
    tools = asyncio.run(
        execute_tool_call_simple(
            response=response,
            connection=connection,
            output_type="json",
            # server_path=url,
        )
    )

    print(tools)
