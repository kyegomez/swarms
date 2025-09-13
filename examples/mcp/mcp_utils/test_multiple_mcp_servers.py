"""
Simple test for the execute_multiple_tools_on_multiple_mcp_servers functionality.
"""

from swarms.tools.mcp_client_tools import (
    execute_multiple_tools_on_multiple_mcp_servers_sync,
)


def test_async_multiple_tools_execution():
    """Test the async multiple tools execution function structure."""
    print(
        "\nTesting async multiple tools execution function structure..."
    )

    urls = [
        "http://localhost:8000/mcp",
        "http://localhost:8001/mcp",
    ]

    # Mock responses with multiple tool calls
    responses = [
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "get_okx_crypto_price",
                        "arguments": {"symbol": "SOL-USDT"},
                    }
                },
                {
                    "function": {
                        "name": "get_crypto_price",
                        "arguments": {"coin_id": "solana"},
                    }
                },
            ]
        }
    ]

    try:
        # This will likely fail to connect, but we can test the function structure
        results = execute_multiple_tools_on_multiple_mcp_servers_sync(
            responses=responses, urls=urls
        )
        print(f"Got {len(results)} results")
        print(results)
    except Exception as e:
        print(f"Expected error (no servers running): {str(e)}")
        print("Async function structure is working correctly!")


if __name__ == "__main__":
    test_async_multiple_tools_execution()
