#!/usr/bin/env python3

import json
import time
from swarms.tools.base_tool import BaseTool


# Define some test functions
def get_coin_price(coin_id: str, vs_currency: str = "usd") -> str:
    """Get the current price of a specific cryptocurrency."""
    # Simulate API call with some delay
    time.sleep(1)

    # Mock data for testing
    mock_data = {
        "bitcoin": {"usd": 45000, "usd_market_cap": 850000000000},
        "ethereum": {"usd": 2800, "usd_market_cap": 340000000000},
    }

    result = mock_data.get(
        coin_id, {coin_id: {"usd": 1000, "usd_market_cap": 1000000}}
    )
    return json.dumps(result)


def get_top_cryptocurrencies(
    limit: int = 10, vs_currency: str = "usd"
) -> str:
    """Fetch the top cryptocurrencies by market capitalization."""
    # Simulate API call with some delay
    time.sleep(1)

    # Mock data for testing
    mock_data = [
        {"id": "bitcoin", "name": "Bitcoin", "current_price": 45000},
        {"id": "ethereum", "name": "Ethereum", "current_price": 2800},
        {"id": "cardano", "name": "Cardano", "current_price": 0.5},
        {"id": "solana", "name": "Solana", "current_price": 150},
        {"id": "polkadot", "name": "Polkadot", "current_price": 25},
    ]

    return json.dumps(mock_data[:limit])


# Mock tool call objects (simulating OpenAI ChatCompletionMessageToolCall)
class MockToolCall:
    def __init__(self, name, arguments, call_id):
        self.type = "function"
        self.id = call_id
        self.function = MockFunction(name, arguments)


class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = (
            arguments
            if isinstance(arguments, str)
            else json.dumps(arguments)
        )


def test_function_calls():
    # Create BaseTool instance
    tool = BaseTool(
        tools=[get_coin_price, get_top_cryptocurrencies], verbose=True
    )

    # Create mock tool calls (similar to what OpenAI returns)
    tool_calls = [
        MockToolCall(
            "get_coin_price",
            {"coin_id": "bitcoin", "vs_currency": "usd"},
            "call_1",
        ),
        MockToolCall(
            "get_top_cryptocurrencies",
            {"limit": 5, "vs_currency": "usd"},
            "call_2",
        ),
    ]

    print("Testing list of tool call objects...")
    print(
        f"Tool calls: {[(call.function.name, call.function.arguments) for call in tool_calls]}"
    )

    # Test sequential execution
    print("\n=== Sequential Execution ===")
    start_time = time.time()
    results_sequential = (
        tool.execute_function_calls_from_api_response(
            tool_calls, sequential=True, return_as_string=True
        )
    )
    sequential_time = time.time() - start_time

    print(f"Sequential execution took: {sequential_time:.2f} seconds")
    for result in results_sequential:
        print(f"Result: {result[:100]}...")

    # Test parallel execution
    print("\n=== Parallel Execution ===")
    start_time = time.time()
    results_parallel = tool.execute_function_calls_from_api_response(
        tool_calls,
        sequential=False,
        max_workers=2,
        return_as_string=True,
    )
    parallel_time = time.time() - start_time

    print(f"Parallel execution took: {parallel_time:.2f} seconds")
    for result in results_parallel:
        print(f"Result: {result[:100]}...")

    print(f"\nSpeedup: {sequential_time/parallel_time:.2f}x")

    # Test with raw results (not as strings)
    print("\n=== Raw Results ===")
    raw_results = tool.execute_function_calls_from_api_response(
        tool_calls, sequential=False, return_as_string=False
    )

    for i, result in enumerate(raw_results):
        print(
            f"Raw result {i+1}: {type(result)} - {str(result)[:100]}..."
        )


if __name__ == "__main__":
    test_function_calls()
