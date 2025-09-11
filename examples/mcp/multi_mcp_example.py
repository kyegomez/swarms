#!/usr/bin/env python3
"""
Multi-MCP Agent Example

This example demonstrates how to use multiple MCP (Model Context Protocol) servers
with a single Swarms agent. The agent can access tools from different MCP servers
simultaneously, enabling powerful cross-server functionality.

Prerequisites:
1. Start the OKX crypto server: python multi_mcp_guide/okx_crypto_server.py
2. Start the agent tools server: python multi_mcp_guide/mcp_agent_tool.py
3. Install required dependencies: pip install swarms mcp fastmcp requests

Usage:
    python examples/multi_agent/multi_mcp_example.py
"""

from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)


def create_multi_mcp_agent():
    """
    Create an agent that can access multiple MCP servers.

    Returns:
        Agent: Configured agent with access to multiple MCP servers
    """
    return Agent(
        agent_name="Multi-MCP-Financial-Agent",
        agent_description="Advanced financial analysis agent with multi-MCP capabilities",
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        max_loops=3,
        mcp_urls=[
            "http://0.0.0.0:8001/mcp",  # OKX Crypto Server
            "http://0.0.0.0:8000/mcp",  # Agent Tools Server
        ],
        model_name="gpt-4o-mini",
        output_type="all",
    )


def basic_crypto_analysis():
    """
    Basic example: Get cryptocurrency prices using OKX MCP server.
    """
    print("=== Basic Crypto Analysis ===")

    agent = create_multi_mcp_agent()

    # Simple crypto price lookup
    result = agent.run(
        "Get the current price of Bitcoin, Ethereum, and Solana using the OKX crypto tools"
    )

    print("Result:", result)
    print("\n" + "=" * 50 + "\n")


def advanced_multi_step_analysis():
    """
    Advanced example: Combine crypto data with agent creation.
    """
    print("=== Advanced Multi-Step Analysis ===")

    agent = create_multi_mcp_agent()

    # Complex multi-step task
    result = agent.run(
        """
    Perform a comprehensive crypto market analysis:

    1. Get current prices for Bitcoin, Ethereum, and Solana
    2. Get 24h trading volumes for these cryptocurrencies
    3. Create a technical analysis agent to analyze the price trends
    4. Create a market sentiment agent to provide additional insights
    5. Summarize all findings in a comprehensive report
    """
    )

    print("Result:", result)
    print("\n" + "=" * 50 + "\n")


def custom_system_prompt_example():
    """
    Example with custom system prompt for better tool coordination.
    """
    print("=== Custom System Prompt Example ===")

    custom_prompt = """
    You are a sophisticated financial analysis agent with access to multiple specialized tools:

    CRYPTO TOOLS (from OKX server):
    - get_okx_crypto_price: Get current cryptocurrency prices
    - get_okx_crypto_volume: Get 24h trading volumes

    AGENT CREATION TOOLS (from Agent Tools server):
    - create_agent: Create specialized agents for specific analysis tasks

    INSTRUCTIONS:
    1. Always use the most appropriate tool for each task
    2. When creating agents, give them specific, focused tasks
    3. Provide clear, actionable insights based on the data
    4. If a tool fails, try alternative approaches
    5. Always explain your reasoning and methodology

    Your goal is to provide comprehensive, data-driven financial analysis.
    """

    agent = Agent(
        agent_name="Custom-Multi-MCP-Agent",
        system_prompt=custom_prompt,
        mcp_urls=[
            "http://0.0.0.0:8001/mcp",  # OKX Crypto
            "http://0.0.0.0:8000/mcp",  # Agent Tools
        ],
        model_name="gpt-4o-mini",
        max_loops=2,
    )

    result = agent.run(
        "Analyze the current crypto market and create specialized agents to help with investment decisions"
    )

    print("Result:", result)
    print("\n" + "=" * 50 + "\n")


def error_handling_example():
    """
    Example demonstrating error handling with multiple MCP servers.
    """
    print("=== Error Handling Example ===")

    agent = Agent(
        agent_name="Robust-Multi-MCP-Agent",
        system_prompt="You are a resilient agent that handles errors gracefully.",
        mcp_urls=[
            "http://0.0.0.0:8001/mcp",  # OKX Crypto (should be running)
            "http://0.0.0.0:8002/mcp",  # Non-existent server (will fail)
            "http://0.0.0.0:8000/mcp",  # Agent Tools (should be running)
        ],
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    try:
        result = agent.run(
            "Try to use all available tools. If some fail, work with what's available."
        )
        print("Result:", result)
    except Exception as e:
        print(f"Error occurred: {e}")

    print("\n" + "=" * 50 + "\n")


def main():
    """
    Main function to run all examples.
    """
    print("Multi-MCP Agent Examples")
    print("=" * 50)
    print("Make sure the MCP servers are running:")
    print(
        "1. OKX Crypto Server: python multi_mcp_guide/okx_crypto_server.py"
    )
    print(
        "2. Agent Tools Server: python multi_mcp_guide/mcp_agent_tool.py"
    )
    print("=" * 50)

    try:
        # Run examples
        basic_crypto_analysis()
        advanced_multi_step_analysis()
        custom_system_prompt_example()
        error_handling_example()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the MCP servers are running and accessible.")


if __name__ == "__main__":
    main()
