"""
Multi-Tool Agent Tutorial with X402 Tools

This comprehensive tutorial demonstrates how to build agents that utilize
multiple X402 tools simultaneously. This example covers tool selection,
coordination, and error handling patterns for complex multi-tool workflows.

Key concepts:
- Tool selection based on task requirements
- Parallel tool execution for efficiency
- Error handling and fallback mechanisms
- Coordination of multiple data sources
- Result synthesis from diverse tool outputs
"""

from swarms import Agent

# Note: This example assumes swarms_tools package is installed
# Install with: pip install swarms-tools
# For this demo, we'll show the pattern even if tools aren't available

try:
    from swarms_tools import (
        exa_search,
        yahoo_finance_api,
        coin_gecko_coin_api,
    )

    TOOLS_AVAILABLE = True
except ImportError:
    print(
        "swarms_tools package not available - showing conceptual example"
    )
    TOOLS_AVAILABLE = False

    # Mock tools for demonstration
    def exa_search(query: str) -> str:
        return f"Mock search results for: {query}"

    def yahoo_finance_api(symbol: str) -> str:
        return f"Mock financial data for: {symbol}"

    def coin_gecko_coin_api(coin_id: str) -> str:
        return f"Mock crypto data for: {coin_id}"


# Create a comprehensive multi-tool agent
multi_tool_agent = Agent(
    agent_name="ComprehensiveAnalysisAgent",
    system_prompt="""You are a comprehensive analysis agent with access to multiple research and data tools.
    Use the available tools strategically to gather information from diverse sources.
    Coordinate between web search, financial data, and cryptocurrency information to provide
    well-rounded analysis. Always cite your sources and explain your reasoning.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    tools=(
        [
            exa_search,  # Web search and research
            yahoo_finance_api,  # Financial market data
            coin_gecko_coin_api,  # Cryptocurrency information
        ]
        if TOOLS_AVAILABLE
        else []
    ),
)

# Complex analysis task requiring multiple tools
comprehensive_task = """
Conduct a comprehensive analysis of Tesla (TSLA) stock and its potential impact on the EV market.
Include:
1. Current stock performance and analyst ratings
2. Recent news and market sentiment
3. Cryptocurrency correlation (Dogecoin, since Elon Musk influence)
4. Competitive landscape and market share
5. Future outlook and investment recommendation

Use all available tools to gather diverse perspectives and provide a balanced analysis.
"""

response = multi_tool_agent.run(comprehensive_task)
print(response)
