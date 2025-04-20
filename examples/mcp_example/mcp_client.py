from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams
from swarms.prompts.agent_prompts import MATH_AGENT_PROMPT

# Fallback in case the import fails
if 'MATH_AGENT_PROMPT' not in locals():
    MATH_AGENT_PROMPT = """You are a specialized math agent that can perform calculations by calling external math service APIs.
Key responsibilities:
1. Understand mathematical queries and break them down into basic operations
2. Use available math tools (add, multiply, divide) appropriately
3. Provide clear explanations of calculations
4. Handle errors gracefully if operations fail

Remember to use the available MCP tools for calculations rather than doing them directly."""

def main():
    # Configure MCP server connection
    math_server = MCPServerSseParams(
        url="http://0.0.0.0:8000/mcp",
        headers={"Content-Type": "application/json"},
        timeout=5.0,
        sse_read_timeout=30.0
    )

    # Initialize math agent
    math_agent = Agent(
        agent_name="Math Agent",
        agent_description="Specialized agent for mathematical computations",
        system_prompt=MATH_AGENT_PROMPT,
        max_loops=auto,
        mcp_servers=[math_server],
        streaming_on=True
    )

    print("\nMath Agent System Initialized")
    print("\nAvailable operations:")
    print("Math Agent: add, multiply, divide")

    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")

        if query.lower() == 'exit':
            break

        # Process with math agent
        math_result = math_agent.run(query)
        print("\nMath Agent Response:", math_result)

if __name__ == "__main__":
    main()