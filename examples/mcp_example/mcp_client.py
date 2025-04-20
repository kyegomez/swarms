from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stdout,
           level="INFO",
           format="{time} | {level} | {module}:{function}:{line} - {message}")

# Math prompt for testing MCP integration
MATH_PROMPT = """
You are a math calculator assistant that uses tools to perform calculations.

When asked for calculations, determine the operation and numbers, then use one of these tools:
- add: Add two numbers
- multiply: Multiply two numbers
- divide: Divide first number by second

FORMAT as JSON:
{"tool_name": "add", "a": 5, "b": 10}
"""


def main():
    """Main function to test MCP integration with Agent."""
    print("=== MINIMAL MCP AGENT INTEGRATION TEST ===")
    print("Testing only the core MCP integration with Agent")

    try:
        # Create the server parameters correctly
        logger.info("Creating MCP server parameters...")
        mcp_server = {
            "url": "http://0.0.0.0:8000",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            "timeout": 10.0,
            "sse_read_timeout": 30.0
        }

        # Log the server params to verify they're correct
        logger.info(f"MCP Server URL: {mcp_server['url']}")
        logger.info(f"MCP Headers: {mcp_server['headers']}")

        # Create agent with minimal configuration
        logger.info("Creating Agent with MCP integration...")
        agent = Agent(
            agent_name="MCP Test Agent",
            system_prompt=MATH_PROMPT,
            mcp_servers=[mcp_server],  # Pass server config as a list of dicts
            verbose=True)

        print("\nAgent created successfully!")
        print("Enter a math query or 'exit' to quit")

        # Simple interaction loop
        while True:
            query = input("\nMath query: ").strip()
            if query.lower() == 'exit':
                break

            # Run the agent, which should use the MCP server
            logger.info(f"Processing query: {query}")
            result = agent.run(query)

            # Display result
            print(f"\nResult: {result}")

    except Exception as e:
        logger.error(f"Error during MCP integration test: {e}", exc_info=True)
        print(f"\nERROR: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
