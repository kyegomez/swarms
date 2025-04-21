from swarms import Agent
from loguru import logger
import sys
from swarms.prompts.agent_prompts import MATH_PROMPT
# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")

# Math prompt for testing MCP integration


def main():
    """Test MCP integration with Agent."""
    print("=== MINIMAL MCP AGENT INTEGRATION TEST ===")

    try:
        # Create the MCP server parameters as a dictionary
        mcp_server = {
            "url": "http://0.0.0.0:8000",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            "timeout": 10.0,
            "sse_read_timeout": 30.0
        }

        # Create agent with minimal configuration
        agent = Agent(
            agent_name="MCP Test Agent",
            system_prompt=MATH_PROMPT,
            mcp_servers=[mcp_server],  # Pass as a list of dictionaries
            model_name="gpt-4o-mini",
            verbose=False  # Reduce verbosity to focus on errors
        )

        print("\nAgent created successfully!")
        print("Enter a math query or 'exit' to quit")

        # Simple interaction loop
        while True:
            query = input("\nMath query: ").strip()
            if query.lower() == 'exit':
                break

            # Run the agent
            print(f"\nProcessing: {query}")
            result = agent.run(query)

            # Display result
            print(f"\nResult: {result}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
