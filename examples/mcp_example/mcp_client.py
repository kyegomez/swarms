from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams, MCPServerSse, mcp_flow_get_tool_schema
from loguru import logger
import sys
import asyncio
import json
import httpx
import time

# Configure logging for more detailed output
logger.remove()
logger.add(sys.stdout,
           level="DEBUG",
           format="{time} | {level} | {module}:{function}:{line} - {message}")

# Relaxed prompt that doesn't enforce strict JSON formatting



# Create server parameters
def get_server_params():
    """Get the MCP server connection parameters."""
    return MCPServerSseParams(
        url=
        "http://127.0.0.1:8000",  # Use 127.0.0.1 instead of localhost/0.0.0.0
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        },
        timeout=15.0,  # Longer timeout
        sse_read_timeout=60.0  # Longer read timeout
    )


def initialize_math_system():
    """Initialize the math agent with MCP server configuration."""
    # Create the agent with the MCP server configuration
    math_agent = Agent(agent_name="Math Assistant",
                       agent_description="Friendly math calculator",
                       system_prompt=MATH_AGENT_PROMPT,
                       max_loops=1,
                       mcp_servers=[get_server_params()],
                       model_name="gpt-3.5-turbo",
                       verbose=True)

    return math_agent


# Function to get list of available tools from the server
async def get_tools_list():
    """Fetch and format the list of available tools from the server."""
    try:
        server_params = get_server_params()
        tools = await mcp_flow_get_tool_schema(server_params)

        if not tools:
            return "No tools are currently available on the server."

        # Format the tools information
        tools_info = "Available tools:\n"
        for tool in tools:
            tools_info += f"\n- {tool.name}: {tool.description or 'No description'}\n"
            if tool.parameters and hasattr(tool.parameters, 'properties'):
                tools_info += "  Parameters:\n"
                for param_name, param_info in tool.parameters.properties.items(
                ):
                    param_type = param_info.get('type', 'unknown')
                    param_desc = param_info.get('description',
                                                'No description')
                    tools_info += f"    - {param_name} ({param_type}): {param_desc}\n"

        return tools_info
    except Exception as e:
        logger.error(f"Failed to get tools list: {e}")
        return f"Error retrieving tools list: {str(e)}"


# Function to test server connection
def test_server_connection():
    """Test if the server is reachable and responsive."""
    try:
        # Create a short-lived connection to check server
        server = MCPServerSse(get_server_params())

        # Try connecting (this is synchronous)
        asyncio.run(server.connect())
        asyncio.run(server.cleanup())
        logger.info("✅ Server connection test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Server connection test failed: {e}")
        return False


# Manual math operation handler as ultimate fallback
def manual_math(query):
    """Parse and solve a math problem without using the server."""
    query = query.lower()

    # Check if user is asking for available tools/functions
    if "list" in query and ("tools" in query or "functions" in query
                            or "operations" in query):
        return """
Available tools:
1. add - Add two numbers together (e.g., "add 3 and 4")
2. multiply - Multiply two numbers together (e.g., "multiply 5 and 6") 
3. divide - Divide the first number by the second (e.g., "divide 10 by 2")
"""

    try:
        if "add" in query or "plus" in query or "sum" in query:
            # Extract numbers using a simple approach
            numbers = [int(s) for s in query.split() if s.isdigit()]
            if len(numbers) >= 2:
                result = numbers[0] + numbers[1]
                return f"The sum of {numbers[0]} and {numbers[1]} is {result}"

        elif "multiply" in query or "times" in query or "product" in query:
            numbers = [int(s) for s in query.split() if s.isdigit()]
            if len(numbers) >= 2:
                result = numbers[0] * numbers[1]
                return f"The product of {numbers[0]} and {numbers[1]} is {result}"

        elif "divide" in query or "quotient" in query:
            numbers = [int(s) for s in query.split() if s.isdigit()]
            if len(numbers) >= 2:
                if numbers[1] == 0:
                    return "Cannot divide by zero"
                result = numbers[0] / numbers[1]
                return f"{numbers[0]} divided by {numbers[1]} is {result}"

        return "I couldn't parse your math request. Try something like 'add 3 and 4'."
    except Exception as e:
        logger.error(f"Manual math error: {e}")
        return f"Error performing calculation: {str(e)}"


def main():
    try:
        logger.info("Initializing math system...")

        # Test server connection first
        server_available = test_server_connection()

        if server_available:
            math_agent = initialize_math_system()
            print("\nMath Calculator Ready! (Server connection successful)")
        else:
            print(
                "\nServer connection failed - using fallback calculator mode")
            math_agent = None

        print("Ask me any math question!")
        print("Examples: 'what is 5 plus 3?' or 'can you multiply 4 and 6?'")
        print("Type 'list tools' to see available operations")
        print("Type 'exit' to quit\n")

        while True:
            try:
                query = input("What would you like to calculate? ").strip()
                if not query:
                    continue
                if query.lower() == 'exit':
                    break

                # Handle special commands
                if query.lower() in ('list tools', 'show tools',
                                     'available tools', 'what tools'):
                    if server_available:
                        # Get tools list from server
                        tools_info = asyncio.run(get_tools_list())
                        print(f"\n{tools_info}\n")
                    else:
                        # Use manual fallback
                        print(manual_math("list tools"))
                    continue

                logger.info(f"Processing query: {query}")

                # First try the agent if available
                if math_agent and server_available:
                    try:
                        result = math_agent.run(query)
                        print(f"\nResult: {result}\n")
                        continue
                    except Exception as e:
                        logger.error(f"Agent error: {e}")
                        print("Agent encountered an error, trying fallback...")

                # If agent fails or isn't available, use manual calculator
                result = manual_math(query)
                print(f"\nCalculation result: {result}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Sorry, there was an error: {str(e)}")

    except Exception as e:
        logger.error(f"System initialization error: {e}")
        print(f"Failed to start the math system: {str(e)}")


if __name__ == "__main__":
    main()
