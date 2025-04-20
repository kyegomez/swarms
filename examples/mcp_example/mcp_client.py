import os
from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams
from swarms.prompts.agent_prompts import MATH_AGENT_PROMPT
from loguru import logger

# Set OpenAI API key

def initialize_math_system():
    """Initialize the math agent with MCP server configuration."""
    # Configure MCP server connection with SSE transport
    math_server = MCPServerSseParams(
        url="http://localhost:8000",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        },
        timeout=5.0,
        sse_read_timeout=30.0)

    # Initialize math agent with specific model
    math_agent = Agent(
        agent_name="Math Agent",
        agent_description="Basic math calculator",
        system_prompt=MATH_AGENT_PROMPT,
        max_loops=1,
        mcp_servers=[math_server],
        streaming_on=False,
        model_name="gpt-4o-mini",
        temperature=0.1)

    return math_agent

def process_query(math_agent, query):
    """Process a single math query."""
    try:
        result = math_agent.run(query)
        # Clean up the result to show only the number or error message
        if isinstance(result, (int, float)):
            return result
        elif "error" in result.lower():
            return result
        else:
            # Try to extract just the number from the result
            try:
                return float(result)
            except:
                return "Error: Invalid result format"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Initialize the math system
    math_agent = initialize_math_system()
    print("\nMath Calculator Ready!")
    print("Available operations: add, multiply, divide")
    print("Example: 'add 5 and 3' or 'multiply 4 by 6'")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("Enter math operation: ").strip()
            if not query:
                continue
            if query.lower() == 'exit':
                break

            result = process_query(math_agent, query)
            print(f"Result: {result}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
