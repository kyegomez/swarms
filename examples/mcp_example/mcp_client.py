from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams
from swarms.prompts.agent_prompts import MATH_AGENT_PROMPT
from loguru import logger

def initialize_math_system():
    """Initialize the math agent with MCP server configuration."""
    math_server = MCPServerSseParams(
        url="http://0.0.0.0:8000",
        headers={"Content-Type": "application/json"},
        timeout=5.0,
        sse_read_timeout=30.0
    )

    math_agent = Agent(
        agent_name="Math Agent",
        agent_description="Basic math calculator", 
        system_prompt=MATH_AGENT_PROMPT,
        max_loops=1,
        mcp_servers=[math_server]
    )

    return math_agent

def main():
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

            result = math_agent.run(query)
            print(f"Result: {result}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()