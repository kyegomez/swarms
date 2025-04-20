
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
        agent_name="Math Assistant",
        agent_description="Friendly math calculator",
        system_prompt=MATH_AGENT_PROMPT,
        max_loops=1,
        mcp_servers=[math_server],
        model_name="gpt-3.5-turbo"
    )

    return math_agent

def main():
    math_agent = initialize_math_system()

    print("\nMath Calculator Ready!")
    print("Ask me any math question!")
    print("Examples: 'what is 5 plus 3?' or 'can you multiply 4 and 6?'")
    print("Type 'exit' to quit\n")

    while True:
        try:
            query = input("What would you like to calculate? ").strip()
            if not query:
                continue
            if query.lower() == 'exit':
                break

            result = math_agent.run(query)
            print(f"\nResult: {result}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
