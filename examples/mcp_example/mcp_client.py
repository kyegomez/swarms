from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams

def main():
    # Configure MCP server connection 
    math_server = MCPServerSseParams(
        url="http://0.0.0.0:8000/mcp",
        headers={"Content-Type": "application/json"},
        timeout=5.0, 
        sse_read_timeout=30.0
    )

    # Initialize agent with MCP server
    agent = Agent(
        agent_name="Math Agent",
        agent_description="Agent for performing mathematical operations",
        system_prompt="""You are a mathematical computation specialist. Use the available MCP server tools to:
        - Add numbers
        - Multiply numbers  
        - Divide numbers

        Always:
        1. Use only tools available from the MCP server
        2. Explain your mathematical approach
        3. Show your work step by step""",
        max_loops=1,
        mcp_servers=[math_server],
        streaming_on=True,
        model_name="gpt-4o-mini"
    )

    print("\nMath Agent initialized with MCP capabilities")
    print("Available operations:")
    print("- Addition")
    print("- Multiplication") 
    print("- Division")

    while True:
        query = input("\nEnter a math problem (or 'exit' to quit): ")

        if query.lower() == 'exit':
            break

        # Process query through agent
        result = agent.run(query)
        print("\nResult:", result)

if __name__ == "__main__":
    main()