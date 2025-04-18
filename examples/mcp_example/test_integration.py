
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT
from swarms.tools.mcp_integration import MCPServerSseParams
import logging

def main():
    # Configure MCP server connection
    server = MCPServerSseParams(
        url="http://0.0.0.0:6274",
        headers={"Content-Type": "application/json"},
        timeout=10.0,
        sse_read_timeout=300.0
    )

    # Initialize agent with MCP capabilities
    agent = Agent(
        agent_name="Math-Agent",
        agent_description="Agent that performs math operations",
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        max_loops=1,
        mcp_servers=[server],
        streaming_on=True
    )

    try:
        # First get available tools from server
        print("\nDiscovering available tools from MCP server...")
        tools = agent.mcp_tool_handling()
        print("\nAvailable tools:", tools)
        
        while True:
            # Get user input
            user_input = input("\nEnter a math operation (or 'exit' to quit): ")
            
            if user_input.lower() == 'exit':
                break
                
            # Process user input through agent
            try:
                result = agent.run(user_input)
                print("\nResult:", result)
            except Exception as e:
                print(f"Error processing request: {e}")

    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
