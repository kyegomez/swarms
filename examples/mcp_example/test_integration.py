
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT
from swarms.tools.mcp_integration import MCPServerSseParams
import logging

def main():
    # Configure multiple MCP server connections
    math_server = MCPServerSseParams(
        url="http://0.0.0.0:6274",
        headers={"Content-Type": "application/json"},
        timeout=10.0,
        sse_read_timeout=300.0
    )

    calc_server = MCPServerSseParams(
        url="http://0.0.0.0:6275", 
        headers={"Content-Type": "application/json"},
        timeout=10.0,
        sse_read_timeout=300.0
    )

    # Initialize multiple agents with different MCP capabilities
    math_agent = Agent(
        agent_name="Math-Agent",
        agent_description="Agent that performs math operations",
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        max_loops=1,
        mcp_servers=[math_server],
        streaming_on=True
    )

    calc_agent = Agent(
        agent_name="Calc-Agent", 
        agent_description="Agent that performs calculations",
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        max_loops=1,
        mcp_servers=[calc_server],
        streaming_on=True
    )

    agents = [math_agent, calc_agent]
    
    try:
        # Test each agent
        for agent in agents:
            print(f"\nTesting {agent.agent_name}...")
            print("\nDiscovering available tools from MCP server...")
            tools = agent.mcp_tool_handling()
            print(f"\nAvailable tools for {agent.agent_name}:", tools)
            
            while True:
                user_input = input(f"\nEnter a math operation for {agent.agent_name} (or 'exit' to switch agent): ")
                
                if user_input.lower() == 'exit':
                    break
                    
                try:
                    result = agent.run(user_input)
                    print(f"\nResult from {agent.agent_name}:", result)
                except Exception as e:
                    print(f"Error processing request: {e}")

    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
