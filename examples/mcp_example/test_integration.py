
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT 
from swarms.tools.mcp_integration import MCPServerSseParams
import logging
import time

def setup_agent(name: str, description: str, servers: list) -> Agent:
    """Setup an agent with MCP server connections"""
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        max_loops=1,
        mcp_servers=servers,
        streaming_on=True
    )

def main():
    # Configure MCP server connections
    math_server = MCPServerSseParams(
        url="http://0.0.0.0:6274",
        headers={"Content-Type": "application/json"},
        timeout=10.0
    )
    
    calc_server = MCPServerSseParams(
        url="http://0.0.0.0:6275",
        headers={"Content-Type": "application/json"},
        timeout=10.0
    )

    # Initialize specialized agents
    coordinator = setup_agent(
        "Coordinator",
        "Analyzes tasks and coordinates between specialized agents",
        [math_server, calc_server]
    )
    
    math_agent = setup_agent(
        "Math-Agent",
        "Handles mathematical calculations",
        [math_server]
    )
    
    business_agent = setup_agent(
        "Business-Agent",
        "Handles business calculations",
        [calc_server]
    )

    print("\nMulti-Agent MCP Test Environment")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("\nEnter your request: ")
            
            if user_input.lower() == 'exit':
                break
                
            # Coordinator analyzes task
            print("\nCoordinator analyzing task...")
            coordinator_response = coordinator.run(
                f"Analyze this task and determine required calculations: {user_input}"
            )
            print(f"\nCoordinator's plan: {coordinator_response}")
            
            # Route to appropriate agent(s)
            if "profit" in user_input.lower() or "margin" in user_input.lower():
                print("\nRouting to Business Agent...")
                result = business_agent.run(user_input)
                print(f"\nBusiness calculation result: {result}")
            
            if any(op in user_input.lower() for op in ['add', 'subtract', 'multiply', 'divide']):
                print("\nRouting to Math Agent...")
                result = math_agent.run(user_input)
                print(f"\nMath calculation result: {result}")

        except Exception as e:
            print(f"Error processing request: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
