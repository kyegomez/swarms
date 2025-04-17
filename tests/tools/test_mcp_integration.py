
import pytest
from swarms.tools.mcp_integration import MCPServerSseParams
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT

def test_interactive_multi_agent_mcp():
    # Configure two MCP servers
    server_one = MCPServerSseParams(
        url="http://0.0.0.0:6274",
        headers={"Content-Type": "application/json"}
    )
    
    server_two = MCPServerSseParams(
        url="http://0.0.0.0:6275", 
        headers={"Content-Type": "application/json"}
    )

    # Create two agents with different roles
    finance_agent = Agent(
        agent_name="Finance-Agent",
        agent_description="Financial analysis expert",
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        max_loops=1,
        mcp_servers=[server_one],
        interactive=True,
        streaming_on=True
    )

    research_agent = Agent(
        agent_name="Research-Agent", 
        agent_description="Market research specialist",
        system_prompt="You are a market research specialist. Analyze market trends and provide insights.",
        max_loops=1,
        mcp_servers=[server_two],
        interactive=True,
        streaming_on=True
    )

    try:
        # Interactive loop
        while True:
            # Get user input for which agent to use
            print("\nWhich agent would you like to interact with?")
            print("1. Finance Agent")
            print("2. Research Agent")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == "3":
                break
                
            # Get the task from user
            task = input("\nEnter your task for the agent: ")
            
            # Route to appropriate agent
            if choice == "1":
                response = finance_agent.run(task)
                print(f"\nFinance Agent Response:\n{response}")
            elif choice == "2":
                response = research_agent.run(task)
                print(f"\nResearch Agent Response:\n{response}")
            else:
                print("Invalid choice, please try again")

    except Exception as e:
        pytest.fail(f"Interactive multi-agent test failed: {e}")

def test_mcp_invalid_params():
    with pytest.raises(Exception):
        mcp_flow(None, {})

if __name__ == "__main__":
    test_interactive_multi_agent_mcp()
