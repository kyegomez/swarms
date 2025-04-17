from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT 
from swarms.tools.mcp_integration import MCPServerSseParams
import logging
from datetime import datetime

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
    math_agent = setup_agent(
        "Math-Agent",
        "Handles mathematical calculations",
        [math_server]
    )

    print("\nMulti-Agent MCP Test Environment")
    print("Type 'exit' to quit\n")

    while True:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_input = input(f"\n[{timestamp}] Enter your request (or Ctrl+C to exit): ")

            if user_input.lower() == 'exit':
                break

            if any(op in user_input.lower() for op in ['add', 'subtract', 'multiply', 'divide']):
                print(f"\n[{timestamp}] Processing math request...")
                result = math_agent.run(user_input)
                print(f"\n[{timestamp}] Math calculation result: {result}")

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
        except Exception as e:
            print(f"[{timestamp}] Error processing request: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()