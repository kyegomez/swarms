from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT

# Configure multiple MCP servers
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

# Create specialized agents with different server access
math_agent = Agent(
    agent_name="Math-Specialist",
    agent_description="Advanced mathematics expert",
    system_prompt="You are a mathematics expert. Use available math operations.",
    max_loops=1,
    mcp_servers=[math_server],
    interactive=True,
    streaming_on=True,
    model_name="gpt-4o-mini"
)

finance_agent = Agent(
    agent_name="Finance-Specialist",
    agent_description="Financial calculations expert", 
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    mcp_servers=[calc_server],
    interactive=True,
    streaming_on=True,
    model_name="gpt-4o-mini"
)

# Multi-server agent with access to all operations
super_agent = Agent(
    agent_name="Super-Calculator",
    agent_description="Multi-capable calculation expert",
    system_prompt="You have access to multiple calculation servers. Use them appropriately.",
    max_loops=1,
    mcp_servers=[math_server, calc_server],
    interactive=True,
    streaming_on=True,
    model_name="gpt-4o-mini"
)

def format_agent_output(agent_name: str, output: str) -> str:
    return f"""
╭─── {agent_name} Response ───╮
{output}
╰───────────────────────────╯
"""

def main():
    print("\nMulti-Agent MCP Test Environment")
    print("Type 'exit' to quit\n")
    print("Available commands:")
    print("- math: <calculation> (e.g., 'math: add 5 and 3')")
    print("- finance: <calculation> (e.g., 'finance: calculate compound interest on 1000 at 5% for 3 years')")
    print("- Any other calculation will use the super agent\n")

    while True:
        try:
            user_input = input("\nEnter your calculation request: ")

            if user_input.lower() == 'exit':
                break

            # Route request to appropriate agent based on keywords
            if 'finance' in user_input.lower():
                response = finance_agent.run(user_input)
                print("\nFinance Agent Response:")
                print("-" * 50)
                print(f"Response: {response}")
                print("-" * 50)
            elif 'math' in user_input.lower():
                response = math_agent.run(user_input)
                print("\nMath Agent Response:")
                print("-" * 50) 
                print(f"Response: {response}")
                print("-" * 50)
            else:
                response = super_agent.run(user_input)
                if isinstance(response, str):
                    print("\nSuper Agent Response:")
                    print("-" * 50)
                    print(f"Calculation Result: {response}")
                    print("-" * 50)
                else:
                    print("\nError: Unexpected response format")

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
        except Exception as e:
            print(f"Error processing request: {e}")

if __name__ == "__main__":
    main()