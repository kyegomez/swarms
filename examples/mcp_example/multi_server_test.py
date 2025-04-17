from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams

# Configure servers
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

# Create agent with access to both servers
calculator = Agent(
    agent_name="Calculator",
    agent_description="Math and calculation expert",
    system_prompt="You are a math expert. Use the available calculation tools to solve problems.",
    max_loops=1,
    mcp_servers=[math_server, calc_server],
    interactive=True,
    streaming_on=True,
    model_name="gpt-4o-mini"
)

def main():
    print("\nMulti-Server Calculator Test")
    print("Type 'exit' to quit\n")
    print("Example commands:")
    print("- add 5 and 3")
    print("- multiply 4 by 7\n")

    while True:
        try:
            user_input = input("\nEnter calculation: ")

            if user_input.lower() == 'exit':
                break

            response = calculator.run(user_input)
            print(f"\nCalculation result: {response}")

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()