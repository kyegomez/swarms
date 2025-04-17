
from swarms.structs.agent import Agent

def main():
    # Initialize basic agent
    agent = Agent(
        agent_name="Basic-Example-Agent",
        agent_description="A simple example agent",
        system_prompt="You are a helpful assistant.",
        model_name="gpt-4",
    )
    
    # Run the agent
    response = agent.run("What is 2+2?")
    print(f"Agent response: {response}")

if __name__ == "__main__":
    main()
