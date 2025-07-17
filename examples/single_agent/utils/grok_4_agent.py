from swarms import Agent

# Initialize a new agent
agent = Agent(
    model_name="xai/grok-4-0709",  # Specify the LLM
    agent_name="financial-agent",
    agent_description="A financial agent that can help with financial planning and investment decisions",
    system_prompt="You are a financial agent that can help with financial planning and investment decisions",
    max_loops=1,  # Set the number of interactions
    interactive=True,  # Enable interactive mode for real-time feedback
    streaming=True,
)

# Run the agent with a task
agent.run("What are the key benefits of using a multi-agent system?")
