from swarms import Agent

# Initialize a new agent
agent = Agent(
    model_name="gpt-4o-mini",  # Specify the LLM
    max_loops=1,  # Set the number of interactions
    interactive=True,  # Enable interactive mode for real-time feedback
    streaming_on=True,
    print_on=False,
)

# Run the agent with a task
agent.run("What are the key benefits of using a multi-agent system?")
