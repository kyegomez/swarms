from swarms import Agent

# Create an agent with a marketplace prompt loaded in one line
# Replace the marketplace_prompt_id with your actual prompt ID from the marketplace
agent = Agent(
    model_name="gpt-4.1",
    marketplace_prompt_id="1191250b-9fb3-42e0-b0e9-25ec83260ab2",  # The prompt ID from the Swarms marketplace
    max_loops="auto",
    interactive=True,
)

# Run the agent - it will use the system prompt loaded from the marketplace
response = agent.run("Hello, what can you help me with?")
print(response)
