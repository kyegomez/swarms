from swarms import Agent

# Create an agent with a marketplace prompt loaded in one line
# Replace the marketplace_prompt_id with your actual prompt ID from the marketplace
agent = Agent(
    model_name="gpt-4o-mini",
    marketplace_prompt_id="0ff9cc2f-390a-4eb1-9d3d-3a045cd2682e",  # The prompt ID from the Swarms marketplace
    max_loops="auto",
    interactive=True,
)

# Run the agent - it will use the system prompt loaded from the marketplace
response = agent.run("Hello, what can you help me with?")
print(response)
