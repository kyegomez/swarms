from swarms import Agent

# Enable real-time streaming
agent = Agent(
    agent_name="StoryAgent",
    model_name="gpt-4o-mini",  # 🔥 This enables real streaming!
    max_loops=4,
    streaming_on=True,
    print_on=True,
    output_type="all",
)

# This will now stream in real-time with beautiful UI!
response = agent.run(
    "Tell me a detailed story about Humanity colonizing the stars"
)
# print(response)
