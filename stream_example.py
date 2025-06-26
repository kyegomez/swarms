from swarms import Agent

# Enable real-time streaming
agent = Agent(
    agent_name="StoryAgent",
    model_name="gpt-4o-mini", 
    streaming_on=True,  # 🔥 This enables real streaming!
    max_loops=1,
)

# This will now stream in real-time with beautiful UI!
response = agent.run("Tell me a detailed story...")