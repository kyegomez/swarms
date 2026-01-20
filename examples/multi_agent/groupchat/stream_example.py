from swarms import Agent

# Enable real-time streaming
agent = Agent(
    agent_name="StoryAgent",
    # model_name="groq/llama-3.1-8b-instant",
    model_name="claude-3-5-sonnet-20240620",
    # system_prompt="",
    streaming_on=True,  # 🔥 This enables real streaming!
    max_loops=1,
    print_on=True,
    output_type="all",
)

# This will now stream in real-time with beautiful UI!
response = agent.run(
    "Tell me a detailed story about Humanity colonizing the stars"
)
print(response)
