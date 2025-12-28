from swarms import Agent
from voice_agents import StreamingTTSCallback

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    top_p=None,
    streaming_on=True,
    interactive=False,
)

# Create the streaming TTS callback
tts_callback = StreamingTTSCallback(voice="alloy", model="tts-1")

# # Run the agent with streaming TTS callback
# out = agent.run(
#     task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
#     streaming_callback=tts_callback,
# )

# # Flush any remaining text in the buffer
# tts_callback.flush()

# print(out)
