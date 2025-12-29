from swarms import Agent
from voice_agents.main import stream_tts_openai

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    top_p=None,
)

# Run the agent with streaming TTS callback
out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
)

stream_tts_openai(
    [out],
    stream_mode=True,
)
