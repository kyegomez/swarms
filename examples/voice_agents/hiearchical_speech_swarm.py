"""
Hierarchical Swarm with Speech Capabilities

This example demonstrates a hierarchical swarm where agents communicate
with each other through voice using text-to-speech (TTS) capabilities.
Each agent has a unique voice, making it easy to distinguish who is speaking.
"""

from swarms import Agent, HierarchicalSwarm
from voice_agents import StreamingTTSCallback

# Create TTS callbacks for each agent with distinct voices
tts_callbacks = {
    "Research-Analyst": StreamingTTSCallback(
        voice="onyx", model="openai/tts-1"
    ),  # Deeper, authoritative voice
    "Data-Analyst": StreamingTTSCallback(
        voice="nova", model="openai/tts-1"
    ),  # Softer, analytical voice
    "Strategy-Consultant": StreamingTTSCallback(
        voice="alloy", model="openai/tts-1"
    ),  # Clear, professional voice
    "Director": StreamingTTSCallback(
        voice="echo", model="openai/tts-1"
    ),  # Distinctive voice for director
}

# Create specialized agents with streaming enabled for TTS
# Assign TTS callbacks directly to each agent
research_agent = Agent(
    agent_name="Research-Analyst",
    agent_description="Specialized in comprehensive research and data gathering",
    model_name="gpt-4.1",
    max_loops=1,
    verbose=False,
    streaming_on=True,  # Required for TTS streaming
    streaming_callback=tts_callbacks.get(
        "Research-Analyst"
    ),  # Direct TTS callback
)

analysis_agent = Agent(
    agent_name="Data-Analyst",
    agent_description="Expert in data analysis and pattern recognition",
    model_name="gpt-4.1",
    max_loops=1,
    verbose=False,
    streaming_on=True,  # Required for TTS streaming
    streaming_callback=tts_callbacks.get(
        "Data-Analyst"
    ),  # Direct TTS callback
)

strategy_agent = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Specialized in strategic planning and recommendations",
    model_name="gpt-4.1",
    max_loops=1,
    verbose=False,
    streaming_on=True,  # Required for TTS streaming
    streaming_callback=tts_callbacks.get(
        "Strategy-Consultant"
    ),  # Direct TTS callback
)

# Create hierarchical swarm
swarm = HierarchicalSwarm(
    name="Swarms Corporation Operations",
    description="Enterprise-grade hierarchical swarm for complex task execution with voice communication",
    agents=[research_agent, analysis_agent, strategy_agent],
    max_loops=1,
    interactive=False,
    director_model_name="gpt-4.1",
    director_temperature=0.7,
    director_top_p=None,
    planning_enabled=True,
)


# Define the task
task = (
    "Conduct a comprehensive analysis of renewable energy stocks. "
    "Research the current market trends, analyze the data, and provide "
    "strategic recommendations for investment."
)

# Run the swarm (agents already have their TTS callbacks assigned)
try:
    result = swarm.run(task=task)

    # Flush all TTS buffers to ensure everything is spoken
    for callback in tts_callbacks.values():
        callback.flush()

except Exception:
    # Still flush buffers on error
    for callback in tts_callbacks.values():
        callback.flush()
    raise
