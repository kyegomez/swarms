"""
Hierarchical Voice Swarm Example

This example demonstrates a hierarchical swarm where agents communicate
through voice using text-to-speech capabilities with distinct voices.
"""

from swarms import Agent, HierarchicalSwarm
from voice_agents import StreamingTTSCallback

tts_callbacks = {
    "Research-Analyst": StreamingTTSCallback(
        voice="onyx", model="openai/tts-1"
    ),
    "Data-Analyst": StreamingTTSCallback(
        voice="nova", model="openai/tts-1"
    ),
    "Strategy-Consultant": StreamingTTSCallback(
        voice="alloy", model="openai/tts-1"
    ),
    "Director": StreamingTTSCallback(
        voice="echo", model="openai/tts-1"
    ),
}

research_agent = Agent(
    agent_name="Research-Analyst",
    agent_description="Specialized in comprehensive research and data gathering",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
    streaming_on=True,
    streaming_callback=tts_callbacks.get("Research-Analyst"),
)

analysis_agent = Agent(
    agent_name="Data-Analyst",
    agent_description="Expert in data analysis and pattern recognition",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
    streaming_on=True,
    streaming_callback=tts_callbacks.get("Data-Analyst"),
)

strategy_agent = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Develops strategic recommendations and business insights",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
    streaming_on=True,
    streaming_callback=tts_callbacks.get("Strategy-Consultant"),
)

director_agent = Agent(
    agent_name="Director",
    agent_description="Coordinates the overall workflow and synthesizes final recommendations",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
    streaming_on=True,
    streaming_callback=tts_callbacks.get("Director"),
)

swarm = HierarchicalSwarm(
    name="Voice-Hierarchical-Swarm",
    director=director_agent,
    workers=[research_agent, analysis_agent, strategy_agent],
    max_loops=2,
)

task = "Analyze the renewable energy market and provide strategic recommendations"
result = swarm.run(task)

print("Hierarchical Voice Swarm Result:")
print(result)
