"""
Hierarchical Voice Agent Example

This example demonstrates hierarchical voice agent systems where agents
communicate through voice using distinct TTS voices. Each agent in the
hierarchy has a unique voice profile for clear identification.
"""

from swarms import Agent, HierarchicalSwarm

# Note: Requires voice-agents package: pip install voice-agents
try:
    from voice_agents import StreamingTTSCallback

    # Create TTS callbacks with distinct voices for each role
    director_callback = StreamingTTSCallback(voice="alloy", model="openai/tts-1")
    researcher_callback = StreamingTTSCallback(voice="onyx", model="openai/tts-1")
    analyst_callback = StreamingTTSCallback(voice="nova", model="openai/tts-1")

    # Create hierarchical agents with voice capabilities
    director = Agent(
        agent_name="Director",
        system_prompt="You are the director coordinating complex research projects with voice communication.",
        model_name="gpt-4o-mini",
        max_loops=1,
        streaming_on=True,
        streaming_callback=director_callback,
    )

    researcher = Agent(
        agent_name="Researcher",
        system_prompt="You are a research specialist gathering comprehensive information.",
        model_name="gpt-4o-mini",
        max_loops=1,
        streaming_on=True,
        streaming_callback=researcher_callback,
    )

    analyst = Agent(
        agent_name="Analyst",
        system_prompt="You are a data analyst providing insights and recommendations.",
        model_name="gpt-4o-mini",
        max_loops=1,
        streaming_on=True,
        streaming_callback=analyst_callback,
    )

    # Create hierarchical swarm with voice communication
    voice_swarm = HierarchicalSwarm(
        name="VoiceResearchSwarm",
        description="Hierarchical swarm with voice-enabled agents",
        agents=[researcher, analyst],
        director=director,
        max_loops=1,
    )

    task = "Research emerging AI technologies in healthcare and provide strategic recommendations."
    result = voice_swarm.run(task)

    # Flush TTS buffers
    director_callback.flush()
    researcher_callback.flush()
    analyst_callback.flush()

    print(result)

except ImportError:
    print("Voice agents package not available. Install with: pip install voice-agents")