"""
Single Voice Agent Example

This example demonstrates a single agent with voice capabilities
using text-to-speech (TTS) for real-time speech output.
"""

from swarms import Agent
from voice_agents import StreamingTTSCallback

tts_callback = StreamingTTSCallback(
    voice="onyx",
    model="openai/tts-1",
)

agent = Agent(
    agent_name="Voice-Agent",
    system_prompt="You are a helpful assistant that speaks responses.",
    model_name="gpt-4o-mini",
    max_loops=1,
    streaming_on=True,
    streaming_callback=tts_callback,
)

task = (
    "Explain the concept of artificial intelligence in simple terms"
)
response = agent.run(task)

print("Voice Agent Response:")
print(response)
