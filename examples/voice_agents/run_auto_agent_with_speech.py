"""Autonomous agent with terminal access and streaming TTS. Requires: pip install voice-agents"""

from swarms import Agent
from voice_agents import StreamingTTSCallback

# Agent with autonomous looping and terminal (bash) access
agent = Agent(
    agent_name="Terminal-Agent",
    agent_description="Agent that can plan tasks and run bash commands on the terminal",
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    max_loops="auto",
    dynamic_context_window=True,
    selected_tools="all",
    top_p=None,
)

# Create the streaming TTS callback
# voice: alloy, echo, fable, onyx, nova, shimmer
tts_callback = StreamingTTSCallback(voice="alloy", model="tts-1")


if __name__ == "__main__":
    result = agent.run(
        task="Use the terminal to list the current directory, and see what files are in it.",
        streaming_callback=tts_callback,
    )

    # Flush any remaining text in the buffer to ensure the last sentence is spoken
    tts_callback.flush()

    print(result)
