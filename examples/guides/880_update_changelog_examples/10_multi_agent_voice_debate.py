"""
Multi-Agent Voice Debate Example

This example demonstrates two agents debating a topic using different
voices with text-to-speech capabilities.
"""

from swarms import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from voice_agents import StreamingTTSCallback

agent1 = Agent(
    agent_name="Socrates",
    system_prompt="You argue from a philosophical perspective, questioning assumptions.",
    model_name="gpt-4o-mini",
    max_loops=1,
    streaming_on=True,
    streaming_callback=StreamingTTSCallback(
        voice="onyx", model="openai/tts-1"
    ),
)

agent2 = Agent(
    agent_name="Simone",
    system_prompt="You argue from a practical perspective, focusing on real-world applications.",
    model_name="gpt-4o-mini",
    max_loops=1,
    streaming_on=True,
    streaming_callback=StreamingTTSCallback(
        voice="nova", model="openai/tts-1"
    ),
)

conversation = Conversation()
task = "Should AI be regulated? Debate this topic."

conversation.add(role="User", content=task)

response1 = agent1.run(task)
conversation.add(role="Socrates", content=response1)

response2 = agent2.run(f"{task}\n\nPrevious argument: {response1}")
conversation.add(role="Simone", content=response2)

result = history_output_formatter(
    conversation.get_history(), "str-all-except-first"
)

print("Multi-Agent Voice Debate Result:")
print(result)
