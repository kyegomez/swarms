from swarms import Agent
from voice_agents import StreamingTTSCallback

tts_callback = StreamingTTSCallback(
    voice="alloy", model="openai/tts-1", stream_mode=True
)

agent = Agent(
    model_name="gpt-4.1",
    marketplace_prompt_id="6d165e47-1827-4abe-9a84-b25005d8e3b4",
    max_loops="auto",
    streaming_on=True,
    interactive=True,
    streaming_callback=tts_callback,
)

response = agent.run("Hello, what can you help me with?")
print(response)
