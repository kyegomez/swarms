# Creating a Single Speech Agent

This tutorial demonstrates how to create a single AI agent with real-time text-to-speech (TTS) capabilities using the Swarms framework and the `voice-agents` package. This setup is ideal for interactive applications where you want the agent to "speak" its responses as they are generated.

## Prerequisites

- Python 3.10+
- OpenAI API key (for both LLM and TTS)
- `swarms` library
- `voice-agents` library

## Tutorial Steps

1. **Install Dependencies**
   Install the necessary packages:
   ```bash
   pip3 install -U swarms voice-agents
   ```

2. **Set Up Environment**
   Ensure your OpenAI API key is set in your environment:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Initialize the Agent**
   Create an agent with `streaming_on=True`. This is crucial for the TTS callback to work in real-time.

4. **Configure the TTS Callback**
   Use the `StreamingTTSCallback` from the `voice-agents` package. You can choose different voices like "alloy", "echo", "fable", "onyx", "nova", or "shimmer".

5. **Run the Agent**
   Pass the `streaming_callback` to the `agent.run()` method.

## Code Example

```python
from swarms import Agent
from voice_agents import StreamingTTSCallback

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4o",
    dynamic_temperature_enabled=True,
    max_loops=1,
    streaming_on=True, # Required for real-time TTS
)

# Create the streaming TTS callback
# voice: alloy, echo, fable, onyx, nova, shimmer
tts_callback = StreamingTTSCallback(voice="alloy", model="tts-1")

# Run the agent with streaming TTS callback
out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
    streaming_callback=tts_callback,
)

# Flush any remaining text in the buffer to ensure the last sentence is spoken
tts_callback.flush()

print(out)
```

## How it Works

- **Streaming**: When `streaming_on=True`, the agent yields tokens as they are generated.
- **Callback**: The `StreamingTTSCallback` collects these tokens into sentences and sends them to the OpenAI TTS API.
- **Real-time Audio**: Audio is played back as soon as the first sentence is processed, significantly reducing latency compared to waiting for the full response.
- **Flushing**: The `.flush()` method is called at the end to process any remaining text that didn't end with a sentence delimiter.

