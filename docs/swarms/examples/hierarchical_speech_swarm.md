# Hierarchical Swarm with Speech Capabilities

This tutorial demonstrates how to create a hierarchical swarm where multiple specialized agents communicate through voice using text-to-speech (TTS) capabilities. Each agent has a unique voice, making it easy to distinguish who is speaking during collaborative task execution.

## Overview

A hierarchical swarm combines the power of multi-agent collaboration with voice communication. In this architecture:

- **Director Agent**: Coordinates the overall workflow and distributes tasks

- **Worker Agents**: Specialized agents that execute specific tasks

- **Voice Communication**: Each agent speaks their responses using distinct TTS voices

This creates an immersive experience where you can hear agents collaborating in real-time.

## Prerequisites

- Python 3.10+
- OpenAI API key (for both LLM and TTS)
- `swarms` library
- `voice-agents` library

## Tutorial Steps

1. **Install Dependencies**
   ```bash
   pip3 install -U swarms voice-agents
   ```

2. **Set Up Environment**
   Ensure your OpenAI API key is set:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Create TTS Callbacks**
   Define distinct voices for each agent to differentiate speakers.

4. **Initialize Agents with TTS**
   Create specialized agents with `streaming_on=True` and assign TTS callbacks directly.

5. **Create Hierarchical Swarm**
   Set up the swarm with your speech-enabled agents.

6. **Run the Swarm**
   Execute tasks and listen to agents collaborate through voice.

## Complete Code Example

```python
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
    streaming_callback=tts_callbacks.get("Research-Analyst"),  # Direct TTS callback
)

analysis_agent = Agent(
    agent_name="Data-Analyst",
    agent_description="Expert in data analysis and pattern recognition",
    model_name="gpt-4.1",
    max_loops=1,
    verbose=False,
    streaming_on=True,  # Required for TTS streaming
    streaming_callback=tts_callbacks.get("Data-Analyst"),  # Direct TTS callback
)

strategy_agent = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Specialized in strategic planning and recommendations",
    model_name="gpt-4.1",
    max_loops=1,
    verbose=False,
    streaming_on=True,  # Required for TTS streaming
    streaming_callback=tts_callbacks.get("Strategy-Consultant"),  # Direct TTS callback
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
```

## Key Components Explained

### 1. TTS Callback Configuration

Each agent gets a unique voice to distinguish speakers:

```python
tts_callbacks = {
    "Research-Analyst": StreamingTTSCallback(voice="onyx", model="openai/tts-1"),
    "Data-Analyst": StreamingTTSCallback(voice="nova", model="openai/tts-1"),
    "Strategy-Consultant": StreamingTTSCallback(voice="alloy", model="openai/tts-1"),
    "Director": StreamingTTSCallback(voice="echo", model="openai/tts-1"),
}
```

**Available Voices:**

| Voice    | Description                      |
|----------|----------------------------------|
| `alloy`  | Clear, professional voice        |
| `echo`   | Distinctive, commanding voice    |
| `fable`  | Warm, narrative voice            |
| `onyx`   | Deeper, authoritative voice      |
| `nova`   | Softer, analytical voice         |
| `shimmer`| Bright, energetic voice          |

### 2. Agent Configuration

Key requirements for speech-enabled agents:

- **`streaming_on=True`**: Enables real-time token streaming required for TTS

- **`streaming_callback`**: Direct assignment of TTS callback to each agent

- **`max_loops=1`**: Typically set to 1 for hierarchical swarms (director handles coordination)

```python
research_agent = Agent(
    agent_name="Research-Analyst",
    agent_description="Specialized in comprehensive research and data gathering",
    model_name="gpt-4.1",
    max_loops=1,
    verbose=False,
    streaming_on=True,  # Required for TTS streaming
    streaming_callback=tts_callbacks.get("Research-Analyst"),  # Direct TTS callback
)
```

### 3. Hierarchical Swarm Setup

The swarm coordinates multiple agents through a director:

```python
swarm = HierarchicalSwarm(
    name="Swarms Corporation Operations",
    description="Enterprise-grade hierarchical swarm for complex task execution",
    agents=[research_agent, analysis_agent, strategy_agent],
    max_loops=1,
    director_model_name="gpt-4.1",
    director_temperature=0.7,
    planning_enabled=True,
)
```

**Key Parameters:**
- `agents`: List of worker agents with TTS capabilities
- `director_model_name`: Model for the coordinating director
- `planning_enabled`: Allows director to create execution plans
- `max_loops`: Number of feedback iterations

### 4. Buffer Flushing

Always flush TTS buffers after execution to ensure all speech is played:

```python
# Flush all TTS buffers to ensure everything is spoken
for callback in tts_callbacks.values():
    callback.flush()
```

This is critical because the TTS callback buffers text and may not automatically flush incomplete sentences.

## How It Works

1. **Task Distribution**: The director agent receives the task and creates a plan
2. **Agent Assignment**: Director distributes subtasks to specialized worker agents
3. **Real-time Speech**: As each agent generates responses, tokens are streamed to their TTS callback
4. **Voice Differentiation**: Each agent's unique voice makes it clear who is speaking
5. **Collaboration**: Agents can reference each other's work, creating a natural conversation flow

## Advanced Customization

### Custom Voice Selection

Choose voices that match agent personalities:

```python
# Authoritative leader
leader_voice = StreamingTTSCallback(voice="onyx", model="openai/tts-1")

# Analytical researcher
researcher_voice = StreamingTTSCallback(voice="nova", model="openai/tts-1")

# Professional consultant
consultant_voice = StreamingTTSCallback(voice="alloy", model="openai/tts-1")
```


## Best Practices

| Best Practice            | Description                                                                              |
|--------------------------|------------------------------------------------------------------------------------------|
| **Voice Selection**      | Use distinct voices for each agent to avoid confusion                                    |
| **Buffer Management**    | Always flush TTS buffers after execution                                                 |
| **Error Handling**       | Flush buffers even on errors to prevent audio glitches                                   |
| **Streaming Requirement**| Always set `streaming_on=True` for TTS to work                                           |
| **Direct Assignment**    | Assign TTS callbacks directly to agents for better control                               |


### Tips for Audio Playback

- **Audio Overlap**: Agents normally speak sequentially, but if you hear overlapping audio, check that agents aren’t being executed concurrently. Adjust `max_loops` or modify the execution order if necessary.

- **Missing Audio**: Always flush TTS buffers after execution with `callback.flush()`. Make sure agents are generating responses and that the TTS callback is actively receiving streamed tokens.
