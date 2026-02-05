# Voice Agents Examples

This directory contains examples for building speech-enabled agents with the Swarms framework using the `voice-agents` package for streaming text-to-speech (TTS) and speech-to-text (STT) capabilities.

## Requirements

- Install the [voice-agents](https://pypi.org/project/voice-agents/) package: `pip install voice-agents`
- OpenAI API key (for TTS/STT when using OpenAI models)

## Examples

| Example | Description |
|---------|-------------|
| [agent_speech.py](agent_speech.py) | Basic agent with speech output capabilities |
| [agent_with_speech.py](agent_with_speech.py) | Speech-enabled agent with streaming TTS callback |
| [debate_with_speech.py](debate_with_speech.py) | Multi-agent debate with voice output for each agent |
| [google_calendar_agent.py](google_calendar_agent.py) | Voice agent integrated with Google Calendar |
| [hiearchical_speech_swarm.py](hiearchical_speech_swarm.py) | Hierarchical swarm where each role has a distinct voice |
| [run_auto_agent_with_speech.py](run_auto_agent_with_speech.py) | Autonomous agent with terminal (bash) access and streaming TTS |

## Usage

Use `StreamingTTSCallback` from the `voice_agents` package with any Swarms agent's `streaming_callback` parameter. You can choose voices such as `alloy`, `echo`, `fable`, `onyx`, `nova`, or `shimmer`.

```python
from swarms import Agent
from voice_agents import StreamingTTSCallback

agent = Agent(model_name="anthropic/claude-sonnet-4-5", ...)
tts_callback = StreamingTTSCallback(voice="alloy", model="tts-1")
result = agent.run(task="Hello!", streaming_callback=tts_callback)
tts_callback.flush()
```

## Related

- [Single agent speech](https://docs.swarms.world/en/latest/swarms/examples/single_agent_speech/) documentation
- [Hierarchical speech swarm](https://docs.swarms.world/en/latest/swarms/examples/hierarchical_speech_swarm/) documentation
- [guides/changelog_890/](../guides/changelog_890/) and [guides/880_update_changelog_examples/](../guides/880_update_changelog_examples/) for more voice examples
