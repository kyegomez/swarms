# Voice Agents Overview

The Swarms framework supports the creation of interactive, voice-enabled agents through integration with the `voice-agents` package. These agents can perceive and respond using human-like speech, enabling hands-free interaction and more natural user experiences.

## Core Features

- **Real-time Streaming TTS**: Leveraging OpenAI's TTS models to speak responses sentence-by-sentence as they are generated, minimizing latency.
- **Differentiated Voices**: Multiple voice profiles (alloy, onyx, nova, etc.) to give each agent in a swarm a unique personality.
- **Speech-to-Text (STT)**: Integration for voice-based task inputs, allowing users to talk directly to their agent swarms.
- **Seamless Integration**: Works with the standard `Agent` class and complex multi-agent architectures like debates and sequential workflows.

## Available Tutorials

In this section, you will find step-by-step guides on implementing voice capabilities:

1. [**Single Speech Agent**](single_agent_speech.md): Learn how to add real-time text-to-speech to a single standalone agent.
2. [**Multi-Agent Speech Debate**](multi_agent_speech_debate.md): A complex example showing two agents debating a topic using different voices, with optional voice-to-text input.

## Getting Started

To use these features, you'll need to install the `voice-agents` package alongside `swarms`:

```bash
pip install -U swarms voice-agents
```

You'll also need a valid `OPENAI_API_KEY` to access the TTS and STT models.

