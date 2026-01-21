# LiteLLM with Swarms

LiteLLM provides a unified interface for 100+ LLM providers. Swarms uses LiteLLM to support multiple providers through a single API.

## Quick Start

```python
from swarms import Agent

# Use any LiteLLM-supported model
agent = Agent(
    model_name="gpt-4o-mini",  # Change this to any provider
    max_loops=1,
)

response = agent.run("Hello, world!")
```

## Supported Providers

Switch providers by changing `model_name`:

```python
# OpenAI
Agent(model_name="gpt-4o")
Agent(model_name="gpt-4o-mini")
Agent(model_name="gpt-3.5-turbo")

# Anthropic Claude
Agent(model_name="claude-3-5-sonnet-20241022")
Agent(model_name="claude-3-opus")

# Google Gemini
Agent(model_name="gemini/gemini-pro")
Agent(model_name="gemini/gemini-1.5-pro")

# Azure OpenAI
Agent(model_name="azure/gpt-4")

# Ollama (local)
Agent(model_name="ollama/llama2")
Agent(model_name="ollama/mistral")

# Cohere
Agent(model_name="command-r")
Agent(model_name="command-r-plus")

# DeepSeek
Agent(model_name="deepseek/deepseek-chat")
Agent(model_name="deepseek/deepseek-r1")

# Groq
Agent(model_name="groq/llama-3.1-70b-versatile")

# OpenRouter
Agent(model_name="openrouter/google/palm-2-chat-bison")

# X.AI
Agent(model_name="xai/grok-beta")
```

## Using LiteLLM Wrapper Directly

```python
from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=2000,
    verbose=True,
)

response = llm.run("What is machine learning?")
```

## Features

### 1. Vision (Image Input)

```python
from swarms import Agent

agent = Agent(model_name="gpt-4o", max_loops=1)

# Supports: file path, URL, or base64
response = agent.run(
    "Describe this image",
    img="path/to/image.jpg"  # or URL or base64
)
```

### 2. Tool/Function Calling

```python
from swarms.utils.litellm_wrapper import LiteLLM

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

llm = LiteLLM(
    model_name="gpt-4o",
    tools_list_dictionary=tools,
    tool_choice="auto",
)

response = llm.run("What's the weather in San Francisco?")
```

### 3. Reasoning Models

```python
from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(
    model_name="openai/o1-preview",
    reasoning_enabled=True,
    max_tokens=4000,
)

response = llm.run("Solve this complex math problem...")
```

### 4. Streaming

```python
from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(model_name="gpt-4o", stream=True)

for chunk in llm.run("Tell me a story"):
    print(chunk, end="", flush=True)
```

### 5. Audio Input

```python
from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(
    model_name="gpt-4o",
    audio="path/to/audio.wav",
)

response = llm.run("Transcribe this audio")
```

### 6. Advanced Configuration

```python
from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(
    model_name="gpt-4o",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=4000,
    stream=False,
    verbose=True,
    retries=3,
    caching=False,
    top_p=1.0,
)

response = llm.run("Explain neural networks")
```

## Provider Setup

### Azure OpenAI

```python
import os
os.environ["AZURE_API_KEY"] = "your-key"
os.environ["AZURE_API_BASE"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-02-15-preview"

agent = Agent(model_name="azure/gpt-4", max_loops=1)
```

### Anthropic Claude

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key"

agent = Agent(model_name="claude-3-5-sonnet-20241022", max_loops=1)
```

### Google Gemini

```python
import os
os.environ["GEMINI_API_KEY"] = "your-key"

agent = Agent(model_name="gemini/gemini-pro", max_loops=1)
```

### Ollama (Local)

```python
# No API key needed - ensure Ollama is running
agent = Agent(model_name="ollama/llama2", max_loops=1)
```

## Complete Examples

### Multi-Provider Comparison

```python
from swarms import Agent

models = ["gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini/gemini-pro"]
task = "Explain quantum computing in one paragraph."

for model_name in models:
    print(f"\n=== {model_name} ===")
    agent = Agent(model_name=model_name, max_loops=1)
    response = agent.run(task)
    print(response[:200])
```

### Vision Analysis

```python
from swarms import Agent

agent = Agent(model_name="gpt-4o", max_loops=1)

response = agent.run(
    "Analyze this image and describe what you see.",
    img="https://example.com/image.jpg"
)
print(response)
```

### Streaming Response

```python
from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(model_name="gpt-4o", stream=True)

print("Response: ", end="")
for chunk in llm.run("Write a short poem about AI"):
    print(chunk, end="", flush=True)
```

## Resources

- **LiteLLM Docs**: https://docs.litellm.ai/
- **Providers**: https://docs.litellm.ai/docs/providers
- **Swarms Wrapper**: `swarms/utils/litellm_wrapper.py`
