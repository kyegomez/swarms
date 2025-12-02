# Basic Examples Overview

Start your Swarms journey with single-agent examples. Learn how to create agents, use tools, process images, integrate with different LLM providers, and publish to the marketplace.

## What You'll Learn

| Topic | Description |
|-------|-------------|
| **Agent Basics** | Create and configure individual agents |
| **Tool Integration** | Equip agents with callable tools and functions |
| **Vision Capabilities** | Process images and multi-modal inputs |
| **LLM Providers** | Connect to OpenAI, Anthropic, Groq, and more |
| **Utilities** | Streaming, output types, and marketplace publishing |

---

## Individual Agent Examples

### Core Agent Usage

| Example | Description | Link |
|---------|-------------|------|
| **Basic Agent** | Fundamental agent creation and execution | [View Example](../swarms/examples/basic_agent.md) |

### Tool Usage

| Example | Description | Link |
|---------|-------------|------|
| **Agents with Vision and Tool Usage** | Combine vision and tools in one agent | [View Example](../swarms/examples/vision_tools.md) |
| **Agents with Callable Tools** | Equip agents with Python functions as tools | [View Example](../swarms/examples/agent_with_tools.md) |
| **Agent with Structured Outputs** | Get consistent JSON/structured responses | [View Example](../swarms/examples/agent_structured_outputs.md) |
| **Message Transforms** | Manage context with message transformations | [View Example](../swarms/structs/transforms.md) |

### Vision & Multi-Modal

| Example | Description | Link |
|---------|-------------|------|
| **Agents with Vision** | Process and analyze images | [View Example](../swarms/examples/vision_processing.md) |
| **Agent with Multiple Images** | Handle multiple images in one request | [View Example](../swarms/examples/multiple_images.md) |

### Utilities

| Example | Description | Link |
|---------|-------------|------|
| **Agent with Streaming** | Stream responses in real-time | [View Example](./agent_stream.md) |
| **Agent Output Types** | Different output formats (str, json, dict, yaml) | [View Example](../swarms/examples/agent_output_types.md) |
| **Gradio Chat Interface** | Build chat UIs for your agents | [View Example](../swarms/ui/main.md) |
| **Agent with Gemini Nano Banana** | Jarvis-style agent example | [View Example](../swarms/examples/jarvis_agent.md) |
| **Agent Marketplace Publishing** | Publish agents to the Swarms marketplace | [View Example](./marketplace_publishing_quickstart.md) |

---

## LLM Provider Examples

Connect your agents to various language model providers:

| Provider | Description | Link |
|----------|-------------|------|
| **Overview** | Guide to all supported providers | [View Guide](../swarms/examples/model_providers.md) |
| **OpenAI** | GPT-4, GPT-4o, GPT-4o-mini integration | [View Example](../swarms/examples/openai_example.md) |
| **Anthropic** | Claude models integration | [View Example](../swarms/examples/claude.md) |
| **Groq** | Ultra-fast inference with Groq | [View Example](../swarms/examples/groq.md) |
| **Cohere** | Cohere Command models | [View Example](../swarms/examples/cohere.md) |
| **DeepSeek** | DeepSeek models integration | [View Example](../swarms/examples/deepseek.md) |
| **Ollama** | Local models with Ollama | [View Example](../swarms/examples/ollama.md) |
| **OpenRouter** | Access multiple providers via OpenRouter | [View Example](../swarms/examples/openrouter.md) |
| **XAI** | Grok models from xAI | [View Example](../swarms/examples/xai.md) |
| **Azure OpenAI** | Enterprise Azure deployment | [View Example](../swarms/examples/azure.md) |
| **Llama4** | Meta's Llama 4 models | [View Example](../swarms/examples/llama4.md) |
| **Custom Base URL** | Connect to any OpenAI-compatible API | [View Example](../swarms/examples/custom_base_url_example.md) |

---

## Next Steps

After mastering basic agents, explore:

- [Multi-Agent Architectures](./multi_agent_architectures_overview.md) - Coordinate multiple agents
- [Tools Documentation](../swarms/tools/main.md) - Deep dive into tool creation
- [CLI Guides](./cli_guides_overview.md) - Run agents from command line
