# Model Providers Overview

Swarms supports a vast array of model providers, giving you the flexibility to choose the best model for your specific use case. Whether you need high-performance inference, cost-effective solutions, or specialized capabilities, Swarms has you covered.

## Supported Model Providers

| Provider | Description | Documentation |
|----------|-------------|---------------|
| **OpenAI** | Industry-leading language models including GPT-4, GPT-4o, and GPT-4o-mini. Perfect for general-purpose tasks, creative writing, and complex reasoning. | [OpenAI Integration](openai_example.md) |
| **Anthropic/Claude** | Advanced AI models known for their safety, helpfulness, and reasoning capabilities. Claude models excel at analysis, coding, and creative tasks. | [Claude Integration](claude.md) |
| **Groq** | Ultra-fast inference platform offering real-time AI responses. Ideal for applications requiring low latency and high throughput. | [Groq Integration](groq.md) |
| **Cohere** | Enterprise-grade language models with strong performance on business applications, text generation, and semantic search. | [Cohere Integration](cohere.md) |
| **DeepSeek** | Advanced reasoning models including the DeepSeek Reasoner (R1). Excellent for complex problem-solving and analytical tasks. | [DeepSeek Integration](deepseek.md) |
| **Ollama** | Local model deployment platform allowing you to run open-source models on your own infrastructure. No API keys required. | [Ollama Integration](ollama.md) |
| **OpenRouter** | Unified API gateway providing access to hundreds of models from various providers through a single interface. | [OpenRouter Integration](openrouter.md) |
| **XAI** | xAI's Grok models offering unique capabilities for research, analysis, and creative tasks with advanced reasoning abilities. | [XAI Integration](xai.md) |
| **vLLM** | High-performance inference library for serving large language models with optimized memory usage and throughput. | [vLLM Integration](vllm_integration.md) |
| **Llama4** | Meta's latest open-source language models including Llama-4-Maverick and Llama-4-Scout variants with expert routing capabilities. | [Llama4 Integration](llama4.md) |
| **Azure OpenAI** | Enterprise-grade OpenAI models through Microsoft's cloud infrastructure with enhanced security, compliance, and enterprise features. | [Azure Integration](azure.md) |

## Quick Start

All model providers follow a consistent pattern in Swarms. Here's the basic template:

```python
from swarms import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize agent with your chosen model
agent = Agent(
    agent_name="Your-Agent-Name",
    model_name="gpt-4o-mini",  # Varies by provider
    system_prompt="Your system prompt here",
    agent_description="Description of what your agent does.",
)

# Run your agent
response = agent.run("Your query here")
```

## Model Selection Guide

### For High-Performance Applications

- **OpenAI GPT-4o**: Best overall performance and reasoning

- **Anthropic Claude**: Excellent safety and analysis capabilities

- **DeepSeek R1**: Advanced reasoning and problem-solving

### For Cost-Effective Solutions

- **OpenAI GPT-4o-mini**: Great performance at lower cost

- **Ollama**: Free local deployment

- **OpenRouter**: Access to cost-effective models

### For Real-Time Applications

- **Groq**: Ultra-fast inference

- **vLLM**: Optimized for high throughput

### For Specialized Tasks

- **Llama4**: Expert routing for complex workflows

- **XAI Grok**: Advanced research capabilities

- **Cohere**: Strong business applications

## Environment Setup

Most providers require API keys. Add them to your `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Groq
GROQ_API_KEY=your_groq_key

# Cohere
COHERE_API_KEY=your_cohere_key

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_key

# XAI
XAI_API_KEY=your_xai_key

# Azure OpenAI
AZURE_API_KEY=your_azure_openai_api_key
AZURE_API_BASE=https://your-resource-name.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
```

!!! note "No API Key Required"
    Ollama and vLLM can be run locally without API keys, making them perfect for development and testing.

## Advanced Features

### Multi-Model Workflows

Swarms allows you to create workflows that use different models for different tasks:

```python
from swarms import Agent, ConcurrentWorkflow

# Research agent using Claude for analysis
research_agent = Agent(
    agent_name="Research-Agent",
    model_name="claude-3-sonnet-20240229",
    system_prompt="You are a research expert."
)

# Creative agent using GPT-4o for content generation
creative_agent = Agent(
    agent_name="Creative-Agent", 
    model_name="gpt-4.1",
    system_prompt="You are a creative content expert."
)

# Workflow combining both agents
workflow = ConcurrentWorkflow(
    name="Research-Creative-Workflow",
    agents=[research_agent, creative_agent]
)
```

### Model Routing

Automatically route tasks to the most appropriate model:

```python
from swarms import Agent, ModelRouter

# Define model preferences for different task types
model_router = ModelRouter(
    models={
        "analysis": "claude-3-sonnet-20240229",
        "creative": "gpt-4.1", 
        "fast": "gpt-4o-mini",
        "local": "ollama/llama2"
    }
)

# Agent will automatically choose the best model
agent = Agent(
    agent_name="Smart-Agent",
    llm=model_router,
    system_prompt="You are a versatile assistant."
)
```

## Getting Help

- **Documentation**: Each provider has detailed documentation with examples

- **Community**: Join the Swarms community for support and best practices

- **Issues**: Report bugs and request features on GitHub

- **Discussions**: Share your use cases and learn from others

!!! success "Ready to Get Started?"
    Choose a model provider from the table above and follow the detailed integration guide. Each provider offers unique capabilities that can enhance your Swarms applications.
