# Azure OpenAI Integration

This guide demonstrates how to integrate Azure OpenAI models with Swarms for enterprise-grade AI applications. Azure OpenAI provides access to OpenAI models through Microsoft's cloud infrastructure with enhanced security, compliance, and enterprise features.

## Prerequisites

- Azure subscription with OpenAI service enabled
- Azure OpenAI resource deployed
- Python 3.7+
- Swarms library
- LiteLLM library

## Installation

First, install the required dependencies:

```bash
pip install -U swarms
```

## Environment Setup

### 1. Azure OpenAI Configuration

Set up your Azure OpenAI environment variables in a `.env` file:

```bash
# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_openai_api_key
AZURE_API_BASE=https://your-resource-name.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview

# Optional: Model deployment names (if different from model names)
AZURE_GPT4_DEPLOYMENT_NAME=gpt-4
AZURE_GPT35_DEPLOYMENT_NAME=gpt-35-turbo
```

### 2. Verify Available Models

Check what Azure models are available using LiteLLM:

```python
from litellm import model_list

# List all available Azure models
print("Available Azure models:")
for model in model_list:
    if "azure" in model:
        print(f"  - {model}")
```

Common Azure model names include:

- `azure/gpt-4.1`

- `azure/gpt-4o`

- `azure/gpt-4o-mini`



## Models Supported

```txt
azure_ai/grok-3
azure_ai/global/grok-3
azure_ai/global/grok-3-mini
azure_ai/grok-3-mini
azure_ai/deepseek-r1
azure_ai/deepseek-v3
azure_ai/deepseek-v3-0324
azure_ai/jamba-instruct
azure_ai/jais-30b-chat
azure_ai/mistral-nemo
azure_ai/mistral-medium-2505
azure_ai/mistral-large
azure_ai/mistral-small
azure_ai/mistral-small-2503
azure_ai/mistral-large-2407
azure_ai/mistral-large-latest
azure_ai/ministral-3b
azure_ai/Llama-3.2-11B-Vision-Instruct
azure_ai/Llama-3.3-70B-Instruct
azure_ai/Llama-4-Scout-17B-16E-Instruct
azure_ai/Llama-4-Maverick-17B-128E-Instruct-FP8
azure_ai/Llama-3.2-90B-Vision-Instruct
azure_ai/Meta-Llama-3-70B-Instruct
azure_ai/Meta-Llama-3.1-8B-Instruct
azure_ai/Meta-Llama-3.1-70B-Instruct
azure_ai/Meta-Llama-3.1-405B-Instruct
azure_ai/Phi-4-mini-instruct
azure_ai/Phi-4-multimodal-instruct
azure_ai/Phi-4
azure_ai/Phi-3.5-mini-instruct
azure_ai/Phi-3.5-vision-instruct
azure_ai/Phi-3.5-MoE-instruct
azure_ai/Phi-3-mini-4k-instruct
azure_ai/Phi-3-mini-128k-instruct
azure_ai/Phi-3-small-8k-instruct
azure_ai/Phi-3-small-128k-instruct
azure_ai/Phi-3-medium-4k-instruct
azure_ai/Phi-3-medium-128k-instruct
azure_ai/cohere-rerank-v3.5
azure_ai/cohere-rerank-v3-multilingual
azure_ai/cohere-rerank-v3-english
azure_ai/Cohere-embed-v3-english
azure_ai/Cohere-embed-v3-multilingual
azure_ai/embed-v-4-0
azure/gpt-4o-mini-tts
azure/computer-use-preview
azure/gpt-4o-audio-preview-2024-12-17
azure/gpt-4o-mini-audio-preview-2024-12-17
azure/gpt-4.1
azure/gpt-4.1-2025-04-14
azure/gpt-4.1-mini
azure/gpt-4.1-mini-2025-04-14
azure/gpt-4.1-nano
azure/gpt-4.1-nano-2025-04-14
azure/o3-pro
azure/o3-pro-2025-06-10
azure/o3
azure/o3-2025-04-16
azure/o3-deep-research
azure/o4-mini
azure/gpt-4o-mini-realtime-preview-2024-12-17
azure/eu/gpt-4o-mini-realtime-preview-2024-12-17
azure/us/gpt-4o-mini-realtime-preview-2024-12-17
azure/gpt-4o-realtime-preview-2024-12-17
azure/us/gpt-4o-realtime-preview-2024-12-17
azure/eu/gpt-4o-realtime-preview-2024-12-17
azure/gpt-4o-realtime-preview-2024-10-01
azure/us/gpt-4o-realtime-preview-2024-10-01
azure/eu/gpt-4o-realtime-preview-2024-10-01
azure/o4-mini-2025-04-16
azure/o3-mini-2025-01-31
azure/us/o3-mini-2025-01-31
azure/eu/o3-mini-2025-01-31
azure/tts-1
azure/tts-1-hd
azure/whisper-1
azure/gpt-4o-transcribe
azure/gpt-4o-mini-transcribe
azure/o3-mini
azure/o1-mini
azure/o1-mini-2024-09-12
azure/us/o1-mini-2024-09-12
azure/eu/o1-mini-2024-09-12
azure/o1
azure/o1-2024-12-17
azure/us/o1-2024-12-17
azure/eu/o1-2024-12-17
azure/codex-mini
azure/o1-preview
azure/o1-preview-2024-09-12
azure/us/o1-preview-2024-09-12
azure/eu/o1-preview-2024-09-12
azure/gpt-4.5-preview
azure/gpt-4o
azure/global/gpt-4o-2024-11-20
azure/gpt-4o-2024-08-06
azure/global/gpt-4o-2024-08-06
azure/gpt-4o-2024-11-20
azure/us/gpt-4o-2024-11-20
azure/eu/gpt-4o-2024-11-20
azure/gpt-4o-2024-05-13
azure/global-standard/gpt-4o-2024-08-06
azure/us/gpt-4o-2024-08-06
azure/eu/gpt-4o-2024-08-06
azure/global-standard/gpt-4o-2024-11-20
azure/global-standard/gpt-4o-mini
azure/gpt-4o-mini
azure/gpt-4o-mini-2024-07-18
azure/us/gpt-4o-mini-2024-07-18
azure/eu/gpt-4o-mini-2024-07-18
azure/gpt-4-turbo-2024-04-09
azure/gpt-4-0125-preview
azure/gpt-4-1106-preview
azure/gpt-4-0613
azure/gpt-4-32k-0613
azure/gpt-4-32k
azure/gpt-4
azure/gpt-4-turbo
azure/gpt-4-turbo-vision-preview
azure/gpt-35-turbo-16k-0613
azure/gpt-35-turbo-1106
azure/gpt-35-turbo-0613
azure/gpt-35-turbo-0301
azure/gpt-35-turbo-0125
azure/gpt-3.5-turbo-0125
azure/gpt-35-turbo-16k
azure/gpt-35-turbo
azure/gpt-3.5-turbo
azure/mistral-large-latest
azure/mistral-large-2402
azure/command-r-plus
azure/ada
azure/text-embedding-ada-002
azure/text-embedding-3-large
azure/text-embedding-3-small
azure/gpt-image-1
azure/low/1024-x-1024/gpt-image-1
azure/medium/1024-x-1024/gpt-image-1
azure/high/1024-x-1024/gpt-image-1
azure/low/1024-x-1536/gpt-image-1
azure/medium/1024-x-1536/gpt-image-1
azure/high/1024-x-1536/gpt-image-1
azure/low/1536-x-1024/gpt-image-1
azure/medium/1536-x-1024/gpt-image-1
azure/high/1536-x-1024/gpt-image-1
azure/standard/1024-x-1024/dall-e-3
azure/hd/1024-x-1024/dall-e-3
azure/standard/1024-x-1792/dall-e-3
azure/standard/1792-x-1024/dall-e-3
azure/hd/1024-x-1792/dall-e-3
azure/hd/1792-x-1024/dall-e-3
azure/standard/1024-x-1024/dall-e-2
azure/gpt-3.5-turbo-instruct-0914
azure/gpt-35-turbo-instruct
azure/gpt-35-turbo-instruct-0914
```


## Basic Usage

### Simple Agent with Azure Model

```python
import os
from dotenv import load_dotenv
from swarms import Agent

# Load environment variables
load_dotenv()

# Initialize agent with Azure model
agent = Agent(
    agent_name="Azure-Agent",
    agent_description="An agent powered by Azure OpenAI",
    system_prompt="You are a helpful assistant powered by Azure OpenAI.",
    model_name="azure/gpt-4o-mini",
    max_loops=1,
    max_tokens=1000,
    dynamic_temperature_enabled=True,
    output_type="str",
)

# Run the agent
response = agent.run("Explain quantum computing in simple terms.")
print(response)
```

## Advanced Configuration

### Quantitative Trading Agent Example

Here's a comprehensive example of a quantitative trading agent using Azure models:

```python
import os
from dotenv import load_dotenv
from swarms import Agent

# Load environment variables
load_dotenv()

# Initialize the quantitative trading agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent powered by Azure OpenAI",
    system_prompt="""You are an expert quantitative trading agent with deep expertise in:
    - Algorithmic trading strategies and implementation
    - Statistical arbitrage and market making
    - Risk management and portfolio optimization
    - High-frequency trading systems
    - Market microstructure analysis
    - Quantitative research methodologies
    - Financial mathematics and stochastic processes
    - Machine learning applications in trading
    
    Your core responsibilities include:
    1. Developing and backtesting trading strategies
    2. Analyzing market data and identifying alpha opportunities
    3. Implementing risk management frameworks
    4. Optimizing portfolio allocations
    5. Conducting quantitative research
    6. Monitoring market microstructure
    7. Evaluating trading system performance
    
    You maintain strict adherence to:
    - Mathematical rigor in all analyses
    - Statistical significance in strategy development
    - Risk-adjusted return optimization
    - Market impact minimization
    - Regulatory compliance
    - Transaction cost analysis
    - Performance attribution
    
    You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
    model_name="azure/gpt-4o",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_loops="auto",
    interactive=True,
    no_reasoning_prompt=True,
    streaming_on=True,
    max_tokens=4096,
)

# Example usage
response = agent.run(
    task="What are the best top 3 ETFs for gold coverage? Provide detailed analysis including expense ratios, liquidity, and tracking error."
)
print(response)
```


## Next Steps

- Check out [LiteLLM Azure integration](https://docs.litellm.ai/docs/providers/azure)

- Learn about [Swarms multi-agent architectures](../structs/index.md)

- Discover [advanced tool integrations](agent_with_tools.md)
