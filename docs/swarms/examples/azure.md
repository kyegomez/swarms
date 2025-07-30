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
- `azure/gpt-4`
- `azure/gpt-4o`
- `azure/gpt-4o-mini`
- `azure/gpt-35-turbo`
- `azure/gpt-35-turbo-16k`

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
