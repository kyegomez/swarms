# Custom Base URL and API Key Configuration

This guide demonstrates how to configure Swarms agents with custom base URLs and API keys to use models from various providers beyond the default configurations.

The Swarms framework supports over 600+ models through the `model_name` parameter, and you can easily integrate with any model provider by specifying custom `llm_base_url` and `llm_api_key` parameters. This flexibility allows you to use models from:

| Provider                     | Description                                 |
|------------------------------|---------------------------------------------|
| OpenRouter                   | Aggregator for 100+ models from various providers |
| Together AI                  | Multi-model provider with OpenAI-compatible API |
| Replicate                    | API for running open-source models          |
| Hugging Face Inference API   | Access to Hugging Face hosted models        |
| Custom local endpoints       | Your own hosted or local LLM endpoints      |
| Any OpenAI-compatible API    | Any service that implements the OpenAI API spec |

## Basic Configuration

### Import and Setup

```python
import os
from dotenv import load_dotenv
from swarms import Agent

# Load environment variables
load_dotenv()
```

### Basic Custom Base URL Example

```python
# Example using OpenRouter with a specific model
agent = Agent(
    agent_name="Custom-Model-Agent",
    agent_description="Agent using custom base URL and API key",
    model_name="openrouter/qwen/qwen3-vl-235b-a22b-instruct",
    llm_base_url="https://openrouter.ai/api/v1",
    llm_api_key=os.getenv("OPENROUTER_API_KEY"),
    max_loops=1,
    streaming_on=True,
)

# Run the agent
result = agent.run("What are the top 5 energy stocks to invest in?")
print(result)
```

## Provider-Specific Examples

### 1. OpenRouter Integration

OpenRouter provides access to 100+ models from various providers:

```python
# Using Claude via OpenRouter
claude_agent = Agent(
    agent_name="Claude-Agent",
    agent_description="Agent using Claude via OpenRouter",
    model_name="anthropic/claude-3.5-sonnet",
    llm_base_url="https://openrouter.ai/api/v1",
    llm_api_key=os.getenv("OPENROUTER_API_KEY"),
    max_loops=1,
    temperature=0.7,
)

# Using GPT-4 via OpenRouter
gpt4_agent = Agent(
    agent_name="GPT4-Agent",
    agent_description="Agent using GPT-4 via OpenRouter",
    model_name="openai/gpt-4-turbo",
    llm_base_url="https://openrouter.ai/api/v1",
    llm_api_key=os.getenv("OPENROUTER_API_KEY"),
    max_loops=1,
    temperature=0.5,
)

# Using Qwen model via OpenRouter
qwen_agent = Agent(
    agent_name="Qwen-Agent",
    agent_description="Agent using Qwen via OpenRouter",
    model_name="openrouter/qwen/qwen3-vl-235b-a22b-instruct",
    llm_base_url="https://openrouter.ai/api/v1",
    llm_api_key=os.getenv("OPENROUTER_API_KEY"),
    max_loops=1,
    dynamic_temperature_enabled=True,
)
```

### 2. Together AI Integration

```python
# Using Together AI models
together_agent = Agent(
    agent_name="Together-Agent",
    agent_description="Agent using Together AI models",
    model_name="meta-llama/Llama-2-70b-chat-hf",
    llm_base_url="https://api.together.xyz/v1",
    llm_api_key=os.getenv("TOGETHER_API_KEY"),
    max_loops=1,
    temperature=0.7,
)

# Using CodeLlama via Together AI
codellama_agent = Agent(
    agent_name="CodeLlama-Agent",
    agent_description="Agent specialized in code generation",
    model_name="codellama/CodeLlama-34b-Instruct-Python",
    llm_base_url="https://api.together.xyz/v1",
    llm_api_key=os.getenv("TOGETHER_API_KEY"),
    max_loops=1,
    temperature=0.1,  # Lower temperature for code generation
)
```

### 3. Hugging Face Inference API

```python
# Using Hugging Face models
hf_agent = Agent(
    agent_name="HuggingFace-Agent",
    agent_description="Agent using Hugging Face models",
    model_name="microsoft/DialoGPT-large",
    llm_base_url="https://api-inference.huggingface.co/models",
    llm_api_key=os.getenv("HUGGINGFACE_API_KEY"),
    max_loops=1,
    temperature=0.7,
)
```

### 4. Custom Local Endpoint

```python
# Using a local model server (e.g., Ollama, etc.)
local_agent = Agent(
    agent_name="Local-Agent",
    agent_description="Agent using local model endpoint",
    model_name="llama-2-7b-chat",  # Model name as configured in your local server
    llm_base_url="http://localhost:8000/v1",
    llm_api_key="dummy-key",  # May not be required for local endpoints
    max_loops=1,
    temperature=0.7,
)
```

### 5. Replicate Integration

```python
# Using Replicate models
replicate_agent = Agent(
    agent_name="Replicate-Agent",
    agent_description="Agent using Replicate models",
    model_name="meta/llama-2-70b-chat",
    llm_base_url="https://api.replicate.com/v1",
    llm_api_key=os.getenv("REPLICATE_API_TOKEN"),
    max_loops=1,
    temperature=0.7,
)
```

## Advanced Configuration Examples



### Agent with Custom Headers

```python
# For providers that require custom headers
custom_agent = Agent(
    agent_name="Custom-Headers-Agent",
    agent_description="Agent with custom headers",
    model_name="your-custom-model",
    llm_base_url="https://your-custom-api.com/v1",
    llm_api_key=os.getenv("CUSTOM_API_KEY"),
    max_loops=1,
    # Additional headers can be passed through the model initialization
    # This depends on the specific provider's requirements
)
```

## Environment Variables Setup

Create a `.env` file in your project root:

```bash
# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Together AI
TOGETHER_API_KEY=your_together_api_key_here

# Hugging Face
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Replicate
REPLICATE_API_TOKEN=your_replicate_token_here

# Custom API
CUSTOM_API_KEY=your_custom_api_key_here
```

## Complete Working Example

Here's a complete example that demonstrates the custom base URL functionality:

```python
import os
from dotenv import load_dotenv
from swarms import Agent

load_dotenv()


agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="openrouter/qwen/qwen3-vl-235b-a22b-instruct",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=True,
    llm_base_url="https://openrouter.ai/api/v1",
    llm_api_key=os.getenv("OPENROUTER_API_KEY"),
)
return agent

def main():
    
    # Define the trading analysis task
    task = """
    Analyze the current energy sector and provide:
    1. Top 5 energy stocks across nuclear, solar, gas, and other energy sources
    2. Brief analysis of each stock's potential
    3. Market trends affecting the energy sector
    4. Risk assessment for each recommendation
    """
    
    # Run the agent
    print("Running quantitative trading analysis...")
    result = agent.run(task)

    print(result)

if __name__ == "__main__":
    main()
```

## Supported Model Providers

The Swarms framework supports models from these major providers through custom base URLs:

| Provider           | Description                                             |
|--------------------|--------------------------------------------------------|
| **OpenRouter**     | 100+ models from various providers                     |
| **Together AI**    | Open-source models like Llama, CodeLlama, etc.         |
| **Hugging Face**   | Inference API for thousands of models                  |
| **Replicate**      | Various open-source and proprietary models             |
| **Anthropic**      | Claude models (via OpenRouter or direct API)           |
| **Google**         | Gemini models (via OpenRouter)                         |
| **Meta**           | Llama models (via various providers)                   |
| **Custom Endpoints** | Any OpenAI-compatible API                            |

## Best Practices

| Best Practice            | Description                                                        |
|--------------------------|--------------------------------------------------------------------|
| Environment Variables    | Always use environment variables for API keys                      |
| Error Handling           | Implement proper error handling for API failures                   |
| Rate Limiting            | Be aware of rate limits for different providers                    |
| Cost Management          | Monitor usage and costs across different providers                 |
| Model Selection          | Choose models based on your specific use case requirements         |
| Fallback Models          | Implement fallback models for production reliability               |

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API key is correctly set in environment variables
2. **Base URL Format**: Verify the base URL format matches the provider's requirements
3. **Model Name**: Use the exact model name as specified by the provider
4. **Rate Limits**: Implement retry logic for rate limit errors
5. **Network Issues**: Handle network connectivity issues gracefully

### Debug Tips

```python
# Enable debug mode to see detailed logs
agent = Agent(
    # ... other parameters
    verbose=True,  # Enable verbose logging
)
```

## Conclusion

The custom base URL and API key configuration in Swarms provides incredible flexibility to use models from any provider. This approach allows you to:

- Access 600+ models through a unified interface

- Switch between providers easily

- Implement cost-effective solutions

- Use specialized models for specific tasks

- Maintain production reliability with fallback models

Start experimenting with different models and providers to find the best combination for your specific use cases!
