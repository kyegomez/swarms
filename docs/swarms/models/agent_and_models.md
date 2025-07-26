# Model Integration in Agents

!!! info "About Model Integration"
    Agents supports multiple model providers through LiteLLM integration, allowing you to easily switch between different language models. This document outlines the available providers and how to use them with agents.

## Important Note on Model Names

!!! warning "Required Format"
    When specifying a model in an agent, you must use the format `provider/model_name`. For example:
    ```python
    "openai/gpt-4"
    "anthropic/claude-3-opus-latest"
    "cohere/command-r-plus"
    ```
    This format ensures the agent knows which provider to use for the specified model.

## Available Model Providers

### OpenAI

??? info "OpenAI Models"
    - **Provider name**: `openai`
    - **Available Models**:
        - `gpt-4`
        - `gpt-3.5-turbo`
        - `gpt-4-turbo-preview`

### Anthropic
??? info "Anthropic Models"
    - **Provider name**: `anthropic`
    - **Available Models**:
        - **Claude 3 Opus**:
            - `claude-3-opus-latest`
            - `claude-3-opus-20240229`
        - **Claude 3 Sonnet**:
            - `claude-3-sonnet-20240229`
            - `claude-3-5-sonnet-latest`
            - `claude-3-5-sonnet-20240620`
            - `claude-3-7-sonnet-latest`
            - `claude-3-7-sonnet-20250219`
            - `claude-3-5-sonnet-20241022`
        - **Claude 3 Haiku**:
            - `claude-3-haiku-20240307`
            - `claude-3-5-haiku-20241022`
            - `claude-3-5-haiku-latest`
        - **Legacy Models**:
            - `claude-2`
            - `claude-2.1`
            - `claude-instant-1`
            - `claude-instant-1.2`

### Cohere
??? info "Cohere Models"
    - **Provider name**: `cohere`
    - **Available Models**:
        - **Command**:
            - `command`
            - `command-r`
            - `command-r-08-2024`
            - `command-r7b-12-2024`
        - **Command Light**:
            - `command-light`
        - **Command R Plus**:
            - `command-r-plus`
            - `command-r-plus-08-2024`

### Google
??? info "Google Models"
    - **Provider name**: `google`
    - **Available Models**:
        - `gemini-pro`
        - `gemini-pro-vision`

### Mistral
??? info "Mistral Models"
    - **Provider name**: `mistral`
    - **Available Models**:
        - `mistral-tiny`
        - `mistral-small`
        - `mistral-medium`

## Using Different Models In Your Agents

To use a different model with your Swarms agent, specify the model name in the `model_name` parameter when initializing the Agent, using the provider/model_name format:

```python
from swarms import Agent

# Using OpenAI's GPT-4
agent = Agent(
    agent_name="Research-Agent",
    model_name="openai/gpt-4o",  # Note the provider/model_name format
    # ... other parameters
)

# Using Anthropic's Claude
agent = Agent(
    agent_name="Analysis-Agent",
    model_name="anthropic/claude-3-sonnet-20240229",  # Note the provider/model_name format
    # ... other parameters
)

# Using Cohere's Command
agent = Agent(
    agent_name="Text-Agent",
    model_name="cohere/command-r-plus",  # Note the provider/model_name format
    # ... other parameters
)
```

## Model Configuration

When using different models, you can configure various parameters:

```python
agent = Agent(
    agent_name="Custom-Agent",
    model_name="openai/gpt-4",
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=2000,  # Maximum tokens in response
    top_p=0.9,       # Nucleus sampling parameter
    frequency_penalty=0.0,  # Reduces repetition
    presence_penalty=0.0,   # Encourages new topics
    # ... other parameters
)
```

## Best Practices

### Model Selection
!!! tip "Choosing the Right Model"
    - Choose models based on your specific use case
    - Consider cost, performance, and feature requirements
    - Test different models for your specific task

### Error Handling
!!! warning "Error Management"
    - Implement proper error handling for model-specific errors
    - Handle rate limits and API quotas appropriately

### Cost Management
!!! note "Cost Considerations"
    - Monitor token usage and costs
    - Use appropriate model sizes for your needs

## Example Use Cases

### 1. Complex Analysis (GPT-4)

```python
agent = Agent(
    agent_name="Analysis-Agent",
    model_name="openai/gpt-4",  # Note the provider/model_name format
    temperature=0.3,  # Lower temperature for more focused responses
    max_tokens=4000
)
```

### 2. Creative Tasks (Claude)

```python
agent = Agent(
    agent_name="Creative-Agent",
    model_name="anthropic/claude-3-sonnet-20240229",  # Note the provider/model_name format
    temperature=0.8,  # Higher temperature for more creative responses
    max_tokens=2000
)
```

### 3. Vision Tasks (Gemini)

```python
agent = Agent(
    agent_name="Vision-Agent",
    model_name="google/gemini-pro-vision",  # Note the provider/model_name format
    temperature=0.4,
    max_tokens=1000
)
```

## Troubleshooting

!!! warning "Common Issues"
    If you encounter issues with specific models:

    1. Verify your API keys are correctly set
    2. Check model availability in your region
    3. Ensure you have sufficient quota/credits
    4. Verify the model name is correct and supported

## Additional Resources

- [LiteLLM Documentation](https://docs.litellm.ai/){target=_blank}
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference){target=_blank}
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api){target=_blank}
- [Google AI Documentation](https://ai.google.dev/docs){target=_blank}
