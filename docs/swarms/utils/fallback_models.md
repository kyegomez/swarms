# Fallback Models in Swarms Agent

The Swarms Agent now supports automatic fallback to alternative models when the primary model fails. This feature enhances reliability and ensures your agents can continue operating even when specific models are unavailable or experiencing issues.

## Overview

The fallback model system allows you to specify one or more alternative models that will be automatically tried if the primary model encounters an error. This is particularly useful for:

- **High availability**: Ensure your agents continue working even if a specific model is down
- **Cost optimization**: Use cheaper models as fallbacks for non-critical tasks
- **Rate limiting**: Switch to alternative models when hitting rate limits
- **Model-specific issues**: Handle temporary model-specific problems

## Configuration

### Single Fallback Model

```python
from swarms import Agent

# Configure a single fallback model
agent = Agent(
    model_name="gpt-4.1",  # Primary model
    fallback_model_name="gpt-4o-mini",  # Fallback model
    max_loops=1
)
```

### Multiple Fallback Models

```python
from swarms import Agent

# Configure multiple fallback models using unified list
agent = Agent(
    fallback_models=["gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku"],  # First is primary, rest are fallbacks
    max_loops=1
)
```

### Combined Configuration

```python
from swarms import Agent

# You can use both single fallback and fallback list
agent = Agent(
    model_name="gpt-4.1",  # Primary model
    fallback_model_name="gpt-4o-mini",  # Single fallback
    fallback_models=["gpt-3.5-turbo", "claude-3-haiku"],  # Additional fallbacks
    max_loops=1
)

# Or use the unified list approach (recommended)
agent = Agent(
    fallback_models=["gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku"],
    max_loops=1
)
# Final order: gpt-4o -> gpt-4o-mini -> gpt-3.5-turbo -> claude-3-haiku
```

## How It Works

1. **Primary Model**: The agent starts with the specified primary model
2. **Error Detection**: When an LLM call fails, the system catches the error
3. **Automatic Switching**: The agent automatically switches to the next available model
4. **Retry**: The failed operation is retried with the new model
5. **Exhaustion**: If all models fail, the original error is raised

## API Reference

### Constructor Parameters

- `fallback_model_name` (str, optional): Single fallback model name
- `fallback_models` (List[str], optional): List of fallback model names

### Methods

#### `get_available_models() -> List[str]`
Returns the complete list of available models in order of preference.

```python
models = agent.get_available_models()
print(models)  # ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
```

#### `get_current_model() -> str`
Returns the currently active model name.

```python
current = agent.get_current_model()
print(current)  # 'gpt-4o'
```

#### `is_fallback_available() -> bool`
Checks if fallback models are configured.

```python
has_fallback = agent.is_fallback_available()
print(has_fallback)  # True
```

#### `switch_to_next_model() -> bool`
Manually switch to the next available model. Returns `True` if successful, `False` if no more models available.

```python
success = agent.switch_to_next_model()
if success:
    print(f"Switched to: {agent.get_current_model()}")
else:
    print("No more models available")
```

#### `reset_model_index()`
Reset to the primary model.

```python
agent.reset_model_index()
print(agent.get_current_model())  # Back to primary model
```

## Examples

### Basic Usage

```python
from swarms import Agent

# Create agent with fallback models
agent = Agent(
    model_name="gpt-4.1",
    fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"],
    max_loops=1
)

# Run a task - will automatically use fallbacks if needed
response = agent.run("Write a short story about AI")
print(response)
```

### Monitoring Model Usage

```python
from swarms import Agent

agent = Agent(
    model_name="gpt-4.1",
    fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"],
    max_loops=1
)

print(f"Available models: {agent.get_available_models()}")
print(f"Current model: {agent.get_current_model()}")

# Run task
response = agent.run("Analyze this data")

# Check if fallback was used
if agent.get_current_model() != "gpt-4.1":
    print(f"Used fallback model: {agent.get_current_model()}")
```

### Manual Model Switching

```python
from swarms import Agent

agent = Agent(
    model_name="gpt-4.1",
    fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"],
    max_loops=1
)

# Manually switch models
print(f"Starting with: {agent.get_current_model()}")

agent.switch_to_next_model()
print(f"Switched to: {agent.get_current_model()}")

agent.switch_to_next_model()
print(f"Switched to: {agent.get_current_model()}")

# Reset to primary
agent.reset_model_index()
print(f"Reset to: {agent.get_current_model()}")
```

## Error Handling

The fallback system handles various types of errors:

- **API Errors**: Rate limits, authentication issues
- **Model Errors**: Model-specific failures
- **Network Errors**: Connection timeouts, network issues
- **Configuration Errors**: Invalid model names, unsupported features

### Error Logging

The system provides detailed logging when fallbacks are used:

```
WARNING: Agent 'my-agent' switching to fallback model: gpt-4o-mini (attempt 2/3)
INFO: Retrying with fallback model 'gpt-4o-mini' for agent 'my-agent'
```

## Best Practices

### 1. Model Selection
- Choose fallback models that are compatible with your use case
- Consider cost differences between models
- Ensure fallback models support the same features (e.g., function calling, vision)

### 2. Order Matters
- Place most reliable models first
- Consider cost-performance trade-offs
- Test fallback models to ensure they work for your tasks

### 3. Monitoring
- Monitor which models are being used
- Track fallback usage patterns
- Set up alerts for excessive fallback usage

### 4. Error Handling
- Implement proper error handling in your application
- Consider graceful degradation when all models fail
- Log fallback usage for analysis

## Limitations

1. **Model Compatibility**: Fallback models must be compatible with your use case
2. **Feature Support**: Not all models support the same features (e.g., function calling, vision)
3. **Cost Implications**: Different models have different pricing
4. **Performance**: Fallback models may have different performance characteristics

## Troubleshooting

### Common Issues

1. **All Models Failing**: Check API keys and network connectivity
2. **Feature Incompatibility**: Ensure fallback models support required features
3. **Rate Limiting**: Consider adding delays between model switches
4. **Configuration Errors**: Verify model names are correct

### Debug Mode

Enable verbose logging to see detailed fallback information:

```python
agent = Agent(
    model_name="gpt-4.1",
    fallback_models=["gpt-4o-mini"],
    verbose=True  # Enable detailed logging
)
```

## Migration Guide

### From No Fallback to Fallback

If you're upgrading from an agent without fallback support:

```python
# Before
agent = Agent(model_name="gpt-4.1")

# After
agent = Agent(
    model_name="gpt-4.1",
    fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"]
)
```

### Backward Compatibility

The fallback system is fully backward compatible. Existing agents will continue to work without any changes.

## Conclusion

The fallback model system provides a robust way to ensure your Swarms agents remain operational even when individual models fail. By configuring appropriate fallback models, you can improve reliability, handle rate limits, and optimize costs while maintaining the same simple API.
