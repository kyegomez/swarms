# Loading Prompts from the Swarms Marketplace

Load production-ready prompts from the Swarms Marketplace directly into your agents with a single parameter. This feature enables one-line prompt loading, making it easy to leverage community-created prompts without manual copy-pasting.

## Overview

The Swarms Marketplace hosts a collection of expertly crafted prompts for various use cases. Instead of manually copying prompts or managing them in separate files, you can now load them directly into your agent using the `marketplace_prompt_id` parameter.

## Prerequisites

Before using this feature, ensure you have:

1. **Swarms installed**:
```bash
pip install -U swarms
```

2. **A Swarms API key** - Get yours at [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)

3. **Set your API key** as an environment variable:
```bash
export SWARMS_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Usage

Load a marketplace prompt in one line by providing the `marketplace_prompt_id`:

```python
from swarms import Agent

agent = Agent(
    model_name="gpt-4o-mini",
    marketplace_prompt_id="your-prompt-uuid-here",
    max_loops=1,
)

response = agent.run("Your task here")
print(response)
```

That's it! The agent automatically fetches the prompt from the marketplace and uses it as the system prompt.

### Finding Prompt IDs

To find prompt IDs:

1. Visit the [Swarms Marketplace](https://swarms.world/marketplace)
2. Browse or search for prompts that fit your use case
3. Click on a prompt to view its details
4. Copy the prompt's UUID from the URL or the prompt details page

## Complete Example

Here's a complete working example:

```python
from swarms import Agent

# Create an agent with a marketplace prompt
agent = Agent(
    model_name="gpt-4o-mini",
    marketplace_prompt_id="0ff9cc2f-390a-4eb1-9d3d-3a045cd2682e",
    max_loops="auto",
    interactive=True,
)

# Run the agent - it uses the system prompt from the marketplace
response = agent.run("Hello, what can you help me with?")
print(response)
```

## How It Works

When you provide a `marketplace_prompt_id`, the agent:

1. **Fetches the prompt** from the Swarms Marketplace API during initialization
2. **Sets the system prompt** from the marketplace data
3. **Optionally updates agent metadata** - If you haven't set a custom `agent_name` or `agent_description`, these will be populated from the marketplace prompt data
4. **Logs the operation** - You'll see a confirmation message when the prompt is loaded successfully

```
🛒 [MARKETPLACE] Loaded prompt 'Your Prompt Name' from Swarms Marketplace
```

## Configuration Options

### Combining with Other Parameters

You can combine `marketplace_prompt_id` with any other agent parameters:

```python
from swarms import Agent

agent = Agent(
    # Marketplace prompt
    marketplace_prompt_id="your-prompt-uuid",
    
    # Model configuration
    model_name="gpt-4o",
    max_tokens=4096,
    temperature=0.7,
    
    # Agent behavior
    max_loops=3,
    verbose=True,
    
    # Tools
    tools=[your_tool_function],
)
```

### Overriding Agent Name and Description

By default, the agent will use the name and description from the marketplace prompt if you haven't set them. To use your own:

```python
agent = Agent(
    marketplace_prompt_id="your-prompt-uuid",
    agent_name="My Custom Agent Name",  # This overrides the marketplace name
    agent_description="My custom description",  # This overrides the marketplace description
    model_name="gpt-4o-mini",
)
```

## Error Handling

The feature includes built-in error handling:

### Prompt Not Found

If the prompt ID doesn't exist:

```python
# This will raise a ValueError with a helpful message
agent = Agent(
    marketplace_prompt_id="non-existent-id",
    model_name="gpt-4o-mini",
)
# ValueError: Prompt with ID 'non-existent-id' not found in the marketplace.
# Please verify the prompt ID is correct.
```

### Missing API Key

If the `SWARMS_API_KEY` environment variable is not set:

```python
# This will raise a ValueError
agent = Agent(
    marketplace_prompt_id="your-prompt-uuid",
    model_name="gpt-4o-mini",
)
# ValueError: Swarms API key is not set. Please set the SWARMS_API_KEY environment variable.
# You can get your key here: https://swarms.world/platform/api-keys
```


## Best Practices

1. **Store prompt IDs in configuration** - Keep your prompt IDs in environment variables or config files for easy updates

2. **Handle errors gracefully** - Wrap agent creation in try-except blocks for production code

3. **Cache prompts for offline use** - If you need offline capability, fetch and store prompts locally as backup

4. **Version your prompts** - When updating marketplace prompts, consider creating new versions rather than overwriting

5. **Monitor prompt usage** - Track which prompts are being used in your applications for analytics

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValueError: Swarms API key is not set` | Set the `SWARMS_API_KEY` environment variable |
| `ValueError: Prompt not found` | Verify the prompt ID is correct on the marketplace |
| `Connection timeout` | Check your internet connection and try again |
| Agent not using expected prompt | Ensure you're not also setting `system_prompt` parameter |

## Related Resources

- [Swarms Marketplace](https://swarms.world/marketplace) - Browse available prompts
- [Publishing Prompts](../../../swarms_platform/monetize.md) - Share your own prompts
- [Agent Reference](../../structs/agent.md) - Full agent documentation
- [API Key Management](../../../swarms_platform/apikeys.md) - Manage your API keys

