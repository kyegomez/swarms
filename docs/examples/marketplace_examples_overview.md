# Marketplace Integration Overview

Integrate with the Swarms Marketplace to discover, load, and share production-ready prompts, agents, and tools. The marketplace enables seamless integration between your code and the Swarms community ecosystem.

## What You'll Learn

| Topic | Description |
|-------|-------------|
| **Loading Prompts** | Fetch and use prompts from the marketplace with one line of code |
| **Publishing Agents** | Share your agents with the community and monetize your creations |
| **Marketplace Discovery** | Browse and discover community-created prompts, agents, and tools |
| **API Integration** | Programmatically interact with the marketplace |

---

## Marketplace Integration

The Swarms Marketplace (`https://swarms.world`) is a community hub where developers share and discover:

- **🤖 Agents**: Ready-to-use agents for specific tasks and industries
- **💡 Prompts**: Production-ready system prompts for various use cases
- **🛠️ Tools**: APIs, integrations, and utilities that extend agent capabilities

### Key Features

| Feature | Description |
|---------|-------------|
| **One-Line Prompt Loading** | Load marketplace prompts directly into agents using `marketplace_prompt_id` |
| **Direct Publishing** | Publish agents to the marketplace with minimal configuration |
| **Automatic Integration** | Seamlessly integrates with marketplace API |
| **Monetization Ready** | Set pricing for your shared agents and prompts |
| **Community Discovery** | Browse and discover community-created resources |

---

## Marketplace Examples

### Loading Prompts from Marketplace

| Example | Description | Link |
|---------|-------------|------|
| **Loading Prompts** | Load production-ready prompts from the marketplace into your agents | [View Tutorial](../swarms/examples/marketplace_prompt_loading.md) |

**Quick Example:**
```python
from swarms import Agent

# Load a prompt from the marketplace
# The prompt ID is found in the URL: https://swarms.world/prompt/{prompt-id}
agent = Agent(
    model_name="gpt-4o-mini",
    marketplace_prompt_id="75fc0d28-b0d0-4372-bc04-824aa388b7d2",  # From URL or metadata section
    max_loops=1,
)

response = agent.run("Your task here")
```

**Finding Prompt IDs:** The prompt ID is the UUID found in the marketplace URL (e.g., `https://swarms.world/prompt/75fc0d28-b0d0-4372-bc04-824aa388b7d2`) or in the Metadata section of the prompt listing page.

### Publishing to Marketplace

| Example | Description | Link |
|---------|-------------|------|
| **Agent Publishing** | Publish your agents to the marketplace for community use | [View Tutorial](./marketplace_publishing_quickstart.md) |

**Quick Example:**
```python
from swarms import Agent

# Create and publish an agent
agent = Agent(
    agent_name="My-Specialized-Agent",
    agent_description="Expert agent for specific tasks",
    model_name="gpt-4o-mini",
    publish_to_marketplace=True,  # Enable publishing
    # ... additional configuration
)
```

---

## Prerequisites

Before using marketplace features, ensure you have:

1. **Swarms installed**:
```bash
pip install -U swarms
```

2. **A Swarms API key** - Get yours at [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)

3. **Set your API key** as an environment variable:
```bash
export SWARMS_API_KEY="your-api-key-here"
```

---

## How It Works

### Loading Prompts

When you provide a `marketplace_prompt_id` to an agent:

1. **Fetches the prompt** from the Swarms Marketplace API during initialization
2. **Sets the system prompt** from the marketplace data
3. **Optionally updates agent metadata** - Agent name and description are populated from marketplace data if not set
4. **Logs the operation** - Confirmation message when prompt is loaded successfully

### Publishing Agents

When you publish an agent to the marketplace:

1. **Validates configuration** - Ensures required fields are present
2. **Uploads to marketplace** - Sends agent configuration to marketplace API
3. **Generates marketplace listing** - Creates a discoverable listing on swarms.world
4. **Enables monetization** - Optional pricing configuration for your agent

---

## Use Cases

### For Consumers

- **Rapid Prototyping**: Quickly test different prompts without manual copy-pasting
- **Best Practices**: Use community-validated prompts for production systems
- **Discovery**: Find specialized agents for specific industries or tasks
- **Learning**: Study how others structure their agents and prompts

### For Publishers

- **Community Contribution**: Share your expertise with the Swarms community
- **Monetization**: Earn revenue from your agent creations
- **Visibility**: Get your agents discovered by developers worldwide
- **Collaboration**: Build on top of community-created resources

---

## Related Resources

- [Swarms Marketplace Platform](../swarms_platform/index.md) - Marketplace overview and features
- [Share and Discover](../swarms_platform/share_and_discover.md) - Marketplace browsing guide
- [Monetization Guide](../swarms_platform/monetize.md) - How to monetize your agents
- [API Key Management](../swarms_platform/apikeys.md) - Managing your API keys
- [Agent Reference](../swarms/structs/agent.md) - Full agent documentation

---

## Next Steps

1. **Get Started**: [Load your first marketplace prompt](../swarms/examples/marketplace_prompt_loading.md)
2. **Publish**: [Share your agent with the community](./marketplace_publishing_quickstart.md)
3. **Explore**: [Browse the marketplace](https://swarms.world/marketplace)
4. **Learn More**: [Marketplace platform documentation](../swarms_platform/index.md)

