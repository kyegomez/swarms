# Swarms CLI Marketplace Guide

The Swarms CLI marketplace commands allow you to search, browse, and install agents from the [Swarms Marketplace](https://swarms.world).

## Prerequisites

1. **API Key**: Set your `SWARMS_API_KEY` environment variable:
   ```bash
   export SWARMS_API_KEY="your-api-key-here"
   ```
   Get your API key at: [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)

2. **Install Swarms**:
   ```bash
   pip install swarms
   ```

## Commands Overview

| Command | Description |
|---------|-------------|
| `swarms marketplace` | Show marketplace help |
| `swarms marketplace list` | List available agents |
| `swarms marketplace search` | Search for agents |
| `swarms marketplace info <id>` | View agent details |
| `swarms marketplace install <id>` | Install agent locally |

## Command Reference

### List Agents

List all available agents in the marketplace:

```bash
swarms marketplace list
```

**Options:**
- `--limit <n>`: Maximum results (default: 20, max: 100)
- `--category <cat>`: Filter by category
- `--free-only`: Show only free agents

**Examples:**
```bash
# List first 10 agents
swarms marketplace list --limit 10

# List free agents only
swarms marketplace list --free-only

# List agents in finance category
swarms marketplace list --category finance
```

### Search Agents

Search for agents by keyword:

```bash
swarms marketplace search --query "keyword"
```

**Options:**
- `--query <text>`: Search keyword (searches name, description, tags)
- `--category <cat>`: Filter by category
- `--free-only`: Show only free agents
- `--limit <n>`: Maximum results

**Examples:**
```bash
# Search for trading agents
swarms marketplace search --query "trading"

# Search in finance category
swarms marketplace search --query "analysis" --category "finance"

# Search free agents only
swarms marketplace search --query "automation" --free-only
```

### View Agent Details

Get detailed information about a specific agent:

```bash
swarms marketplace info <agent-id>
```

**Example:**
```bash
swarms marketplace info a2e3d0d3-9b6a-40a3-9904-000f2e1d03e3
```

This displays:
- Agent name and description
- Category and language
- Price (free or paid)
- Tags and requirements
- Use cases

### Install Agent

Download and install an agent to your local directory:

```bash
swarms marketplace install <agent-id>
```

**Options:**
- `--output-dir <path>`: Directory to save the agent (default: current directory)

**Examples:**
```bash
# Install to current directory
swarms marketplace install a2e3d0d3-9b6a-40a3-9904-000f2e1d03e3

# Install to specific directory
swarms marketplace install a2e3d0d3-9b6a-40a3-9904-000f2e1d03e3 --output-dir ./my_agents
```

The installed file includes:
- Agent metadata as docstring
- Requirements as comments
- Agent code or template

## Available Categories

- `finance`
- `research`
- `coding`
- `content`
- `data-analysis`
- `automation`
- `customer-service`
- `healthcare`
- `legal`
- `marketing`
- `education`
- `general`

## Python API

You can also use the marketplace programmatically:

```python
from swarms.utils.agent_marketplace import (
    query_agents,
    get_agent_by_id,
    install_agent,
    list_available_categories,
)

# List categories
categories = list_available_categories()

# Search agents
result = query_agents(search="trading", category="finance", limit=10)
agents = result["agents"]

# Get agent details
agent = get_agent_by_id("agent-uuid-here")

# Install agent
result = install_agent(agent_id="agent-uuid-here", output_dir="./agents")
print(f"Installed to: {result['file_path']}")
```

## Typical Workflow

1. **Search** for agents matching your needs:
   ```bash
   swarms marketplace search --query "customer service"
   ```

2. **Review** the list and note the agent ID

3. **Get details** about a specific agent:
   ```bash
   swarms marketplace info <agent-id>
   ```

4. **Install** the agent:
   ```bash
   swarms marketplace install <agent-id> --output-dir ./agents
   ```

5. **Use** the agent in your code:
   ```python
   from agents.my_agent import agent
   result = agent.run("Your task here")
   ```

## Troubleshooting

### API Key Error
```
Swarms API key is not set
```
**Solution:** Set your `SWARMS_API_KEY` environment variable or add it to your `.env` file.

### Agent Not Found
```
Agent with ID 'xxx' not found
```
**Solution:** Verify the agent ID is correct using `swarms marketplace list`.

### Rate Limit Exceeded
```
HTTP 429: Too Many Requests
```
**Solution:** The API has a limit of 500 requests per day. Wait until midnight UTC for reset.

## Support

- Documentation: [https://docs.swarms.world](https://docs.swarms.world)
- Marketplace: [https://swarms.world](https://swarms.world)
- Issues: [https://github.com/kyegomez/swarms/issues](https://github.com/kyegomez/swarms/issues)
