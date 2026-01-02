# SpreadSheetSwarm Example

The SpreadSheetSwarm is a multi-agent architecture that processes tasks concurrently using multiple agents and saves the results to a CSV file. This is perfect for batch processing, data collection, and tracking agent outputs over time.

## How It Works

1. **Concurrent Execution**: Multiple agents work on tasks simultaneously
2. **Task Distribution**: Each agent can have its own task or all agents work on the same task
3. **CSV Tracking**: All results are automatically saved to a CSV file with metadata
4. **Batch Processing**: Supports running multiple loops for repeated tasks

This architecture is ideal for:
- Data collection and analysis
- Batch processing multiple tasks
- Tracking agent performance over time
- Generating reports from multiple agents

## Installation

Install the swarms package using pip:

```bash
pip install -U swarms
```

## Basic Setup

1. First, set up your environment variables:

```python
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY="your-api-key"
```

## Step-by-Step Example

### Step 1: Import Required Modules

```python
from swarms import Agent, SpreadSheetSwarm
```

### Step 2: Create Specialized Agents

```python
# Agent 1: Market Researcher
market_researcher = Agent(
    agent_name="Market-Researcher",
    system_prompt="You are a market research analyst. Analyze market trends, competitors, and opportunities.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Agent 2: Financial Analyst
financial_analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="You are a financial analyst. Analyze financial data, calculate metrics, and provide insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Agent 3: Risk Assessor
risk_assessor = Agent(
    agent_name="Risk-Assessor",
    system_prompt="You are a risk assessment specialist. Identify and evaluate potential risks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Create the SpreadSheetSwarm

```python
swarm = SpreadSheetSwarm(
    name="Market-Analysis-Swarm",
    description="A swarm for comprehensive market analysis",
    agents=[
        market_researcher,
        financial_analyst,
        risk_assessor,
    ],
    max_loops=1,
    autosave=True,
    verbose=True,
)
```

### Step 4: Run the Swarm

```python
# All agents work on the same task concurrently
task = "Analyze the current state of the AI market and provide key insights"

result = swarm.run(task=task)

print(result)
```

### Step 5: Access Results

The results include:
- Run ID and metadata
- Number of tasks completed
- All agent outputs
- Timestamps for each task

The result dictionary contains all this information, which you can access programmatically.

## Advanced: Different Tasks for Each Agent

You can also configure each agent with a different task using a CSV file or by setting tasks programmatically:

```python
# Create swarm with agent-specific tasks
swarm = SpreadSheetSwarm(
    name="Multi-Task-Swarm",
    agents=[market_researcher, financial_analyst, risk_assessor],
    max_loops=1,
)

# Set different tasks for each agent
swarm.agent_tasks = {
    "Market-Researcher": "Research AI market trends",
    "Financial-Analyst": "Calculate market size and growth rate",
    "Risk-Assessor": "Identify market risks and challenges",
}

# Run from configuration
result = swarm.run_from_config()
```

## CSV Output

The swarm automatically saves results to a CSV file in your workspace directory. The CSV includes:
- Run ID
- Agent Name
- Task
- Result
- Timestamp

You can find the CSV file in your `WORKSPACE_DIR` with a filename like:
`spreadsheet_swarm_run_id_{uuid}.csv`

## Support and Community

If you're facing issues or want to learn more, check out the following resources:

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@swarms_corp](https://x.com/swarms_corp) | Latest news and announcements |

