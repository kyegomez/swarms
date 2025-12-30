# AutoSwarmBuilder: 3-Step Quickstart Guide

The AutoSwarmBuilder automatically designs and creates specialized multi-agent teams based on your task description. Simply describe what you need, and it will generate agents with distinct roles, expertise, personalities, and comprehensive system prompts - then orchestrate them using the most appropriate swarm architecture.

## Overview

| Feature | Description |
|---------|-------------|
| **Automatic Agent Generation** | Creates agents with roles, personalities, and expertise based on task |
| **Intelligent Architecture Selection** | Chooses optimal swarm type (Sequential, Concurrent, Hierarchical, etc.) |
| **Comprehensive System Prompts** | Generates detailed prompts with decision-making frameworks |
| **Flexible Execution** | Returns agents, swarm router config, or agent objects |

```
Your Task Description
        │
        ▼
   AutoSwarmBuilder
   (Boss System Prompt)
        │
        ▼
┌───────────────────────┐
│ Auto-Generated Team   │
│  - Agent Roles        │
│  - Personalities      │
│  - System Prompts     │
│  - Architecture Type  │
└───────────────────────┘
        │
        ▼
    Ready to Run
```

---

## Step 1: Install and Import

```bash
pip install swarms
```

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
```

---

## Step 2: Create AutoSwarmBuilder

```python
# Initialize the builder
swarm_builder = AutoSwarmBuilder(
    name="Marketing-Team-Builder",
    description="Builds marketing teams automatically",
    model_name="gpt-4o",  # Boss agent model
    max_loops=1,
    execution_type="return-agents",  # or "return-swarm-router-config", "return-agents-objects"
    verbose=True
)
```

---

## Step 3: Generate and Run

```python
# Describe what you need
task = "Create a marketing team with 4 agents: market researcher, content strategist, copywriter, and social media specialist. They should collaborate on launching a new AI product."

# Auto-generate the team
result = swarm_builder.run(task=task)

# The builder creates:
# - 4 agents with specialized roles
# - Comprehensive system prompts for each
# - Appropriate swarm architecture
# - Ready-to-use configuration

print(result)
```

---

## Complete Example

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
import json

# Create builder
swarm = AutoSwarmBuilder(
    name="Product-Development-Team",
    description="Auto-generates product development teams",
    model_name="gpt-4o",
    max_loops=1,
    execution_type="return-agents",
    verbose=True
)

# Define your need
task = """
Create a product development team with 5 specialized agents:
1. Product Manager - oversees strategy and roadmap
2. UX Designer - focuses on user experience
3. Backend Engineer - handles server-side development
4. Frontend Engineer - builds user interfaces
5. QA Engineer - ensures quality and testing

The team should work together to plan and build a new mobile app feature.
"""

# Generate the team
team_config = swarm.run(task=task)

# View the generated team
print(json.dumps(team_config, indent=2))
```

---

## Execution Types

| Type | Returns | Use Case |
|------|---------|----------|
| `"return-agents"` | List of agent dictionaries | Inspect and customize agents |
| `"return-swarm-router-config"` | Complete SwarmRouter configuration | Ready-to-use swarm |
| `"return-agents-objects"` | List of Agent objects | Direct execution |

### Example: Get Ready-to-Run Swarm

```python
swarm = AutoSwarmBuilder(
    name="Research-Team",
    model_name="gpt-4o",
    execution_type="return-swarm-router-config",  # Get complete swarm
)

result = swarm.run(
    "Create a research team with data analyst, statistician, and research coordinator"
)

# Result is a complete SwarmRouter configuration
# Ready to use immediately
```

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | Required | Name of the builder |
| `description` | Required | Purpose of the builder |
| `model_name` | `"gpt-4o"` | Model for the boss agent that designs teams |
| `max_loops` | `1` | Loops for agent generation |
| `execution_type` | `"return-agents"` | What to return |
| `verbose` | `False` | Enable detailed logging |

---

## Use Cases

| Scenario | Team Description |
|----------|------------------|
| **Content Creation** | "Writers, editors, SEO specialists for blog content" |
| **Software Development** | "Full-stack developers, QA engineers, DevOps for microservices" |
| **Financial Analysis** | "Financial analysts, risk managers, compliance officers for investment portfolio" |
| **Customer Support** | "Support agents, escalation specialists, quality reviewers for customer service" |
| **Research** | "Researchers, data scientists, literature reviewers for scientific study" |

### Example: Financial Analysis Team

```python
swarm = AutoSwarmBuilder(
    name="Financial-Team-Builder",
    model_name="gpt-4o",
    execution_type="return-agents",
)

team = swarm.run(
    """
    Create a financial analysis team with:
    - Equity Analyst: Analyzes stocks and market trends
    - Fixed Income Analyst: Evaluates bonds and debt instruments
    - Risk Manager: Assesses portfolio risk
    - Quantitative Analyst: Builds financial models

    Team should collaborate on portfolio management and investment recommendations.
    """
)

print(f"Generated {len(team)} specialized financial agents")
```

---

## How It Works

1. **Task Analysis**: Boss agent analyzes your requirements
2. **Agent Design**: Creates agents with:
   - Unique roles and purposes
   - Distinct personalities
   - Comprehensive system prompts
   - Specific capabilities and limitations
3. **Architecture Selection**: Chooses optimal swarm type
4. **Configuration Generation**: Outputs ready-to-use configuration
5. **Return**: Provides agents in requested format

---

## Advanced Features

### Custom Boss System Prompt

The boss agent uses a sophisticated system prompt that considers:
- Task decomposition and analysis
- Agent design excellence with personalities
- Communication protocols and collaboration strategies
- Multi-agent architecture selection
- Quality assurance and governance

### Supported Swarm Architectures

The boss can select from:
- AgentRearrange
- MixtureOfAgents
- SpreadSheetSwarm
- SequentialWorkflow
- ConcurrentWorkflow
- GroupChat
- MultiAgentRouter
- HierarchicalSwarm
- MajorityVoting
- And more...

---

## Best Practices

- **Be Specific**: Provide clear, detailed task descriptions
- **Define Roles**: Specify the types of agents you need
- **State Objectives**: Explain what the team should accomplish
- **Use Powerful Models**: Use gpt-4o or claude-sonnet for best results
- **Review Output**: Always review and potentially customize generated agents

---

## Related Architectures

- [SwarmRouter](../swarms/examples/swarm_router.md) - Routes tasks to appropriate swarms
- [HierarchicalSwarm](../swarms/examples/hierarchical_swarm_example.md) - Manual hierarchical teams
- [Multi-Agent Examples](./multi_agent_architectures_overview.md) - Pre-built architectures

---

## Next Steps

- Explore [AutoSwarmBuilder Tutorial](../swarms/examples/auto_swarm_builder_example.md)
- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/asb)
- Learn about [Agent Design Principles](../swarms/concept/agent_design.md)
