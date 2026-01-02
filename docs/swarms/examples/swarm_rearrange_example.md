# SwarmRearrange Example

The SwarmRearrange is a multi-agent architecture that orchestrates multiple swarms (not individual agents) in a defined flow pattern. It allows you to chain swarms together sequentially or run them in parallel, creating complex multi-swarm workflows.

## How It Works

1. **Swarm Orchestration**: Manages multiple swarms (like HierarchicalSwarm, MixtureOfAgents, etc.)
2. **Flow Patterns**: Uses arrow (`->`) for sequential and comma (`,`) for parallel execution
3. **Flexible Execution**: Supports complex workflows combining sequential and parallel patterns
4. **Human-in-the-Loop**: Optional human intervention points in the workflow

This architecture is perfect for:
- Complex multi-stage workflows
- Combining different swarm architectures
- Orchestrating hierarchical processes
- Multi-phase analysis pipelines

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
from swarms import Agent, SwarmRearrange, SequentialWorkflow, MixtureOfAgents
```

### Step 2: Create Individual Swarms

```python
# Swarm 1: Research Swarm (using SequentialWorkflow)
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist. Gather and analyze information.",
    model_name="gpt-4o-mini",
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You are a technical writer. Write clear, structured documents.",
    model_name="gpt-4o-mini",
)

research_swarm = SequentialWorkflow(
    name="Research-Swarm",
    agents=[researcher, writer],
    max_loops=1,
)

# Swarm 2: Analysis Swarm (using MixtureOfAgents)
analyst1 = Agent(
    agent_name="Analyst-1",
    system_prompt="You are a data analyst. Analyze data and provide insights.",
    model_name="gpt-4o-mini",
)

analyst2 = Agent(
    agent_name="Analyst-2",
    system_prompt="You are a strategic analyst. Provide strategic recommendations.",
    model_name="gpt-4o-mini",
)

analysis_swarm = MixtureOfAgents(
    agents=[analyst1, analyst2],
    layers=1,
)

# Swarm 3: Review Swarm (single agent as a simple swarm)
reviewer = Agent(
    agent_name="Reviewer",
    system_prompt="You are a quality reviewer. Review and provide feedback.",
    model_name="gpt-4o-mini",
)
```

### Step 3: Create SwarmRearrange with Flow Pattern

```python
# Define flow: Research -> Analysis (parallel) -> Review
# This means: Research runs first, then Analysis swarms run in parallel, then Review
swarm_rearrange = SwarmRearrange(
    name="Document-Pipeline",
    description="A pipeline for document creation and review",
    swarms=[research_swarm, analysis_swarm, reviewer],
    flow="Research-Swarm -> Analysis-Swarm, Reviewer",
    max_loops=1,
    verbose=True,
)
```

### Step 4: Run the SwarmRearrange

```python
task = "Create a comprehensive report on AI trends in 2024"

result = swarm_rearrange.run(task=task)

print(result)
```

## Flow Pattern Examples

### Sequential Flow

```python
# All swarms run one after another
flow = "Research-Swarm -> Analysis-Swarm -> Reviewer"
```

### Parallel Flow

```python
# Multiple swarms run at the same time
flow = "Research-Swarm, Analysis-Swarm -> Reviewer"
```

### Complex Flow

```python
# Mix of sequential and parallel
flow = "Research-Swarm -> Analysis-Swarm, Reviewer -> Final-Reviewer"
```

## Advanced: Human-in-the-Loop

You can add human intervention points in your workflow:

```python
def human_reviewer(input_text):
    """Custom human-in-the-loop function"""
    print(f"Review this output: {input_text}")
    feedback = input("Enter your feedback: ")
    return feedback

swarm_rearrange = SwarmRearrange(
    name="Human-Reviewed-Pipeline",
    swarms=[research_swarm, analysis_swarm, reviewer],
    flow="Research-Swarm -> H -> Analysis-Swarm -> Reviewer",
    human_in_the_loop=True,
    custom_human_in_the_loop=human_reviewer,
    max_loops=1,
)
```

## Understanding the Flow

- `->` means sequential: The swarm on the left completes before the one on the right starts
- `,` means parallel: All swarms separated by commas run simultaneously
- `H` means human-in-the-loop: Pauses for human input
- You can combine these patterns for complex workflows

## Example: Multi-Phase Analysis

```python
# Phase 1: Research (sequential)
# Phase 2: Analysis (parallel - multiple analysts)
# Phase 3: Synthesis (sequential)
# Phase 4: Review (parallel - multiple reviewers)

flow = "Research-Swarm -> Analyst-1, Analyst-2, Analyst-3 -> Synthesis-Swarm -> Reviewer-1, Reviewer-2"
```

## Support and Community

If you're facing issues or want to learn more, check out the following resources:

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@swarms_corp](https://x.com/swarms_corp) | Latest news and announcements |

