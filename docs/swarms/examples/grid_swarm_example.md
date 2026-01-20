# Grid Swarm Tutorial

A simple guide to using the grid swarm architecture where agents are arranged in a square grid pattern.

## Overview

The **grid_swarm** implements a grid-based task distribution pattern where agents are arranged in a square grid and tasks are distributed across the grid structure. This is useful for parallel processing scenarios where you have a fixed number of agents and tasks.

| Feature | Description |
|---------|-------------|
| **Grid Layout** | Agents arranged in a square grid pattern |
| **Task Distribution** | Tasks distributed across grid positions |
| **Parallel Processing** | Multiple agents can work simultaneously |

### When to Use Grid Swarm

| Scenario | Recommendation |
|----------|----------------|
| Fixed number of agents forming a perfect square (1, 4, 9, 16, etc.) | Best For |
| Parallel task processing | Best For |
| Scenarios requiring structured agent arrangement | Best For |
| Non-square number of agents | Not Ideal For |
| Sequential dependencies between tasks | Not Ideal For |
| Dynamic agent allocation | Not Ideal For |

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Import Required Modules

```python
from swarms import Agent
from swarms.structs.swarming_architectures import grid_swarm
```

### Step 2: Create Your Agents (Must be a Perfect Square)

```python
# Create 4 agents (2x2 grid)
agent1 = Agent(
    agent_name="Agent-1",
    system_prompt="You are a research specialist.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Agent-2",
    system_prompt="You are an analyst.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent3 = Agent(
    agent_name="Agent-3",
    system_prompt="You are a writer.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent4 = Agent(
    agent_name="Agent-4",
    system_prompt="You are an editor.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Define Your Tasks

```python
tasks = [
    "Research AI trends",
    "Analyze market data",
    "Write a summary",
    "Review the content"
]
```

### Step 4: Run the Grid Swarm

```python
# Execute the grid swarm
result = grid_swarm(
    agents=[agent1, agent2, agent3, agent4],
    tasks=tasks,
    output_type="dict"
)

print(result)
```

---

## Complete Example

```python
from swarms import Agent
from swarms.structs.swarming_architectures import grid_swarm

# Step 1: Create 4 agents for a 2x2 grid
agents = [
    Agent(
        agent_name="Researcher",
        system_prompt="You research topics thoroughly.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Analyst",
        system_prompt="You analyze information and identify patterns.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Writer",
        system_prompt="You write clear and concise content.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Reviewer",
        system_prompt="You review and provide feedback.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

# Step 2: Define tasks (should match or exceed number of agents)
tasks = [
    "Research renewable energy benefits",
    "Analyze solar panel efficiency",
    "Write about wind power",
    "Review energy policy documents"
]

# Step 3: Run the grid swarm
result = grid_swarm(
    agents=agents,
    tasks=tasks,
    output_type="dict"
)

# Step 4: Access results
print("Grid Swarm Results:")
for message in result.get("messages", []):
    print(f"{message['role']}: {message['content'][:100]}...")
```

---

## Understanding the Grid Layout

For a 2x2 grid (4 agents), tasks are distributed as follows:

```
[Agent 1] [Agent 2]
[Agent 3] [Agent 4]
```

Tasks are assigned starting from position (0,0) and moving across rows.

---

## Grid Size Calculation

The grid size is automatically calculated as `grid_size = sqrt(number_of_agents)`.

| Number of Agents | Grid Size | Grid Layout |
|------------------|-----------|-------------|
| 1 | 1x1 | Single agent |
| 4 | 2x2 | 2 rows × 2 columns |
| 9 | 3x3 | 3 rows × 3 columns |
| 16 | 4x4 | 4 rows × 4 columns |
| 25 | 5x5 | 5 rows × 5 columns |

**Important**: The number of agents must form a perfect square!

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `AgentListType` | Required | List of Agent objects (must be perfect square) |
| `tasks` | `List[str]` | Required | List of tasks to be processed |
| `output_type` | `OutputType` | `"dict"` | Output format: "dict" or "list" |

---

## Output Types

### Dictionary Output (Default)

```python
result = grid_swarm(
    agents=[agent1, agent2, agent3, agent4],
    tasks=["Task 1", "Task 2", "Task 3", "Task 4"],
    output_type="dict"
)

# Returns conversation history with all agent responses
```

### List Output

```python
result = grid_swarm(
    agents=[agent1, agent2, agent3, agent4],
    tasks=["Task 1", "Task 2", "Task 3", "Task 4"],
    output_type="list"
)

# Returns list of responses
```

---

## Use Cases

### Use Case 1: Parallel Research

```python
# 4 agents for parallel research
agents = [
    Agent(agent_name="Researcher-1", system_prompt="Research expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Researcher-2", system_prompt="Research expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Researcher-3", system_prompt="Research expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Researcher-4", system_prompt="Research expert", model_name="gpt-4o-mini", max_loops=1),
]

tasks = [
    "Research topic A",
    "Research topic B",
    "Research topic C",
    "Research topic D"
]

result = grid_swarm(agents=agents, tasks=tasks)
```

### Use Case 2: Content Review Grid

```python
# 9 agents for comprehensive review (3x3 grid)
agents = [
    Agent(agent_name=f"Reviewer-{i}", system_prompt="Content reviewer", model_name="gpt-4o-mini", max_loops=1)
    for i in range(9)  # 3x3 grid
]

tasks = [
    f"Review document {i}" for i in range(9)
]

result = grid_swarm(agents=agents, tasks=tasks, output_type="dict")
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Perfect Square Agents** | Ensure you have 1, 4, 9, 16, 25, etc. agents |
| **Task Count** | Have at least as many tasks as agents for optimal distribution |
| **Agent Roles** | Consider giving similar roles to agents in the same grid |
| **Parallel Processing** | This architecture works best for independent tasks |

---

## Limitations

| Limitation | Description |
|------------|-------------|
| **Perfect Square Requirement** | Number of agents must be a perfect square |
| **Fixed Structure** | Grid structure is fixed, not dynamic |
| **Task Distribution** | Tasks are distributed sequentially, not randomly |

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[Mesh Swarm](./mesh_swarm_example.md)** | For random task assignment without grid structure |
| **[Circular Swarm](./circular_swarm_example.md)** | For sequential processing |
| **[Star Swarm](./star_swarm_example.md)** | For central coordination pattern |
