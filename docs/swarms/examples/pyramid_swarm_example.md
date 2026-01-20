# Pyramid Swarm Tutorial

A simple guide to using the pyramid swarm architecture where agents are arranged in a pyramid structure.

## Overview

The **pyramid_swarm** implements a pyramid topology where agents are arranged in hierarchical levels, with fewer agents at the top and more at the bottom. Tasks are distributed starting from the top level and moving down through the pyramid structure.

| Feature | Description |
|---------|-------------|
| **Pyramid Structure** | Agents arranged in triangular levels |
| **Hierarchical Processing** | Tasks flow from top to bottom |
| **Level-Based Distribution** | Tasks assigned by pyramid level |

### When to Use Pyramid Swarm

| Scenario | Recommendation |
|----------|----------------|
| Hierarchical workflows | Best For |
| Tasks requiring top-down processing | Best For |
| Scenarios with leadership and execution layers | Best For |
| Number of agents that form a pyramid (1, 3, 6, 10, 15, etc.) | Best For |
| Flat organizational structures | Not Ideal For |
| Non-hierarchical task processing | Not Ideal For |
| Random or mesh topologies | Not Ideal For |

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
from swarms.structs.swarming_architectures import pyramid_swarm
```

### Step 2: Create Your Agents (Pyramid Numbers)

```python
# Create 6 agents for a 3-level pyramid:
# Level 1: 1 agent (top)
# Level 2: 2 agents (middle)
# Level 3: 3 agents (bottom)

leader = Agent(
    agent_name="Leader",
    system_prompt="You are a leader. You coordinate and make strategic decisions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

manager1 = Agent(
    agent_name="Manager-1",
    system_prompt="You are a manager. You oversee operations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

manager2 = Agent(
    agent_name="Manager-2",
    system_prompt="You are a manager. You oversee operations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

worker1 = Agent(
    agent_name="Worker-1",
    system_prompt="You are a worker. You execute tasks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

worker2 = Agent(
    agent_name="Worker-2",
    system_prompt="You are a worker. You execute tasks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

worker3 = Agent(
    agent_name="Worker-3",
    system_prompt="You are a worker. You execute tasks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Define Your Tasks

```python
tasks = [
    "Strategic planning task",
    "Operational task 1",
    "Operational task 2",
    "Execution task 1",
    "Execution task 2",
    "Execution task 3"
]
```

### Step 4: Run the Pyramid Swarm

```python
# Execute the pyramid swarm
result = pyramid_swarm(
    agents=[leader, manager1, manager2, worker1, worker2, worker3],
    tasks=tasks,
    output_type="dict"
)

print(result)
```

---

## Complete Example

```python
from swarms import Agent
from swarms.structs.swarming_architectures import pyramid_swarm

# Step 1: Create agents for a 3-level pyramid (6 agents total)
# Level 1: 1 agent
ceo = Agent(
    agent_name="CEO",
    system_prompt="You are the CEO. You make strategic decisions and set direction.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Level 2: 2 agents
vp1 = Agent(
    agent_name="VP-Engineering",
    system_prompt="You are VP of Engineering. You oversee technical operations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

vp2 = Agent(
    agent_name="VP-Marketing",
    system_prompt="You are VP of Marketing. You oversee marketing operations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Level 3: 3 agents
engineer1 = Agent(
    agent_name="Engineer-1",
    system_prompt="You are an engineer. You implement technical solutions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

engineer2 = Agent(
    agent_name="Engineer-2",
    system_prompt="You are an engineer. You implement technical solutions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

marketer = Agent(
    agent_name="Marketer",
    system_prompt="You are a marketer. You execute marketing campaigns.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Step 2: Define tasks
tasks = [
    "Set company strategy",
    "Plan engineering roadmap",
    "Plan marketing strategy",
    "Implement feature A",
    "Implement feature B",
    "Launch campaign"
]

# Step 3: Run the pyramid swarm
result = pyramid_swarm(
    agents=[ceo, vp1, vp2, engineer1, engineer2, marketer],
    tasks=tasks,
    output_type="dict"
)

# Step 4: Access results
print("Pyramid Swarm Results:")
for message in result.get("messages", []):
    print(f"{message['role']}: {message['content'][:100]}...")
```

---

## Understanding the Pyramid Structure

The pyramid structure is calculated automatically based on the number of agents. Tasks are distributed starting from the top level and moving down.

| Level | Agents per Level | Cumulative Agents | Example |
|-------|------------------|-------------------|---------|
| 1 | 1 | 1 | Top leader |
| 2 | 2 | 3 | 1 + 2 |
| 3 | 3 | 6 | 1 + 2 + 3 |
| 4 | 4 | 10 | 1 + 2 + 3 + 4 |
| 5 | 5 | 15 | 1 + 2 + 3 + 4 + 5 |
| 6 | 6 | 21 | 1 + 2 + 3 + 4 + 5 + 6 |

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `AgentListType` | Required | List of Agent objects (must form a pyramid) |
| `tasks` | `List[str]` | Required | List of tasks to be processed |
| `output_type` | `OutputType` | `"dict"` | Output format: "dict" or "list" |

---

## Output Types

### Dictionary Output (Default)

```python
result = pyramid_swarm(
    agents=[agent1, agent2, agent3, agent4, agent5, agent6],
    tasks=["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6"],
    output_type="dict"
)

# Returns full conversation history with hierarchical processing
```

### List Output

```python
result = pyramid_swarm(
    agents=[agent1, agent2, agent3],
    tasks=["Task 1", "Task 2", "Task 3"],
    output_type="list"
)

# Returns list of responses
```

---

## Use Cases

### Use Case 1: Organizational Hierarchy

```python
# 10 agents for a 4-level pyramid
agents = [
    # Level 1: CEO
    Agent(agent_name="CEO", system_prompt="Strategic leader", model_name="gpt-4o-mini", max_loops=1),
    # Level 2: VPs
    Agent(agent_name="VP-1", system_prompt="VP operations", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="VP-2", system_prompt="VP operations", model_name="gpt-4o-mini", max_loops=1),
    # Level 3: Directors
    Agent(agent_name="Director-1", system_prompt="Director operations", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Director-2", system_prompt="Director operations", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Director-3", system_prompt="Director operations", model_name="gpt-4o-mini", max_loops=1),
    # Level 4: Managers
    Agent(agent_name="Manager-1", system_prompt="Manager operations", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Manager-2", system_prompt="Manager operations", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Manager-3", system_prompt="Manager operations", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Manager-4", system_prompt="Manager operations", model_name="gpt-4o-mini", max_loops=1),
]

tasks = [f"Task {i+1}" for i in range(10)]

result = pyramid_swarm(agents=agents, tasks=tasks)
```

### Use Case 2: Research Hierarchy

```python
# 6 agents for a 3-level research pyramid
agents = [
    # Principal Investigator
    Agent(agent_name="PI", system_prompt="Principal investigator", model_name="gpt-4o-mini", max_loops=1),
    # Research Associates
    Agent(agent_name="RA-1", system_prompt="Research associate", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="RA-2", system_prompt="Research associate", model_name="gpt-4o-mini", max_loops=1),
    # Research Assistants
    Agent(agent_name="Assistant-1", system_prompt="Research assistant", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Assistant-2", system_prompt="Research assistant", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Assistant-3", system_prompt="Research assistant", model_name="gpt-4o-mini", max_loops=1),
]

tasks = [
    "Design research methodology",
    "Plan data collection",
    "Plan analysis approach",
    "Collect data set 1",
    "Collect data set 2",
    "Collect data set 3"
]

result = pyramid_swarm(agents=agents, tasks=tasks, output_type="dict")
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Pyramid Numbers** | Use 1, 3, 6, 10, 15, 21, etc. agents for proper pyramid structure |
| **Hierarchical Roles** | Assign roles that match the pyramid levels |
| **Top-Down Thinking** | Design tasks to flow from strategic (top) to tactical (bottom) |
| **Level Alignment** | Ensure agents at each level have appropriate responsibilities |

---

## Pyramid Number Calculation

The number of levels is calculated using:
```
levels = (-1 + sqrt(1 + 8 * num_agents)) / 2
```

| Number of Agents | Number of Levels | Structure |
|------------------|------------------|----------|
| 1 | 1 | Single agent |
| 3 | 2 | 1 top + 2 middle |
| 6 | 3 | 1 top + 2 middle + 3 bottom |
| 10 | 4 | 1 + 2 + 3 + 4 |
| 15 | 5 | 1 + 2 + 3 + 4 + 5 |
| 21 | 6 | 1 + 2 + 3 + 4 + 5 + 6 |

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[HierarchicalSwarm](./hierarchical_swarm_example.md)** | For more flexible hierarchical structures |
| **[Star Swarm](./star_swarm_example.md)** | For central coordination without strict hierarchy |
| **[Grid Swarm](./grid_swarm_example.md)** | For flat grid-based processing |
