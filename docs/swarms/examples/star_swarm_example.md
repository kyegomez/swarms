# Star Swarm Tutorial

A simple guide to using the star swarm architecture where a central agent processes tasks first, followed by others.

## Overview

The **star_swarm** implements a star topology communication pattern where one central agent processes all tasks first, and then other agents process the same tasks. This is useful for scenarios where you need a central coordinator or initial processing before distributed work.

| Feature | Description |
|---------|-------------|
| **Central Agent** | First agent processes all tasks initially |
| **Distributed Processing** | Other agents then process the same tasks |
| **Two-Phase Processing** | Central processing followed by distributed processing |

### When to Use Star Swarm

| Scenario | Recommendation |
|----------|----------------|
| Tasks requiring initial coordination or setup | Best For |
| Scenarios with a lead agent and supporting agents | Best For |
| Two-phase processing workflows | Best For |
| Fully parallel processing | Not Ideal For |
| Independent task execution | Not Ideal For |
| Scenarios without a central coordinator | Not Ideal For |

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
from swarms.structs.swarming_architectures import star_swarm
```

### Step 2: Create Your Agents

```python
# Central agent (processes first)
coordinator = Agent(
    agent_name="Coordinator",
    system_prompt="You are a coordinator. You analyze tasks and provide initial insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Supporting agents (process after coordinator)
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analyst.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Define Your Tasks

```python
tasks = [
    "Analyze market trends",
    "Research competitor strategies"
]
```

### Step 4: Run the Star Swarm

```python
# Execute the star swarm
# First agent is the central agent
result = star_swarm(
    agents=[coordinator, researcher, analyst],
    tasks=tasks,
    output_type="dict"
)

print(result)
```

---

## Complete Example

```python
from swarms import Agent
from swarms.structs.swarming_architectures import star_swarm

# Step 1: Create central coordinator agent
coordinator = Agent(
    agent_name="Project-Coordinator",
    system_prompt="""You are a project coordinator. You analyze tasks, 
    break them down, and provide initial direction.""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Step 2: Create supporting agents
researcher = Agent(
    agent_name="Research-Specialist",
    system_prompt="You research topics thoroughly and gather information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Data-Analyst",
    system_prompt="You analyze data and identify patterns.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Content-Writer",
    system_prompt="You synthesize information into clear content.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Step 3: Define tasks
tasks = [
    "Develop a marketing strategy for a new product",
    "Create a content plan for Q4"
]

# Step 4: Run the star swarm
# Note: First agent in the list is the central agent
result = star_swarm(
    agents=[coordinator, researcher, analyst, writer],
    tasks=tasks,
    output_type="dict"
)

# Step 5: Access results
print("Star Swarm Results:")
for message in result.get("messages", []):
    role = message['role']
    content = message['content'][:150]
    print(f"{role}: {content}...")
```

---

## Understanding the Flow

The star swarm processes tasks in two phases:

**Phase 1: Central Agent Processing**
- Task 1 → Central Agent
- Task 2 → Central Agent
- (All tasks processed by central agent first)

**Phase 2: Supporting Agents Processing**
- Task 1 → Supporting Agent 1
- Task 1 → Supporting Agent 2
- Task 2 → Supporting Agent 1
- Task 2 → Supporting Agent 2
- (Each supporting agent processes all tasks)

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `AgentListType` | Required | List of Agent objects (first is central) |
| `tasks` | `List[str]` | Required | List of tasks to be processed |
| `output_type` | `OutputType` | `"dict"` | Output format: "dict" or "list" |

---

## Output Types

### Dictionary Output (Default)

```python
result = star_swarm(
    agents=[coordinator, agent1, agent2],
    tasks=["Task 1", "Task 2"],
    output_type="dict"
)

# Returns full conversation history including:
# - Central agent's processing of all tasks
# - Supporting agents' processing of all tasks
```

### List Output

```python
result = star_swarm(
    agents=[coordinator, agent1, agent2],
    tasks=["Task 1", "Task 2"],
    output_type="list"
)

# Returns list of all responses
```

---

## Use Cases

### Use Case 1: Project Coordination

```python
# Central project manager
project_manager = Agent(
    agent_name="Project-Manager",
    system_prompt="You coordinate projects and break down tasks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Team members
team = [
    Agent(agent_name="Developer", system_prompt="Technical implementation", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Designer", system_prompt="UI/UX design", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Tester", system_prompt="Quality assurance", model_name="gpt-4o-mini", max_loops=1),
]

tasks = [
    "Plan the new feature implementation",
    "Coordinate the release process"
]

result = star_swarm(
    agents=[project_manager] + team,
    tasks=tasks
)
```

### Use Case 2: Research Coordination

```python
# Lead researcher
lead_researcher = Agent(
    agent_name="Lead-Researcher",
    system_prompt="You coordinate research efforts and provide initial analysis.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Research team
research_team = [
    Agent(agent_name="Data-Collector", system_prompt="Collects research data", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Statistician", system_prompt="Statistical analysis", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Writer", system_prompt="Research writing", model_name="gpt-4o-mini", max_loops=1),
]

tasks = [
    "Research climate change impacts",
    "Analyze renewable energy adoption"
]

result = star_swarm(
    agents=[lead_researcher] + research_team,
    tasks=tasks,
    output_type="dict"
)
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Central Agent Selection** | Choose the first agent carefully as it coordinates all tasks |
| **Clear Roles** | Give the central agent a coordination/leadership role |
| **Supporting Agents** | Ensure supporting agents have complementary skills |
| **Task Clarity** | Make tasks clear for both central and supporting agents |

---

## Key Characteristics

| Characteristic | Description |
|---------------|-------------|
| **Sequential Central Processing** | Central agent processes all tasks before others |
| **Independent Supporting Processing** | Supporting agents process tasks independently |
| **Full Context** | Central agent sees conversation history, supporting agents see tasks directly |

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[Broadcast](./broadcast_example.md)** | For one-to-many communication pattern |
| **[Circular Swarm](./circular_swarm_example.md)** | For sequential round-robin processing |
| **[Mesh Swarm](./mesh_swarm_example.md)** | For random task assignment |
