# Circular Swarm Tutorial

A simple guide to using the circular swarm architecture where agents pass tasks in a circular manner.

## Overview

The **circular_swarm** implements a circular communication pattern where agents process tasks sequentially in a round-robin fashion. Each agent sees the full conversation history and can build upon previous agents' work.

| Feature | Description |
|---------|-------------|
| **Circular Processing** | Agents process tasks in a circular order |
| **Full Context** | Each agent sees complete conversation history |
| **Sequential Flow** | Tasks flow through agents in a predictable pattern |

### When to Use Circular Swarm

| Scenario | Recommendation |
|----------|----------------|
| Tasks requiring sequential processing | Best For |
| Workflows where each agent builds on the previous one | Best For |
| Situations needing predictable agent order | Best For |
| Tasks requiring parallel processing | Not Ideal For |
| Independent task execution | Not Ideal For |
| Random or dynamic agent selection | Not Ideal For |

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
from swarms.structs.swarming_architectures import circular_swarm
```

### Step 2: Create Your Agents

```python
# Create specialized agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist. Gather and present factual information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analyst. Interpret data and identify patterns.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You synthesize information into clear, actionable insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Define Your Tasks

```python
tasks = [
    "Research the latest trends in AI development",
    "Analyze the impact of remote work on productivity"
]
```

### Step 4: Run the Circular Swarm

```python
# Execute the circular swarm
result = circular_swarm(
    agents=[researcher, analyst, writer],
    tasks=tasks,
    output_type="dict"  # Options: "dict" or "list"
)

print(result)
```

---

## Complete Example

```python
from swarms import Agent
from swarms.structs.swarming_architectures import circular_swarm

# Step 1: Create agents with distinct roles
researcher = Agent(
    agent_name="Research-Specialist",
    system_prompt="You research and gather factual information on topics.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Data-Analyst",
    system_prompt="You analyze data and identify patterns and insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Content-Writer",
    system_prompt="You synthesize information into clear, actionable content.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Step 2: Define tasks
tasks = [
    "Analyze the benefits of renewable energy",
    "Research sustainable transportation solutions"
]

# Step 3: Run the circular swarm
result = circular_swarm(
    agents=[researcher, analyst, writer],
    tasks=tasks,
    output_type="dict"
)

# Step 4: Access the results
print("Conversation History:")
for message in result.get("messages", []):
    print(f"{message['role']}: {message['content'][:100]}...")
```

---

## Understanding the Flow

The circular swarm processes tasks as follows:

1. **Task 1** → Agent 1 → Agent 2 → Agent 3
2. **Task 2** → Agent 1 → Agent 2 → Agent 3
3. And so on...

Each agent sees the full conversation history, allowing them to build upon previous responses.

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `AgentListType` | Required | List of Agent objects to participate |
| `tasks` | `List[str]` | Required | List of tasks to be processed |
| `output_type` | `OutputType` | `"dict"` | Output format: "dict" or "list" |

---

## Output Types

### Dictionary Output (Default)

```python
result = circular_swarm(
    agents=[agent1, agent2, agent3],
    tasks=["Task 1", "Task 2"],
    output_type="dict"
)

# Returns:
# {
#     "messages": [
#         {"role": "User", "content": "Task 1"},
#         {"role": "Researcher", "content": "..."},
#         ...
#     ]
# }
```

### List Output

```python
result = circular_swarm(
    agents=[agent1, agent2, agent3],
    tasks=["Task 1", "Task 2"],
    output_type="list"
)

# Returns:
# ["Response 1", "Response 2", ...]
```

---

## Use Cases

### Use Case 1: Content Creation Pipeline

```python
agents = [
    Agent(agent_name="Researcher", system_prompt="Research expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Writer", system_prompt="Content writer", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Editor", system_prompt="Editor and proofreader", model_name="gpt-4o-mini", max_loops=1),
]

tasks = [
    "Create an article about climate change",
    "Write about AI ethics"
]

result = circular_swarm(agents=agents, tasks=tasks)
```

### Use Case 2: Analysis Workflow

```python
agents = [
    Agent(agent_name="Data-Collector", system_prompt="Collects data", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Analyzer", system_prompt="Analyzes data", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Reporter", system_prompt="Creates reports", model_name="gpt-4o-mini", max_loops=1),
]

tasks = [
    "Analyze Q4 sales data",
    "Review customer feedback trends"
]

result = circular_swarm(agents=agents, tasks=tasks, output_type="dict")
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Agent Order Matters** | Place agents in the order you want them to process tasks |
| **Clear Roles** | Give each agent a distinct, well-defined role |
| **Task Clarity** | Ensure tasks are clear and specific |
| **Output Type** | Use "dict" for full conversation history, "list" for simple responses |

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[RoundRobinSwarm](./roundrobin_example.md)** | For randomized turn order |
| **[SequentialWorkflow](./sequential_example.md)** | For complex workflows with dependencies |
| **[Mesh Swarm](./mesh_swarm_example.md)** | For random task assignment |
