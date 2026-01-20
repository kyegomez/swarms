# Mesh Swarm Tutorial

A simple guide to using the mesh swarm architecture where agents work on tasks randomly from a task queue.

## Overview

The **mesh_swarm** implements a mesh topology where agents work on tasks from a shared task queue until all tasks are processed. Tasks are distributed sequentially to agents in a round-robin fashion, making it ideal for parallel processing scenarios.

| Feature | Description |
|---------|-------------|
| **Task Queue** | Shared queue of tasks distributed to agents |
| **Round-Robin Distribution** | Tasks assigned to agents in order |
| **Automatic Completion** | Continues until all tasks are processed |

### When to Use Mesh Swarm

| Scenario | Recommendation |
|----------|----------------|
| Parallel task processing | Best For |
| Independent tasks that can be processed in any order | Best For |
| Scenarios with more tasks than agents | Best For |
| Load balancing across agents | Best For |
| Tasks with dependencies | Not Ideal For |
| Sequential workflows | Not Ideal For |
| Tasks requiring specific agent expertise | Not Ideal For |

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
from swarms.structs.swarming_architectures import mesh_swarm
```

### Step 2: Create Your Agents

```python
# Create multiple agents
agent1 = Agent(
    agent_name="Worker-1",
    system_prompt="You are a task processor. Process tasks efficiently.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Worker-2",
    system_prompt="You are a task processor. Process tasks efficiently.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent3 = Agent(
    agent_name="Worker-3",
    system_prompt="You are a task processor. Process tasks efficiently.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Define Your Tasks

```python
tasks = [
    "Process document 1",
    "Process document 2",
    "Process document 3",
    "Process document 4",
    "Process document 5"
]
```

### Step 4: Run the Mesh Swarm

```python
# Execute the mesh swarm
result = mesh_swarm(
    agents=[agent1, agent2, agent3],
    tasks=tasks,
    output_type="dict"
)

print(result)
```

---

## Complete Example

```python
from swarms import Agent
from swarms.structs.swarming_architectures import mesh_swarm

# Step 1: Create worker agents
workers = [
    Agent(
        agent_name=f"Worker-{i+1}",
        system_prompt="You process tasks efficiently and accurately.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    for i in range(3)  # 3 workers
]

# Step 2: Define a list of tasks
tasks = [
    "Analyze customer feedback from Q1",
    "Review product specifications",
    "Process inventory data",
    "Generate sales report",
    "Update documentation",
    "Review code changes"
]

# Step 3: Run the mesh swarm
result = mesh_swarm(
    agents=workers,
    tasks=tasks,
    output_type="dict"
)

# Step 4: Access results
print("Mesh Swarm Results:")
print(f"Total messages: {len(result.get('messages', []))}")

# Display each agent's work
for message in result.get("messages", []):
    if message['role'] != 'User':
        print(f"{message['role']}: {message['content'][:100]}...")
```

---

## Understanding the Flow

The mesh swarm distributes tasks as follows:

1. **Task Queue**: All tasks are placed in a queue
2. **Round-Robin Assignment**: Tasks are assigned to agents in order:
   - Task 1 → Agent 1
   - Task 2 → Agent 2
   - Task 3 → Agent 3
   - Task 4 → Agent 1 (wraps around)
   - Task 5 → Agent 2
   - And so on...
3. **Completion**: Process continues until queue is empty

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `AgentListType` | Required | List of Agent objects |
| `tasks` | `List[str]` | Required | List of tasks to be processed |
| `output_type` | `OutputType` | `"dict"` | Output format: "dict" or "list" |

---

## Output Types

### Dictionary Output (Default)

```python
result = mesh_swarm(
    agents=[agent1, agent2, agent3],
    tasks=["Task 1", "Task 2", "Task 3", "Task 4"],
    output_type="dict"
)

# Returns:
# {
#     "messages": [
#         {"role": "User", "content": ["Task 1", "Task 2", ...]},
#         {"role": "Worker-1", "content": "..."},
#         {"role": "Worker-2", "content": "..."},
#         ...
#     ]
# }
```

### List Output

```python
result = mesh_swarm(
    agents=[agent1, agent2, agent3],
    tasks=["Task 1", "Task 2", "Task 3"],
    output_type="list"
)

# Returns list of responses
```

---

## Use Cases

### Use Case 1: Document Processing

```python
# Create processing agents
processors = [
    Agent(
        agent_name=f"Processor-{i+1}",
        system_prompt="You process and analyze documents.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    for i in range(5)  # 5 processors
]

# List of documents to process
documents = [
    f"Process document {i}" for i in range(20)  # 20 documents
]

result = mesh_swarm(
    agents=processors,
    tasks=documents,
    output_type="dict"
)
```

### Use Case 2: Data Analysis Tasks

```python
# Analysis agents
analysts = [
    Agent(
        agent_name=f"Analyst-{i+1}",
        system_prompt="You analyze data and generate insights.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    for i in range(3)
]

# Analysis tasks
analysis_tasks = [
    "Analyze sales data for January",
    "Analyze sales data for February",
    "Analyze sales data for March",
    "Analyze customer retention",
    "Analyze product performance",
    "Analyze market trends"
]

result = mesh_swarm(
    agents=analysts,
    tasks=analysis_tasks,
    output_type="dict"
)
```

### Use Case 3: Content Generation

```python
# Content creators
creators = [
    Agent(
        agent_name=f"Creator-{i+1}",
        system_prompt="You create engaging content.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    for i in range(4)
]

# Content tasks
content_tasks = [
    "Write blog post about AI",
    "Create social media content",
    "Draft email newsletter",
    "Write product description",
    "Create marketing copy",
    "Write technical documentation"
]

result = mesh_swarm(
    agents=creators,
    tasks=content_tasks,
    output_type="list"
)
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Agent Uniformity** | Use similar agents for uniform task processing |
| **Task Independence** | Ensure tasks can be processed independently |
| **Load Balancing** | More agents = faster processing for many tasks |
| **Task Clarity** | Make tasks clear and self-contained |

---

## Key Characteristics

| Characteristic | Description |
|---------------|-------------|
| **Sequential Distribution** | Tasks assigned in order (not random despite the name) |
| **Queue-Based** | Uses a task queue that depletes as tasks are processed |
| **Automatic Completion** | Stops when all tasks are done |
| **Independent Processing** | Each agent processes tasks independently |

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[Grid Swarm](./grid_swarm_example.md)** | For structured grid-based processing |
| **[Circular Swarm](./circular_swarm_example.md)** | For sequential round-robin with context |
| **[Star Swarm](./star_swarm_example.md)** | For central coordination pattern |
