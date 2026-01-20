# One-to-One Communication Tutorial

A simple guide to using the one-to-one communication pattern between two agents.

## Overview

The **one_to_one** function implements direct communication between two agents, where one agent (sender) processes a task and the other agent (receiver) processes the sender's response. This pattern supports iterative communication loops for refinement and collaboration.

| Feature | Description |
|---------|-------------|
| **Direct Communication** | Two agents communicate directly |
| **Iterative Loops** | Supports multiple communication rounds |
| **Response Chaining** | Receiver processes sender's output |

### When to Use One-to-One

| Scenario | Recommendation |
|----------|----------------|
| Two-agent collaborations | Best For |
| Iterative refinement workflows | Best For |
| Question-answer patterns | Best For |
| Review and feedback loops | Best For |
| Simple agent interactions | Best For |
| Multi-agent scenarios | Not Ideal For |
| Parallel processing | Not Ideal For |
| Complex orchestration | Not Ideal For |

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
from swarms.structs.swarming_architectures import one_to_one
```

### Step 2: Create Two Agents

```python
# Sender agent
sender = Agent(
    agent_name="Researcher",
    system_prompt="You are a researcher. You gather and present information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Receiver agent
receiver = Agent(
    agent_name="Analyst",
    system_prompt="You are an analyst. You analyze information and provide insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Define Your Task

```python
task = "Research the benefits of renewable energy and provide analysis"
```

### Step 4: Run One-to-One Communication

```python
# Execute one-to-one communication
result = one_to_one(
    sender=sender,
    receiver=receiver,
    task=task,
    max_loops=1,  # Number of communication rounds
    output_type="dict"
)

print(result)
```

---

## Complete Example

```python
from swarms import Agent
from swarms.structs.swarming_architectures import one_to_one

# Step 1: Create sender agent
researcher = Agent(
    agent_name="Research-Specialist",
    system_prompt="""You are a research specialist. You conduct thorough research 
    and present findings in a clear, structured manner.""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Step 2: Create receiver agent
analyst = Agent(
    agent_name="Data-Analyst",
    system_prompt="""You are a data analyst. You analyze research findings, 
    identify patterns, and provide actionable insights.""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Step 3: Define the task
task = "Research the impact of AI on job markets and analyze the findings"

# Step 4: Run one-to-one communication with 2 loops
result = one_to_one(
    sender=researcher,
    receiver=analyst,
    task=task,
    max_loops=2,  # 2 rounds of communication
    output_type="dict"
)

# Step 5: Access results
print("One-to-One Communication Results:")
for message in result.get("messages", []):
    role = message['role']
    content = message['content'][:150]
    print(f"\n{role}:")
    print(f"{content}...")
```

---

## Understanding the Flow

The one-to-one pattern works as follows:

**Loop 1:**
1. User provides task
2. Sender processes task → Response 1
3. Receiver processes Response 1 → Response 2

**Loop 2 (if max_loops > 1):**
4. Sender processes Response 2 → Response 3
5. Receiver processes Response 3 → Response 4

And so on for additional loops...

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sender` | `Agent` | Required | The agent that initiates communication |
| `receiver` | `Agent` | Required | The agent that responds to sender |
| `task` | `str` | Required | The initial task to be processed |
| `max_loops` | `int` | `1` | Number of communication rounds |
| `output_type` | `OutputType` | `"dict"` | Output format: "dict" or "list" |

---

## Output Types

### Dictionary Output (Default)

```python
result = one_to_one(
    sender=sender,
    receiver=receiver,
    task="Your task here",
    max_loops=1,
    output_type="dict"
)

# Returns:
# {
#     "messages": [
#         {"role": "User", "content": "Your task here"},
#         {"role": "Researcher", "content": "..."},
#         {"role": "Analyst", "content": "..."}
#     ]
# }
```

### List Output

```python
result = one_to_one(
    sender=sender,
    receiver=receiver,
    task="Your task here",
    max_loops=1,
    output_type="list"
)

# Returns list of responses
```

---

## Use Cases

### Use Case 1: Research and Analysis

```python
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You conduct research and gather information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You analyze research findings and provide insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

result = one_to_one(
    sender=researcher,
    receiver=analyst,
    task="Research climate change solutions and analyze their feasibility",
    max_loops=1
)
```

### Use Case 2: Writer and Editor

```python
writer = Agent(
    agent_name="Writer",
    system_prompt="You write content based on requirements.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    system_prompt="You review and improve written content.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

result = one_to_one(
    sender=writer,
    receiver=editor,
    task="Write an article about sustainable technology",
    max_loops=2  # Writer writes, editor reviews, writer revises
)
```

### Use Case 3: Question and Answer

```python
questioner = Agent(
    agent_name="Questioner",
    system_prompt="You ask thoughtful questions to explore topics.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

answerer = Agent(
    agent_name="Answerer",
    system_prompt="You provide comprehensive answers to questions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

result = one_to_one(
    sender=questioner,
    receiver=answerer,
    task="Explore the topic of quantum computing",
    max_loops=3  # Multiple Q&A rounds
)
```

### Use Case 4: Code Review Pattern

```python
developer = Agent(
    agent_name="Developer",
    system_prompt="You write code and implement features.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

reviewer = Agent(
    agent_name="Code-Reviewer",
    system_prompt="You review code for quality, security, and best practices.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

result = one_to_one(
    sender=developer,
    receiver=reviewer,
    task="Implement a user authentication system",
    max_loops=2  # Developer codes, reviewer reviews, developer fixes
)
```

---

## Iterative Refinement Example

For tasks requiring multiple rounds of refinement:

```python
creator = Agent(
    agent_name="Creator",
    system_prompt="You create initial drafts and content.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

refiner = Agent(
    agent_name="Refiner",
    system_prompt="You refine and improve content iteratively.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Multiple loops for iterative refinement
result = one_to_one(
    sender=creator,
    receiver=refiner,
    task="Create a marketing strategy",
    max_loops=3  # Create → Refine → Refine → Final
)
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Clear Roles** | Define distinct roles for sender and receiver |
| **Appropriate Loops** | Use 1-2 loops for most tasks, 3+ for iterative refinement |
| **Task Clarity** | Make the initial task clear and specific |
| **Complementary Agents** | Ensure agents have complementary skills |

---

## Key Characteristics

| Characteristic | Description |
|---------------|-------------|
| **Simple Pattern** | Easy to understand and implement |
| **Iterative** | Supports multiple rounds of communication |
| **Flexible** | Can model various two-agent workflows |
| **Direct** | No intermediate agents or complex routing |

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[Broadcast](./broadcast_example.md)** | For one-to-many communication |
| **[Circular Swarm](./circular_swarm_example.md)** | For sequential multi-agent processing |
| **[RoundRobinSwarm](./roundrobin_example.md)** | For multi-agent discussions |
