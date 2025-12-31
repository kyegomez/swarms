# RoundRobinSwarm: 3-Step Quickstart Guide

The RoundRobinSwarm implements an AutoGen-style communication pattern where agents are shuffled randomly each loop for varied interaction patterns. Each agent receives the full conversation context to build upon others' responses, creating collaborative and iterative refinement.

## Overview

| Feature | Description |
|---------|-------------|
| **Randomized Order** | Agents are shuffled each loop for varied interaction patterns |
| **Full Context Sharing** | Each agent sees complete conversation history |
| **Collaborative Prompting** | Agents build on each other's contributions |
| **Retry Mechanism** | Automatic retries with exponential backoff for reliability |

```
Loop 1: Agent B → Agent C → Agent A
Loop 2: Agent A → Agent B → Agent C  (shuffled)
Loop 3: Agent C → Agent A → Agent B  (shuffled)

Each agent sees full conversation history
```

---

## Step 1: Install and Import

```bash
pip install swarms
```

```python
from swarms import Agent, RoundRobinSwarm
```

---

## Step 2: Create Agents and Swarm

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

# Create the swarm
swarm = RoundRobinSwarm(
    agents=[researcher, analyst, writer],
    max_loops=2,  # 2 rounds of discussion
    verbose=True
)
```

---

## Step 3: Run the Swarm

```python
# Execute the task
result = swarm.run(
    task="Analyze the impact of remote work on productivity and team collaboration"
)

print(result)
```

---

## Complete Example

```python
from swarms import Agent, RoundRobinSwarm

# Create diverse agents
agents = [
    Agent(
        agent_name="Tech-Expert",
        system_prompt="Technology and implementation specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Business-Strategist",
        system_prompt="Business strategy and ROI specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="UX-Designer",
        system_prompt="User experience and design specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

# Initialize swarm
swarm = RoundRobinSwarm(
    agents=agents,
    max_loops=3,
    output_type="final",  # "final", "dict", "list"
    verbose=True
)

# Run task
result = swarm.run(
    "Design a mobile app feature for real-time team collaboration"
)

print("=" * 60)
print("RESULT:")
print("=" * 60)
print(result)
```

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agents` | Required | List of Agent instances |
| `max_loops` | `1` | Number of discussion rounds |
| `output_type` | `"final"` | Output format: "final", "dict", "list" |
| `verbose` | `False` | Enable detailed logging |
| `callback` | `None` | Function called after each loop |
| `max_retries` | `3` | Maximum retry attempts per agent |

---

## Batch Processing

```python
from swarms import Agent, RoundRobinSwarm

swarm = RoundRobinSwarm(
    agents=[agent1, agent2, agent3],
    max_loops=2
)

tasks = [
    "Evaluate option A for our product strategy",
    "Evaluate option B for our product strategy",
    "Compare options A and B",
]

results = swarm.run_batch(tasks)

for task, result in zip(tasks, results):
    print(f"\nTask: {task}")
    print(f"Result: {result[:150]}...")
```

---

## Use Cases

| Domain | Example |
|--------|---------|
| **Research** | Multiple agents contribute different research perspectives |
| **Strategy** | Team discusses and refines strategic decisions |
| **Design** | Iterative design review and refinement |
| **Analysis** | Multi-perspective analysis of complex topics |

---

## How It Works

1. **Initialization**: Task added to conversation history
2. **Loop Execution**:
   - Agents shuffled randomly for this loop
   - Each agent receives full conversation history
   - Collaborative prompt encourages building on others' insights
   - Agent contributes unique perspective
3. **Iteration**: Process repeats for max_loops rounds
4. **Result**: Full conversation history with all contributions

---

## Best Practices

- **Diverse Agents**: Create agents with different specializations
- **Appropriate Loops**: Use 2-3 loops for most collaborative tasks
- **Clear Roles**: Give each agent a distinct expertise area
- **Monitor Progress**: Use `verbose=True` during development

---

## Related Architectures

- [InteractiveGroupChat](../swarms/examples/igc_example.md) - Real-time group discussions
- [SequentialWorkflow](../swarms/examples/sequential_example.md) - Fixed agent order
- [MajorityVoting](./majority_voting_quickstart.md) - Consensus building

---

## Next Steps

- Explore [RoundRobinSwarm Tutorial](../swarms/examples/roundrobin_example.md)
- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/groupchat)
- Learn about [Multi-Agent Communication](../swarms/concept/multi_agent_communication.md)
