# MajorityVoting: 3-Step Quickstart Guide

The MajorityVoting architecture enables multiple agents to independently evaluate a task, then synthesizes their responses through a consensus agent that evaluates, ranks, and provides a comprehensive final answer. This creates more robust and well-rounded solutions through diverse perspectives.

## Overview

| Feature | Description |
|---------|-------------|
| **Concurrent Execution** | Multiple agents run simultaneously for faster processing |
| **Consensus Building** | Dedicated consensus agent evaluates and synthesizes all responses |
| **Multi-Loop Support** | Iterative refinement over multiple rounds with agent memory |
| **Comprehensive Evaluation** | Assesses accuracy, depth, relevance, clarity, and unique insights |

```
Agent A ─┐
Agent B ─┼──> Concurrent Execution
Agent C ─┘
    │
    ▼
 Consensus Agent
    │
    ▼
Synthesized Final Answer
```

---

## Step 1: Install and Import

Ensure you have Swarms installed and import the MajorityVoting class:

```bash
pip install swarms
```

```python
from swarms import Agent, MajorityVoting
```

---

## Step 2: Create the Voting System

Create agents and initialize the MajorityVoting system:

```python
# Create specialized agents
agent1 = Agent(
    agent_name="Financial-Analysis-Agent-1",
    system_prompt="You are a conservative financial advisor focused on risk management and long-term stability.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Financial-Analysis-Agent-2",
    system_prompt="You are a growth-oriented financial advisor focused on high-potential opportunities.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent3 = Agent(
    agent_name="Financial-Analysis-Agent-3",
    system_prompt="You are a balanced financial advisor focused on diversification and risk-adjusted returns.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the majority voting system
voting_system = MajorityVoting(
    agents=[agent1, agent2, agent3],
    max_loops=1,
    verbose=True
)
```

---

## Step 3: Run the Voting System

Execute a task and get the consensus result:

```python
# Define the task
task = "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown."

# Run the voting system
result = voting_system.run(task=task)

# Print the result
print(result)
```

---

## Complete Example

Here's a complete working example:

```python
from swarms import Agent, MajorityVoting

# Step 1: Create specialized agents with different perspectives
agents = [
    Agent(
        agent_name="Conservative-Analyst",
        system_prompt="You are a conservative financial advisor focused on risk management and long-term stability.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Growth-Analyst",
        system_prompt="You are a growth-oriented financial advisor focused on high-potential opportunities.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Balanced-Analyst",
        system_prompt="You are a balanced financial advisor focused on diversification and risk-adjusted returns.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

# Step 2: Create the majority voting system
voting_system = MajorityVoting(
    agents=agents,
    max_loops=1,  # Single round of voting
    output_type="dict",  # Return as dictionary
    verbose=True
)

# Step 3: Run a task
task = "What are the top 3 AI investment opportunities for 2024?"
result = voting_system.run(task=task)

print("=" * 60)
print("VOTING RESULT:")
print("=" * 60)
print(result)
```

---

## Multi-Loop Consensus Building

Enable iterative refinement with multiple rounds:

```python
from swarms import Agent, MajorityVoting

# Create agents
agents = [
    Agent(agent_name="Analyst-1", system_prompt="Expert financial analyst", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Analyst-2", system_prompt="Expert market researcher", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Analyst-3", system_prompt="Expert risk assessor", model_name="gpt-4o-mini", max_loops=1),
]

# Multi-loop voting system
voting_system = MajorityVoting(
    agents=agents,
    max_loops=3,  # 3 rounds of iterative refinement
    verbose=True
)

# Each loop:
# 1. Agents independently analyze the task
# 2. Consensus agent evaluates and synthesizes responses
# 3. Agents see the consensus and refine their responses in next loop

result = voting_system.run("Analyze the future of quantum computing in enterprise applications")
print(result)
```

---

## Batch Processing

Process multiple tasks sequentially:

```python
from swarms import Agent, MajorityVoting

voting_system = MajorityVoting(
    agents=[agent1, agent2, agent3],
    max_loops=1
)

# Multiple tasks
tasks = [
    "What are the best cloud computing stocks?",
    "Should I invest in renewable energy ETFs?",
    "What's the outlook for semiconductor companies?",
]

# Process all tasks
results = voting_system.batch_run(tasks)

for task, result in zip(tasks, results):
    print(f"\nTask: {task}")
    print(f"Result: {result[:200]}...")
```

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agents` | Required | List of Agent instances to participate in voting |
| `max_loops` | `1` | Number of consensus rounds for iterative refinement |
| `output_type` | `"dict"` | Output format: "dict", "str", "list" |
| `verbose` | `False` | Enable detailed logging |
| `consensus_agent_model_name` | `"gpt-4.1"` | Model for the consensus agent |
| `consensus_agent_prompt` | Default | Custom system prompt for consensus agent |

### Output Types

| Value | Description |
|-------|-------------|
| `"dict"` | Conversation history as dictionary with roles and content |
| `"str"` | All messages formatted as a single string |
| `"list"` | Messages as a list of dictionaries |

---

## Use Cases

| Domain | Example Task |
|--------|---------------|
| **Financial Analysis** | "Compare investment strategies for retirement planning" |
| **Technical Decisions** | "Evaluate database options for a high-traffic application" |
| **Research** | "Synthesize findings on climate change mitigation strategies" |
| **Product Development** | "Analyze feature prioritization for our mobile app" |
| **Strategic Planning** | "Evaluate market entry strategies for Southeast Asia" |

---

## How It Works

1. **Concurrent Execution**: All agents run simultaneously on the same task
2. **Independent Analysis**: Each agent provides its own perspective based on its system prompt
3. **Consensus Evaluation**: The consensus agent:
   - Evaluates each agent's response on multiple dimensions
   - Compares and contrasts different viewpoints
   - Identifies the strongest arguments
   - Synthesizes a comprehensive final answer
4. **Iterative Refinement** (if max_loops > 1): Agents see the consensus and refine their responses

---

## Best Practices

- **Diverse Perspectives**: Create agents with different specializations or viewpoints
- **Clear Prompts**: Give each agent a distinct role and expertise area
- **Appropriate Loops**: Use `max_loops=1` for simple consensus, higher for complex iterative refinement
- **Output Format**: Use `output_type="dict"` to inspect individual agent responses
- **Consensus Quality**: The default consensus prompt is optimized for comprehensive evaluation

---

## Related Architectures

- [LLM Council](./llm_council_quickstart.md) - Council members rank each other's responses
- [CouncilAsAJudge](./council_as_judge_quickstart.md) - Multi-dimensional evaluation with specialized judges
- [DebateWithJudge](./debate_quickstart.md) - Two agents debate with judge synthesis

---

## Next Steps

- Explore [MajorityVoting Tutorial](../swarms/examples/majority_voting_example.md) for advanced examples
- See [Multi-Agent Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/majority_voting) for more use cases
- Learn about [Consensus Mechanisms](../swarms/concept/consensus_mechanisms.md) in multi-agent systems
