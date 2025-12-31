# SocialAlgorithms: 3-Step Quickstart Guide

The SocialAlgorithms architecture provides a flexible framework for defining custom communication patterns between agents. Upload any arbitrary algorithm as a callable that specifies how agents interact, in what order, and what information is shared - enabling complete control over multi-agent coordination.

## Overview

| Feature | Description |
|---------|-------------|
| **Custom Algorithms** | Define any communication pattern as a Python function |
| **Flexible Coordination** | Complete control over agent interaction sequences |
| **Communication Logging** | Optional tracking of all agent-to-agent messages |
| **Timeout Protection** | Configurable execution time limits |

```
Your Custom Algorithm
        │
        ▼
┌───────────────────────┐
│ Agent Communication   │
│  - You define order   │
│  - You define flow    │
│  - You define logic   │
└───────────────────────┘
        │
        ▼
    Results
```

---

## Step 1: Install and Import

```bash
pip install swarms
```

```python
from swarms import Agent, SocialAlgorithms
```

---

## Step 2: Define Your Algorithm

```python
def research_analysis_synthesis(agents, task, **kwargs):
    """
    Custom algorithm: Research → Analysis → Synthesis
    """
    # Agent 0: Research
    research = agents[0].run(f"Research the topic: {task}")

    # Agent 1: Analyze the research
    analysis = agents[1].run(f"Analyze this research: {research}")

    # Agent 2: Synthesize findings
    synthesis = agents[2].run(
        f"Synthesize these findings:\nResearch: {research}\nAnalysis: {analysis}"
    )

    return {
        "research": research,
        "analysis": analysis,
        "synthesis": synthesis
    }
```

---

## Step 3: Create and Run

```python
# Create agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist. Gather comprehensive information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analyst. Interpret data and identify insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

synthesizer = Agent(
    agent_name="Synthesizer",
    system_prompt="You synthesize information into clear recommendations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create social algorithm
social_alg = SocialAlgorithms(
    name="Research-Analysis-Synthesis",
    agents=[researcher, analyst, synthesizer],
    social_algorithm=research_analysis_synthesis,
    verbose=True
)

# Run
result = social_alg.run("The impact of AI on healthcare")
print(result.final_outputs)
```

---

## Complete Example

```python
from swarms import Agent, SocialAlgorithms

# Define consensus-building algorithm
def consensus_algorithm(agents, task, **kwargs):
    """Agents vote and reach consensus"""
    votes = []

    # Each agent provides their vote
    for agent in agents:
        vote = agent.run(f"{task}\n\nProvide your recommendation (A, B, or C):")
        votes.append(vote)

    # Final agent synthesizes consensus
    consensus_agent = agents[-1]
    consensus = consensus_agent.run(
        f"Based on these votes: {votes}\nWhat is the consensus?"
    )

    return {
        "votes": votes,
        "consensus": consensus
    }

# Create agents
agents = [
    Agent(agent_name=f"Voter-{i}", system_prompt="You are an expert evaluator", model_name="gpt-4o-mini", max_loops=1)
    for i in range(4)
]

# Run algorithm
social_alg = SocialAlgorithms(
    name="Consensus-Builder",
    agents=agents,
    social_algorithm=consensus_algorithm,
)

result = social_alg.run("Should we launch Product A, B, or C first?")
print(result.final_outputs)
```

---

## Built-in Algorithm Examples

The framework supports any pattern. Common examples:

### Sequential Pipeline

```python
def sequential_pipeline(agents, task, **kwargs):
    result = task
    for agent in agents:
        result = agent.run(result)
    return result
```

### Parallel Then Aggregate

```python
def parallel_aggregate(agents, task, **kwargs):
    # All agents work in parallel
    results = [agent.run(task) for agent in agents[:-1]]

    # Last agent aggregates
    aggregate = agents[-1].run(f"Synthesize: {results}")
    return {"individual": results, "aggregate": aggregate}
```

### Debate

```python
def debate_algorithm(agents, task, **kwargs):
    rounds = kwargs.get("rounds", 3)
    discussion = []

    for round_num in range(rounds):
        for agent in agents:
            context = "\n".join(discussion)
            response = agent.run(f"{task}\n\nPrevious discussion:\n{context}")
            discussion.append(f"{agent.agent_name}: {response}")

    return {"discussion": discussion}
```

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agents` | Required | List of Agent instances |
| `social_algorithm` | Required | Callable defining communication pattern |
| `max_execution_time` | `300.0` | Timeout in seconds |
| `output_type` | `"dict"` | Output format |
| `verbose` | `False` | Enable logging |
| `enable_communication_logging` | `False` | Track all agent messages |

---

## Communication Logging

```python
social_alg = SocialAlgorithms(
    name="My-Algorithm",
    agents=agents,
    social_algorithm=my_algorithm,
    enable_communication_logging=True,  # Track all communications
    verbose=True
)

result = social_alg.run("Task...")

# Access communication history
history = social_alg.get_communication_history()
for step in history:
    print(f"{step.sender_agent} → {step.receiver_agent}: {step.message[:50]}...")
```

---

## Use Cases

| Pattern | Algorithm Type |
|---------|----------------|
| **Research Pipeline** | Sequential: Research → Analysis → Synthesis |
| **Peer Review** | Parallel evaluation then consensus |
| **Negotiation** | Iterative back-and-forth between agents |
| **Hierarchical** | Manager delegates to workers, reviews results |
| **Voting** | All agents vote, one agent tallies |

---

## Best Practices

- **Clear Algorithm Logic**: Make communication patterns explicit and easy to follow
- **Handle Errors**: Wrap agent calls in try/except for robustness
- **Return Structured Data**: Use dictionaries for complex results
- **Use Timeouts**: Set appropriate `max_execution_time` for safety
- **Enable Logging**: Use `verbose=True` during development

---

## Related Architectures

- [SequentialWorkflow](../swarms/examples/sequential_example.md) - Pre-built sequential pattern
- [ConcurrentWorkflow](../swarms/examples/concurrent_workflow.md) - Pre-built parallel pattern
- [RoundRobinSwarm](./roundrobin_quickstart.md) - Randomized turn-taking

---

## Next Steps

- Explore [SocialAlgorithms Tutorial](../swarms/examples/social_algorithms_example.md)
- See [12+ Algorithm Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/social_algorithms_examples)
- Learn about [Custom Communication Patterns](../swarms/concept/social_algorithms.md)
