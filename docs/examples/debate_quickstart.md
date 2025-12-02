# DebateWithJudge: 3-Step Quickstart Guide

The DebateWithJudge architecture enables structured debates between two agents (Pro and Con) with a Judge providing refined synthesis over multiple rounds. This creates progressively improved answers through iterative argumentation and evaluation.

## Overview

| Feature | Description |
|---------|-------------|
| **Pro Agent** | Argues in favor of a position with evidence and reasoning |
| **Con Agent** | Presents counter-arguments and identifies weaknesses |
| **Judge Agent** | Evaluates both sides and synthesizes the best elements |
| **Iterative Refinement** | Multiple rounds progressively improve the final answer |

```
Agent A (Pro) ↔ Agent B (Con)
      │            │
      ▼            ▼
   Judge / Critic Agent
      │
      ▼
Winner or synthesis → refined answer
```

---

## Step 1: Install and Import

Ensure you have Swarms installed and import the DebateWithJudge class:

```bash
pip install swarms
```

```python
from swarms import DebateWithJudge
```

---

## Step 2: Create the Debate System

Create a DebateWithJudge system using preset agents (the simplest approach):

```python
# Create debate system with preset optimized agents
debate = DebateWithJudge(
    preset_agents=True,      # Use built-in optimized agents
    max_loops=3,             # 3 rounds of debate
    model_name="gpt-4o-mini",
    verbose=True
)
```

---

## Step 3: Run the Debate

Execute the debate on a topic:

```python
# Define the debate topic
topic = "Should artificial intelligence be regulated by governments?"

# Run the debate
result = debate.run(task=topic)

# Print the refined answer
print(result)

# Or get just the final synthesis
final_answer = debate.get_final_answer()
print(final_answer)
```

---

## Complete Example

Here's a complete working example:

```python
from swarms import DebateWithJudge

# Step 1: Create the debate system with preset agents
debate_system = DebateWithJudge(
    preset_agents=True,
    max_loops=3,
    model_name="gpt-4o-mini",
    output_type="str-all-except-first",
    verbose=True,
)

# Step 2: Define a complex topic
topic = (
    "Should artificial intelligence be regulated by governments? "
    "Discuss the balance between innovation and safety."
)

# Step 3: Run the debate and get refined answer
result = debate_system.run(task=topic)

print("=" * 60)
print("DEBATE RESULT:")
print("=" * 60)
print(result)

# Access conversation history for detailed analysis
history = debate_system.get_conversation_history()
print(f"\nTotal exchanges: {len(history)}")
```

---

## Custom Agents Example

Create specialized agents for domain-specific debates:

```python
from swarms import Agent, DebateWithJudge

# Create specialized Pro agent
pro_agent = Agent(
    agent_name="Innovation-Advocate",
    system_prompt=(
        "You are a technology policy expert arguing for innovation and minimal regulation. "
        "You present arguments focusing on economic growth, technological competitiveness, "
        "and the risks of over-regulation stifling progress."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create specialized Con agent
con_agent = Agent(
    agent_name="Safety-Advocate",
    system_prompt=(
        "You are a technology policy expert arguing for strong AI safety regulations. "
        "You present arguments focusing on public safety, ethical considerations, "
        "and the need for government oversight of powerful technologies."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create specialized Judge agent
judge_agent = Agent(
    agent_name="Policy-Analyst",
    system_prompt=(
        "You are an impartial policy analyst evaluating technology regulation debates. "
        "You synthesize the strongest arguments from both sides and provide "
        "balanced, actionable policy recommendations."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create debate system with custom agents
debate = DebateWithJudge(
    agents=[pro_agent, con_agent, judge_agent],  # Pass as list
    max_loops=3,
    verbose=True,
)

result = debate.run("Should AI-generated content require mandatory disclosure labels?")
```

---

## Batch Processing

Process multiple debate topics:

```python
from swarms import DebateWithJudge

debate = DebateWithJudge(preset_agents=True, max_loops=2)

# Multiple topics to debate
topics = [
    "Should remote work become the standard for knowledge workers?",
    "Is cryptocurrency a viable alternative to traditional banking?",
    "Should social media platforms be held accountable for content moderation?",
]

# Process all topics
results = debate.batched_run(topics)

for topic, result in zip(topics, results):
    print(f"\nTopic: {topic}")
    print(f"Result: {result[:200]}...")
```

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preset_agents` | `False` | Use built-in optimized agents |
| `max_loops` | `3` | Number of debate rounds |
| `model_name` | `"gpt-4o-mini"` | Model for preset agents |
| `output_type` | `"str-all-except-first"` | Output format |
| `verbose` | `True` | Enable detailed logging |

### Output Types

| Value | Description |
|-------|-------------|
| `"str-all-except-first"` | Formatted string, excluding initialization (default) |
| `"str"` | All messages as formatted string |
| `"dict"` | Messages as dictionary |
| `"list"` | Messages as list |

---

## Use Cases

| Domain | Example Topic |
|--------|---------------|
| **Policy** | "Should universal basic income be implemented?" |
| **Technology** | "Microservices vs. monolithic architecture for startups?" |
| **Business** | "Should companies prioritize growth or profitability?" |
| **Ethics** | "Is it ethical to use AI in hiring decisions?" |
| **Science** | "Should gene editing be allowed for non-medical purposes?" |

---

## Next Steps

- Explore [DebateWithJudge Reference](../swarms/structs/debate_with_judge.md) for complete API details
- See [Debate Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/debate_examples) for more use cases
- Learn about [Orchestration Methods](../swarms/structs/orchestration_methods.md) for other debate architectures

