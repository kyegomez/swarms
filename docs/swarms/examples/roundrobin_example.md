# RoundRobinSwarm: Complete Guide

A comprehensive guide to using the RoundRobinSwarm architecture for collaborative multi-agent discussions with randomized turn order.

## Overview

The **RoundRobinSwarm** implements an AutoGen-style communication pattern where agents collaborate in randomized order across multiple discussion rounds. Each agent receives full conversation context and is encouraged to build upon others' contributions, creating rich collaborative refinement through varied interaction patterns.

| Feature | Description |
|---------|-------------|
| **Randomized Turn Order** | Agents shuffled each loop preventing predictable patterns |
| **Full Context Awareness** | Every agent sees complete conversation history |
| **Collaborative Prompting** | Built-in prompts encourage agents to acknowledge and extend others' work |
| **Automatic Retry Logic** | Exponential backoff retry mechanism for reliability |
| **Flexible Output Formats** | Support for "final", "dict", and "list" output types |

### When to Use RoundRobinSwarm

**Best For:**
- Collaborative discussions requiring multiple perspectives
- Iterative refinement through group deliberation
- Tasks benefiting from varied interaction patterns
- Situations where order dependency should be minimized

**Not Ideal For:**
- Tasks requiring strict sequential processing
- Simple queries with single-agent answers
- Workflows with fixed dependencies between steps

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Create Agents and Swarm

```python
from swarms import Agent, RoundRobinSwarm

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

### Step 2: Run the Swarm

```python
# Execute the task
result = swarm.run(
    task="Analyze the impact of remote work on productivity and team collaboration"
)

print(result)
```

---

## Basic Example

```python
from swarms import Agent, RoundRobinSwarm

# Create specialized agents
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

strategist = Agent(
    agent_name="Business-Strategist",
    system_prompt="You develop actionable strategies based on research and analysis.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Initialize swarm
swarm = RoundRobinSwarm(
    agents=[researcher, analyst, strategist],
    max_loops=2,
    output_type="final",
    verbose=True
)

# Run collaborative task
result = swarm.run(
    "Analyze market opportunities for AI-powered customer service solutions"
)

print(result)
```

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Agent]` | Required | Agents participating in the swarm |
| `max_loops` | `int` | `1` | Number of discussion rounds |
| `output_type` | `str` | `"final"` | Output format: "final", "dict", "list" |
| `verbose` | `bool` | `False` | Enable detailed logging |
| `callback` | `callable` | `None` | Function called after each loop |
| `max_retries` | `int` | `3` | Maximum retry attempts with exponential backoff |

---

## Advanced Examples

### Example 1: Product Design Review

```python
from swarms import Agent, RoundRobinSwarm

# Design team agents
agents = [
    Agent(
        agent_name="UX-Designer",
        system_prompt="You focus on user experience, usability, and user research.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Product-Manager",
        system_prompt="You balance business goals, user needs, and technical feasibility.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Engineer",
        system_prompt="You evaluate technical implementation, scalability, and maintainability.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Designer",
        system_prompt="You focus on visual design, brand consistency, and aesthetics.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

swarm = RoundRobinSwarm(
    agents=agents,
    max_loops=3,  # 3 rounds for thorough review
    verbose=True
)

result = swarm.run(
    "Review the proposed dashboard redesign focusing on data visualization and user workflows"
)
```

### Example 2: Strategic Planning

```python
from swarms import Agent, RoundRobinSwarm

# Strategy team
agents = [
    Agent(
        agent_name="Market-Analyst",
        system_prompt="Analyze market trends, competition, and opportunities.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Financial-Planner",
        system_prompt="Evaluate financial implications, ROI, and resource allocation.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Operations-Lead",
        system_prompt="Assess operational feasibility, execution plans, and timelines.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

swarm = RoundRobinSwarm(
    agents=agents,
    max_loops=2,
    output_type="dict"  # Get structured output
)

result = swarm.run(
    "Develop a go-to-market strategy for our new SaaS product targeting enterprise customers"
)
```

### Example 3: With Callback Function

```python
from swarms import Agent, RoundRobinSwarm

def progress_callback(loop_num, result):
    """Track progress after each loop"""
    print(f"\n[Loop {loop_num + 1} Complete]")
    print(f"Latest result preview: {result[:100]}...")

swarm = RoundRobinSwarm(
    agents=[agent1, agent2, agent3],
    max_loops=3,
    callback=progress_callback,
    verbose=True
)

result = swarm.run("Complex multi-round task...")
```

---

## Use Cases

### Use Case 1: Research Synthesis

```python
agents = [
    Agent(agent_name="Medical-Researcher", system_prompt="Clinical research expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Data-Scientist", system_prompt="Statistical analysis expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Literature-Reviewer", system_prompt="Research synthesis expert", model_name="gpt-4o-mini", max_loops=1),
]

swarm = RoundRobinSwarm(agents=agents, max_loops=2)
result = swarm.run("Synthesize current research on effectiveness of immunotherapy for melanoma")
```

### Use Case 2: Code Review

```python
agents = [
    Agent(agent_name="Security-Reviewer", system_prompt="Security and vulnerabilities expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Performance-Reviewer", system_prompt="Performance optimization expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Architecture-Reviewer", system_prompt="Code architecture and patterns expert", model_name="gpt-4o-mini", max_loops=1),
]

swarm = RoundRobinSwarm(agents=agents, max_loops=2)
result = swarm.run("Review the authentication middleware implementation in auth.py")
```

### Use Case 3: Content Creation

```python
agents = [
    Agent(agent_name="Researcher", system_prompt="Research and fact-checking specialist", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Writer", system_prompt="Content writing and storytelling specialist", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Editor", system_prompt="Editing, clarity, and refinement specialist", model_name="gpt-4o-mini", max_loops=1),
]

swarm = RoundRobinSwarm(agents=agents, max_loops=2)
result = swarm.run("Create a comprehensive guide on using AI for customer service automation")
```

---

## Best Practices

1. **Agent Diversity**: Create agents with truly distinct roles and perspectives
2. **Loop Count**: Use 2-3 loops for most tasks; more loops for complex discussions
3. **Clear Roles**: Give each agent a specific area of expertise
4. **Verbose Mode**: Enable during development to see interaction patterns
5. **Callback Usage**: Use callbacks for progress tracking in long-running tasks

---

## Common Patterns

### Pattern 1: Multidisciplinary Team

```python
# Create a well-rounded team
team = [
    Agent(agent_name="Technical", system_prompt="Technical expert..."),
    Agent(agent_name="Business", system_prompt="Business expert..."),
    Agent(agent_name="Creative", system_prompt="Creative expert..."),
    Agent(agent_name="User-Focused", system_prompt="User advocate..."),
]

swarm = RoundRobinSwarm(agents=team, max_loops=2)
```

### Pattern 2: Iterative Refinement

```python
# Higher loops for refinement
swarm = RoundRobinSwarm(
    agents=[expert1, expert2, expert3],
    max_loops=4,  # More rounds for deeper refinement
)
```

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[SequentialWorkflow](./sequential_example.md)** | When tasks must flow in fixed order |
| **[InteractiveGroupChat](./igc_example.md)** | For real-time interactive discussions |
| **[MajorityVoting](./majority_voting_example.md)** | When you need consensus via voting |
| **[GroupChat](./groupchat_example.md)** | For expertise-based speaker selection |

---

## Next Steps

- Explore [RoundRobinSwarm Quickstart](../../examples/roundrobin_quickstart.md)
- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/groupchat)
- Learn about [Group Communication Patterns](../../examples/multi_agent_architectures_overview.md)
