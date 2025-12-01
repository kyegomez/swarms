# Debate Examples

This directory contains examples demonstrating debate patterns for multi-agent systems.

## Overview

Debate patterns enable agents to engage in structured discussions, present arguments, and reach conclusions through discourse. This pattern is useful for exploring multiple perspectives on complex topics and arriving at well-reasoned decisions.

## Examples

### DebateWithJudge

The `DebateWithJudge` architecture implements a debate system with self-refinement:

- **Agent A (Pro)** and **Agent B (Con)** present opposing arguments
- Both arguments are evaluated by a **Judge/Critic Agent**
- The Judge provides a winner or synthesis → refined answer
- The process repeats for N loops to progressively improve the answer

**Architecture Flow:**
```
Agent A (Pro) ↔ Agent B (Con)
      │            │
      ▼            ▼
   Judge / Critic Agent
      │
      ▼
Winner or synthesis → refined answer
```

**Initialization Options:**

The `DebateWithJudge` class supports three ways to configure agents:

1. **Preset Agents** (simplest): Use built-in optimized agents
2. **Agent List**: Provide a list of 3 agents `[pro, con, judge]`
3. **Individual Parameters**: Provide each agent separately

**Quick Start with Preset Agents:**
```python
from swarms import DebateWithJudge

# Create debate system with built-in agents (simplest approach)
debate = DebateWithJudge(
    preset_agents=True,
    max_loops=3,
    model_name="gpt-4o-mini"
)

# Run debate
result = debate.run("Should AI be regulated?")
```

**Using Agent List:**
```python
from swarms import Agent, DebateWithJudge

# Create your agents
agents = [pro_agent, con_agent, judge_agent]

# Create debate system with agent list
debate = DebateWithJudge(
    agents=agents,
    max_loops=3
)

result = debate.run("Should AI be regulated?")
```

**Using Individual Agent Parameters:**
```python
from swarms import Agent, DebateWithJudge

# Create Pro, Con, and Judge agents
pro_agent = Agent(agent_name="Pro-Agent", ...)
con_agent = Agent(agent_name="Con-Agent", ...)
judge_agent = Agent(agent_name="Judge-Agent", ...)

# Create debate system
debate = DebateWithJudge(
    pro_agent=pro_agent,
    con_agent=con_agent,
    judge_agent=judge_agent,
    max_loops=3
)

# Run debate
result = debate.run("Should AI be regulated?")
```

## Example Files

| File | Description |
|------|-------------|
| [debate_with_judge_example.py](./debate_with_judge_example.py) | Complete example showing all initialization methods |
| [policy_debate_example.py](./policy_debate_example.py) | Policy debate on AI regulation |
| [technical_architecture_debate_example.py](./technical_architecture_debate_example.py) | Technical architecture debate with batch processing |
| [business_strategy_debate_example.py](./business_strategy_debate_example.py) | Business strategy debate with conversation history |

