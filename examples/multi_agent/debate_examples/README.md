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
- The process repeats for N rounds to progressively improve the answer

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

**Example Usage:**
```python
from swarms import Agent
from swarms.structs.debate_with_judge import DebateWithJudge

# Create Pro, Con, and Judge agents
pro_agent = Agent(agent_name="Pro-Agent", ...)
con_agent = Agent(agent_name="Con-Agent", ...)
judge_agent = Agent(agent_name="Judge-Agent", ...)

# Create debate system
debate = DebateWithJudge(
    pro_agent=pro_agent,
    con_agent=con_agent,
    judge_agent=judge_agent,
    max_rounds=3
)

# Run debate
result = debate.run("Should AI be regulated?")
```

See [debate_with_judge_example.py](./debate_with_judge_example.py) for a complete example.

