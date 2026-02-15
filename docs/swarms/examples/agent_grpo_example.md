# Agent GRPO Tutorial

End-to-end tutorial for `swarms.structs.agent_grpo`.

## Prerequisites

- Python 3.10+
- `pip install -U swarms`

## Example

```python
from swarms import Agent
from swarms.structs.agent_grpo import AgenticGRPO

solver = Agent(agent_name="Math-Solver", model_name="gpt-4.1", max_loops=1)
grpo = AgenticGRPO(
    name="math-grpo",
    description="Sample + score",
    agent=solver,
    n=4,
    correct_answers=["42"],
)
print(grpo.run(task="What is 40 + 2?"))
```

## What this demonstrates

- Correct import and initialization flow for `agent_grpo`
- Minimal execution path suitable for first integration tests
- A baseline pattern to adapt for production use

## Related

- Struct reference: [`swarms/structs/agent_grpo.md`](../structs/agent_grpo.md)
