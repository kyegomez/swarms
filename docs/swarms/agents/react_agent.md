# React Agent

React Agent is the reasoning-and-action pattern used when an agent needs to plan, use tools, observe the result, and continue with the next step. It is useful for workflows where a single answer is not enough and the agent must build context from each action.

The core loop is:

1. Reason about the current task and available context.
2. Act by calling a tool, producing an intermediate answer, or requesting more information.
3. Observe the result of that action.
4. Repeat until the final answer is ready.

## When to Use ReAct

Use ReAct-style agents for tasks that require interaction with tools or iterative planning, such as research workflows, API lookups, data collection, troubleshooting, and multi-step analysis.

For simpler prompts where the agent can answer directly, start with the standard [Agent](../structs/agent.md) documentation. For broader comparisons across reasoning patterns, see the [Reasoning Agents Overview](reasoning_agents_overview.md).

## Related Guides

- [ReAct Agent Tutorial](../examples/react_agent_tutorial.md)
- [Reasoning Agent Router](reasoning_agent_router.md)
- [Iterative Agent](iterative_agent.md)
- [Agent with Tools](../examples/agent_with_tools.md)
- [Agent Memory](agent_memory.md)

## Practical Tips

- Keep the action space small and explicit so each step is easy to inspect.
- Use clear tool descriptions so the agent knows when each tool is appropriate.
- Store important observations in memory when later steps depend on them.
- Set loop limits to prevent open-ended execution on ambiguous tasks.
- Log intermediate steps during development so reasoning, actions, and observations can be reviewed.

## Minimal Pattern

```python
from swarms import Agent

agent = Agent(
    agent_name="Research-ReAct-Agent",
    system_prompt=(
        "Reason about the task, choose the next useful action, "
        "observe the result, and continue until the answer is complete."
    ),
    model_name="gpt-4o-mini",
    max_loops=3,
    react_on=True,
)

result = agent.run("Research the main risks in launching a new AI product.")
print(result)
```

Use this page as the quick route for ReAct concepts, then move into the tutorial or router documentation when you need implementation-specific details.
