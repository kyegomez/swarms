# RoundRobinSwarm Quickstart

`RoundRobinSwarm` lets agents take turns contributing to a task. It is useful when you want several specialists to iteratively refine an answer without assigning one fixed leader.

For the full tutorial, see [RoundRobin Example](../swarms/examples/roundrobin_example.md).

## Minimal Example

```python
from swarms import Agent, RoundRobinSwarm

planner = Agent(
    agent_name="Planner",
    system_prompt="Break the task into clear steps.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

reviewer = Agent(
    agent_name="Reviewer",
    system_prompt="Review the plan and point out missing risks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

swarm = RoundRobinSwarm(
    agents=[planner, reviewer],
    max_loops=2,
)

result = swarm.run("Create a launch checklist for a new developer tool.")
print(result)
```

## When to Use It

Use RoundRobinSwarm when:

- You want each agent to contribute in turn.
- The task benefits from iterative refinement.
- You want a lightweight discussion pattern without a judge or voting step.
- Agent order should be easy to inspect during development.

## Related Guides

- [RoundRobin Example](../swarms/examples/roundrobin_example.md)
- [GroupChat Examples](../swarms/examples/groupchat_comprehensive_examples.md)
- [Sequential Example](../swarms/examples/sequential_example.md)
- [Swarm Architectures](../swarms/concept/swarm_architectures.md)
