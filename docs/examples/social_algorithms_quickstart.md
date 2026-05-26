# Social Algorithms Quickstart

Social algorithms define reusable communication patterns for multi-agent systems. They are helpful when you want agents to negotiate, review, vote, brainstorm, or coordinate with a custom protocol.

For the full walkthrough, see [Social Algorithms Example](../swarms/examples/social_algorithms_example.md).

## Minimal Pattern

Start by choosing the communication behavior before you choose the number of agents. A clear pattern makes the swarm easier to debug.

```python
from swarms import Agent

analyst = Agent(
    agent_name="Analyst",
    system_prompt="Analyze the task and provide practical findings.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

critic = Agent(
    agent_name="Critic",
    system_prompt="Review the findings and identify weaknesses.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

task = "Evaluate the risks in adopting a new workflow automation tool."

analysis = analyst.run(task)
review = critic.run(f"Review this analysis and identify gaps:\n\n{analysis}")

print(review)
```

This two-agent review loop is the simplest version of a social algorithm: one agent produces work, and another agent responds according to a defined role.

## Common Patterns

- **Peer review**: one agent drafts, another critiques.
- **Consensus**: several agents answer, then one response is selected.
- **Negotiation**: agents represent different constraints or stakeholders.
- **Brainstorming**: agents generate options before a final selection step.
- **Hierarchical review**: a coordinator assigns work and validates results.

## Related Guides

- [Social Algorithms Example](../swarms/examples/social_algorithms_example.md)
- [RoundRobin Example](../swarms/examples/roundrobin_example.md)
- [Council as Judge Example](../swarms/examples/council_as_judge_example.md)
- [Majority Voting Example](../swarms/examples/majority_voting_example.md)
- [Swarm Architectures](../swarms/concept/swarm_architectures.md)
