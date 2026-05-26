# DebateWithJudge Quickstart

`DebateWithJudge` runs a structured debate between a pro agent, a con agent, and a judge agent. Use it when a task benefits from opposing viewpoints before a final answer is selected.

For the full API reference, see [DebateWithJudge](../structs/debate_with_judge.md).

## Quick Example

The fastest way to start is with the built-in preset agents.

```python
from swarms import DebateWithJudge

debate = DebateWithJudge(
    preset_agents=True,
    max_loops=3,
)

result = debate.run(
    "Should a startup use a multi-agent system for customer support triage?"
)

print(result)
```

The debate runs through multiple rounds, gathers arguments from both sides, and lets the judge synthesize the final answer.

## When to Use Debate

Use debate workflows when:

- A decision has meaningful tradeoffs.
- You want explicit arguments for and against an option.
- A reviewer should compare competing positions.
- The final answer should include a reasoned synthesis.

For simpler agreement-based workflows, see [Majority Voting](majority_voting_example.md). For broader architecture patterns, see [Swarm Architectures](../concept/swarm_architectures.md).

## Custom Agents

Use custom agents when the debate needs domain-specific roles.

```python
from swarms import Agent, DebateWithJudge

pro_agent = Agent(
    agent_name="Pro-Agent",
    system_prompt="Argue in favor of the proposal with practical evidence.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

con_agent = Agent(
    agent_name="Con-Agent",
    system_prompt="Argue against the proposal and identify risks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

judge_agent = Agent(
    agent_name="Judge-Agent",
    system_prompt="Evaluate both arguments and return a balanced final recommendation.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

debate = DebateWithJudge(
    agents=[pro_agent, con_agent, judge_agent],
    max_loops=2,
)

answer = debate.run("Should we migrate this service to microservices?")
print(answer)
```

## Related Guides

- [DebateWithJudge](../structs/debate_with_judge.md)
- [Council as Judge Example](council_as_judge_example.md)
- [Majority Voting Example](majority_voting_example.md)
- [Swarm Router](swarm_router.md)
- [Swarm Architectures](../concept/swarm_architectures.md)
