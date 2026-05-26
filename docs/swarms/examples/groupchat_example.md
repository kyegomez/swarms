# GroupChat Example

`GroupChat` coordinates multiple agents in a shared conversation. It is useful when agents need to discuss a task, respond to one another, and build toward a shared answer.

For the full walkthrough, see [GroupChat Comprehensive Examples](groupchat_comprehensive_examples.md). This page exists as a short route for docs that link to a basic GroupChat example.

## Minimal Pattern

```python
from swarms import Agent, GroupChat

researcher = Agent(
    agent_name="Researcher",
    system_prompt="Find concise evidence and useful facts.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

reviewer = Agent(
    agent_name="Reviewer",
    system_prompt="Review the research and identify gaps or risks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

group_chat = GroupChat(
    name="Research-Review-Chat",
    agents=[researcher, reviewer],
    max_loops=2,
)

result = group_chat.run("Evaluate whether this release plan is ready for launch.")
print(result)
```

## When to Use GroupChat

Use GroupChat when:

- Agents need to see and respond to each other's messages.
- The task benefits from collaborative discussion.
- You want a conversational record of intermediate reasoning.
- The final answer depends on several specialized viewpoints.

## Related Guides

- [GroupChat Comprehensive Examples](groupchat_comprehensive_examples.md)
- [RoundRobin Example](roundrobin_example.md)
- [Council as Judge Example](council_as_judge_example.md)
- [Swarm Architectures](../concept/swarm_architectures.md)
