# Dynamic Conversational Swarm

Dynamic Conversational Swarm is a multi-agent conversation pattern where agents participate in a flexible discussion rather than following a fixed linear route. The architecture docs link to this route for users exploring adaptive conversation-based coordination.

Use this page as a pattern reference. Current examples show this style through experimental conversation workflows and structured-output demonstrations.

## When to Use

- Open-ended multi-agent discussion
- Adaptive speaker selection
- Exploratory reasoning across several agent perspectives
- Group review before a final answer
- Conversation workflows where the next speaker depends on prior context

## Pattern Shape

A dynamic conversational swarm usually includes:

- A set of agents with distinct roles
- Shared conversation state
- A speaker-selection or routing policy
- A completion rule
- A final summarizer or judge

## Design Tips

- Give each agent a clear role and decision boundary.
- Keep the number of agents small until the conversation quality is stable.
- Use structured outputs when the final response must be machine-readable.
- Add a maximum turn count to avoid open-ended loops.
- Capture the final decision separately from intermediate discussion.

## Related Docs

- [GroupChat](group_chat.md)
- [AgentRearrange](agent_rearrange.md)
- [Swarm Architectures](../concept/swarm_architectures.md)

## Source Example

- `examples/single_agent/tools/structured_outputs/example_meaning_of_life_agents.py`
