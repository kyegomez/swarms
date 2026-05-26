# Interactive Group Chat

Interactive Group Chat is a collaboration pattern for multi-agent discussions where speaker selection, moderation, and turn-taking matter. The architecture docs link to this route for users who want a deeper explanation of the pattern.

Use this page as a route and pattern reference. For runnable source examples, see the group chat examples in the repository.

## When to Use

- Multi-agent brainstorming
- Panel discussions
- Review workflows where agents respond to each other
- Moderated debates or expert conversations
- Human-in-the-loop group coordination

## Pattern Shape

An interactive group chat usually includes:

- A moderator or coordinator
- Two or more specialized agents
- A turn-selection policy
- A shared task or discussion goal
- A stopping condition or final-summary step

## Related Docs

- [GroupChat](group_chat.md)
- [Swarm Architectures](../concept/swarm_architectures.md)
- [AgentRearrange](agent_rearrange.md)
- [Multi-Agent Collaboration](../concept/swarm_architectures.md)

## Source Examples

- `examples/multi_agent/groupchat/enhanced_collaboration_example.py`
- `examples/multi_agent/groupchat/random_dynamic_speaker_example.py`
- `examples/multi_agent/groupchat/speaker_function_examples.py`
- `examples/multi_agent/groupchat/README.md`
