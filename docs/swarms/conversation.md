# Conversation

Conversation is the message-history and memory container used by Swarms agents and workflows. It helps store turns, serialize context, save state, and support longer-running agent interactions.

This page preserves the `swarms/conversation` route used by existing documentation links and points users to the maintained reference docs.

## What Conversation Handles

- Agent and user message history
- System prompts and context
- Serialization to files or structured formats
- Persistent memory patterns
- Conversation inspection during debugging

## Start Here

- [Conversation Reference](structs/conversation.md)
- [Agent Memory](agents/agent_memory.md)
- [Agent Reference](structs/agent.md)
- [Custom Swarm](structs/custom_swarm.md)

## Usage Notes

Use conversation persistence when an agent needs continuity across turns or process restarts. Keep sensitive data handling in mind when saving conversation history, and avoid storing secrets in prompts, tool outputs, or serialized logs.
