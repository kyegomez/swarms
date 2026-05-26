# Agent Design Principles

Good agent design starts with a narrow job, clear context, and predictable outputs. In Swarms, an agent can be simple enough for a single task or specialized enough to participate in a larger multi-agent workflow.

Use this guide when planning an agent before moving into the [Agent](../structs/agent.md) API or a production swarm architecture.

## Start with the Role

Define the agent's responsibility in one sentence. A strong role explains what the agent owns and what it should avoid.

Examples:

- "Summarize customer support tickets and extract escalation risks."
- "Review Python code for correctness and missing tests."
- "Research market signals and return cited findings."

Avoid broad roles such as "general assistant" when the agent is part of a workflow. Narrow roles are easier to evaluate and easier to compose with other agents.

## Write a Clear System Prompt

The system prompt should cover:

- The agent's role.
- The expected output format.
- Domain rules or constraints.
- How to handle uncertainty.
- What information should be preserved for downstream agents.

For structured results, pair the prompt with the [Structured Outputs](../agents/structured_outputs.md) guide so the response is easier to parse and validate.

## Choose Tools Intentionally

Only attach tools the agent actually needs. Each tool should have a clear name, description, and expected input shape. Too many tools can make the agent slower and harder to debug.

For tool setup patterns, see [Tools](../tools/main.md) and [Agent with Tools](../examples/agent_with_tools.md).

## Set Loop and Memory Boundaries

Use `max_loops` to control how much autonomy the agent has. Short tasks usually need one or two loops, while research or planning workflows may need more.

Add memory only when the agent needs to preserve state across steps or runs. For temporary task context, keep the prompt and input concise instead of storing everything.

## Design for Evaluation

Before deploying an agent, decide how success will be checked:

- Does the output match the requested schema?
- Does it cite or preserve important evidence?
- Does it avoid unsupported claims?
- Can another agent or human reviewer act on the result?
- Does it fail safely when information is missing?

## Related Guides

- [Agent](../structs/agent.md)
- [Create Agents from YAML](../agents/create_agents_yaml.md)
- [Structured Outputs](../agents/structured_outputs.md)
- [Agent Memory](../agents/agent_memory.md)
- [Basic Agent Example](../examples/basic_agent.md)
- [Auto Swarm Builder Example](../examples/auto_swarm_builder_example.md)

Good agents are small, inspectable building blocks. Design each one around a clear responsibility, then compose them into swarms when the task genuinely needs multiple perspectives or stages.
