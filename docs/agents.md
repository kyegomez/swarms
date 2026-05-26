# Agents

Agents are the core building block in Swarms. They combine a role, model configuration, prompts, tools, memory, and runtime settings into a reusable worker that can complete tasks on its own or inside a larger swarm.

This page preserves the top-level `agents` route used by existing documentation links and points users to the maintained agent docs.

## Start Here

- [Agent Reference](swarms/structs/agent.md)
- [Agents Index](swarms/agents/index.md)
- [Create Agents from YAML](swarms/agents/create_agents_yaml.md)
- [Agent Skills](swarms/agents/agent_skills.md)
- [Structured Outputs](swarms/agents/structured_outputs.md)

## Common Agent Workflows

- Create a single agent for focused task execution.
- Add tools when the agent needs external data or deterministic actions.
- Use structured outputs when the result must follow a schema.
- Move to a workflow or swarm when several agents need to collaborate.
- Use YAML when you want version-controlled, repeatable agent definitions.

## Related Examples

- [Basic Agent](swarms/examples/basic_agent.md)
- [Agent with Tools](swarms/examples/agent_with_tools.md)
- [Model Providers](swarms/examples/model_providers.md)
