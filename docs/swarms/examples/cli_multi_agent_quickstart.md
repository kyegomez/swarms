# CLI Multi-Agent Quickstart

This quickstart shows the fastest path from a single CLI prompt to a repeatable multi-agent workflow. Use it when you want to run Swarms from the command line without writing a full Python script first.

For the base CLI setup, start with the [CLI Quickstart](../cli/cli_quickstart.md). For YAML configuration details, see the [CLI YAML Guide](../cli/cli_yaml_guide.md).

## 1. Create a YAML Configuration

Create a YAML file that defines each agent's role and the workflow they should run in.

```yaml
name: research-review-swarm
description: Research a topic, review the findings, and produce a concise final answer.

agents:
  - agent_name: Research-Agent
    system_prompt: >
      Research the task carefully and return concrete findings with source notes.
    model_name: gpt-4o-mini
    max_loops: 1

  - agent_name: Review-Agent
    system_prompt: >
      Review the research for gaps, contradictions, and missing evidence.
    model_name: gpt-4o-mini
    max_loops: 1

swarm:
  name: Research Review Swarm
  description: A two-agent workflow for research and review.
  swarm_type: SequentialWorkflow
  task: "Summarize the risks and benefits of using agent swarms for customer support."
```

## 2. Run the Workflow

Use the Swarms CLI command for YAML-based execution from your local environment.

```bash
swarms run path/to/research-review-swarm.yaml
```

The CLI loads the agents, runs the configured workflow, and prints the final result in your terminal.

## 3. Iterate on Agent Roles

Keep each agent focused:

- One agent gathers or transforms information.
- One agent reviews, critiques, or verifies.
- One agent writes the final output when the workflow needs a polished answer.

If the task needs a different architecture, compare options in [Swarm Router](swarm_router.md) and [Swarm Architectures](../concept/swarm_architectures.md).

## 4. Move to Python When Needed

The CLI is a good starting point for repeatable runs. Move to Python when you need custom tools, application integration, custom storage, or richer control over the workflow.

Useful next pages:

- [Create Agents from YAML](../agents/create_agents_yaml.md)
- [CLI Agent Guide](../cli/cli_agent_guide.md)
- [CLI Heavy Swarm Guide](../cli/cli_heavy_swarm_guide.md)
- [CLI LLM Council Guide](../cli/cli_llm_council_guide.md)
- [Basic Agent Example](basic_agent.md)
