# Templates and Applications

Swarms examples include reusable templates for common agent and swarm application shapes. These examples are useful when you want a starting point rather than a blank file.

## Template Categories

| Category | Use for |
| --- | --- |
| Single agent templates | Basic task execution, model-provider setup, RAG, and tools |
| Multi-agent templates | Councils, hierarchical teams, majority voting, and graph workflows |
| AOP templates | Distributed agent services, client scripts, and discovery examples |
| Domain examples | Finance, medical, legal, product, growth, and research workflows |
| Deployment examples | API services, scheduled jobs, and cloud-oriented patterns |

## Recommended Flow

1. Start from the smallest example that matches your workflow.
2. Replace the prompt, model name, and API keys.
3. Run the example locally with a small task.
4. Add tests or smoke checks around your own inputs.
5. Move the workflow into a service or scheduled job only after it is stable.

## Useful Docs

- [Basic Agent](../swarms/examples/basic_agent.md)
- [Agent with Tools](../swarms/examples/agent_with_tools.md)
- [SequentialWorkflow](../swarms/structs/sequential_workflow.md)
- [GraphWorkflow](../swarms/structs/graph_workflow.md)
- [AOP Reference](../swarms/structs/aop.md)

Full examples source: <https://github.com/kyegomez/swarms/tree/master/examples>.
