# ConcurrentWorkflow

This page preserves the `concurrent_workflow` route used by several internal documentation links. The canonical reference page is [ConcurrentWorkflow Documentation](concurrentworkflow.md).

`ConcurrentWorkflow` runs multiple agents in parallel and collects their outputs. Use it when the work can be split into independent subtasks that do not need step-by-step coordination.

## Good Fit

- Batch analysis
- Parallel research
- Independent data processing
- Competitive evaluations across multiple agents
- Browser or tool workflows that can run side by side

## Quick Example

```python
from swarms import Agent, ConcurrentWorkflow


agents = [
    Agent(agent_name="Researcher", model_name="gpt-4.1", max_loops=1),
    Agent(agent_name="Reviewer", model_name="gpt-4.1", max_loops=1),
]

workflow = ConcurrentWorkflow(
    name="parallel-review",
    agents=agents,
)

result = workflow.run("Analyze this proposal from two independent perspectives.")
print(result)
```

## Related Docs

- [Canonical ConcurrentWorkflow Reference](concurrentworkflow.md)
- [ConcurrentWorkflow Examples](../examples/concurrent_workflow.md)
- [Swarm Router](swarm_router.md)
