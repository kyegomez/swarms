# MAKER

MAKER stands for **M**aximal **A**gentic decomposition, first-to-ahead-by-**K** error correction, and **R**ed-flagging. The Swarms architecture docs describe it as a long-horizon reasoning pattern where a task is split into many atomic steps and each step is validated before the workflow commits to a result.

This page preserves the documentation route used by the structs overview and swarm architecture pages. The historical MAKER struct is not part of the current maintained structs surface, so new projects should treat this page as a pattern reference rather than an active API reference.

## Pattern Summary

MAKER is useful when a workflow needs repeated agreement before moving forward:

- Decompose the task into small steps.
- Ask one-shot micro-agents to solve the current step.
- Parse and validate each answer.
- Discard answers that fail red-flag checks.
- Commit only when one candidate is ahead by `k` votes.
- Carry the accepted result into the next step.

The approach prioritizes reliability over speed. It is most appropriate for long or fragile workflows where a single bad intermediate answer can damage the final result.

## When to Use the Pattern

- High-precision research pipelines
- Multi-step analysis with strict validation gates
- Workflows where each intermediate result should be independently checked
- Tasks that benefit from sampling multiple candidate answers before committing

## Current Alternatives

For new Swarms projects, prefer maintained workflow primitives:

- [GraphWorkflow](graph_workflow.md) for explicit directed task graphs
- [PlannerWorkerSwarm](planner_worker_swarm.md) for planner-worker task decomposition
- [SequentialWorkflow](sequential_workflow.md) for ordered step execution
- [Swarm Router](swarm_router.md) for choosing an architecture at runtime

## Related Docs

- [Swarm Architectures](../concept/swarm_architectures.md)
- [Structs Overview](overview.md)
