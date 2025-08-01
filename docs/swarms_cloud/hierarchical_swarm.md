# HierarchicalSwarm

*Implements structured, multi-level task management with clear authority*

**Swarm Type**: `HierarchicalSwarm`

## Overview

The HierarchicalSwarm implements a structured, multi-level approach to task management with clear lines of authority and delegation. This architecture organizes agents in a hierarchical structure where manager agents coordinate and oversee worker agents, enabling efficient task distribution and quality control.

Key features:
- **Structured Hierarchy**: Clear organizational structure with managers and workers
- **Delegated Authority**: Manager agents distribute tasks to specialized workers
- **Quality Oversight**: Multi-level review and validation processes
- **Scalable Organization**: Efficient coordination of large agent teams

## Use Cases

- Complex projects requiring management oversight
- Large-scale content production workflows
- Multi-stage validation and review processes
- Enterprise-level task coordination

## API Usage


## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `role` | string | Agent role: "manager" or "worker" | "worker" |
| `agents` | Array<AgentSpec> | Mix of manager and worker agents | Required |
| `max_loops` | integer | Coordination rounds for managers | 1 |

## Best Practices

- Clearly define manager and worker roles using the `role` parameter
- Give managers higher `max_loops` for coordination activities
- Design worker agents with specialized, focused responsibilities
- Use for complex projects requiring oversight and coordination

## Related Swarm Types

- [SequentialWorkflow](sequential_workflow.md) - For linear task progression
- [MultiAgentRouter](multi_agent_router.md) - For intelligent task routing
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic hierarchy creation