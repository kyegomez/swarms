# Auto

*Intelligently selects the most effective swarm architecture for a given task*

**Swarm Type**: `auto` (or `Auto`)

## Overview

The Auto swarm type intelligently selects the most effective swarm architecture for a given task based on context analysis and task requirements. This intelligent system evaluates the task description and automatically chooses the optimal swarm type from all available architectures, ensuring maximum efficiency and effectiveness.

Key features:
- **Intelligent Selection**: Automatically chooses the best swarm type for each task
- **Context Analysis**: Analyzes task requirements to make optimal decisions
- **Adaptive Architecture**: Adapts to different types of problems automatically
- **Zero Configuration**: No manual architecture selection required

## Use Cases

- When unsure about which swarm type to use
- General-purpose task automation
- Rapid prototyping and experimentation
- Simplified API usage for non-experts

## API Usage



## Selection Logic

The Auto swarm type analyzes various factors to make its selection:

| Factor | Consideration |
|--------|---------------|
| **Task Complexity** | Simple → Single agent, Complex → Multi-agent |
| **Sequential Dependencies** | Dependencies → SequentialWorkflow |
| **Parallel Opportunities** | Independent subtasks → ConcurrentWorkflow |
| **Collaboration Needs** | Discussion required → GroupChat |
| **Expertise Diversity** | Multiple domains → MixtureOfAgents |
| **Management Needs** | Oversight required → HierarchicalSwarm |
| **Routing Requirements** | Task distribution → MultiAgentRouter |

## Best Practices

- Provide detailed task descriptions for better selection
- Use `rules` parameter to guide selection criteria
- Review the selected architecture in response metadata
- Ideal for users new to swarm architectures

## Related Swarm Types

Since Auto can select any swarm type, it's related to all architectures:
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic agent generation
- [SequentialWorkflow](sequential_workflow.md) - Often selected for linear tasks
- [ConcurrentWorkflow](concurrent_workflow.md) - For parallel processing needs
- [MixtureOfAgents](mixture_of_agents.md) - For diverse expertise requirements