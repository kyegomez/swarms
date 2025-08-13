# AutoSwarmBuilder [ Needs an Fix ]

*Automatically configures optimal swarm architectures based on task requirements*

**Swarm Type**: `AutoSwarmBuilder`

## Overview

The AutoSwarmBuilder automatically configures optimal agent architectures based on task requirements and performance metrics, simplifying swarm creation. This intelligent system analyzes the given task and automatically generates the most suitable agent configuration, eliminating the need for manual swarm design.

Key features:
- **Intelligent Configuration**: Automatically designs optimal swarm structures
- **Task-Adaptive**: Adapts architecture based on specific task requirements
- **Performance Optimization**: Selects configurations for maximum efficiency
- **Simplified Setup**: Eliminates manual agent configuration complexity

## Use Cases

- Quick prototyping and experimentation
- Unknown or complex task requirements
- Automated swarm optimization
- Simplified swarm creation for non-experts

## API Usage


## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `task` | string | Task description for automatic optimization | Required |
| `rules` | string | Additional constraints and guidelines | None |
| `max_loops` | integer | Maximum execution rounds | 1 |

## Best Practices

- Provide detailed, specific task descriptions for better optimization
- Use `rules` parameter to guide the automatic configuration
- Ideal for rapid prototyping and experimentation
- Review generated architecture in response metadata

## Related Swarm Types

- [Auto](auto.md) - For automatic swarm type selection
- [MixtureOfAgents](mixture_of_agents.md) - Often selected by AutoSwarmBuilder
- [HierarchicalSwarm](hierarchical_swarm.md) - For complex structured tasks