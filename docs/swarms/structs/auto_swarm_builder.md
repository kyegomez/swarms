# AutoSwarmBuilder Documentation

The `AutoSwarmBuilder` is a powerful class that automatically builds and manages swarms of AI agents to accomplish complex tasks. It uses a boss agent to delegate work and create specialized agents as needed.

## Overview

The AutoSwarmBuilder is designed to:

- Automatically create and coordinate multiple AI agents

- Delegate tasks to specialized agents

- Manage communication between agents

- Handle complex workflows through a swarm router


## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | None | The name of the swarm |
| description | str | None | A description of the swarm's purpose |
| verbose | bool | True | Whether to output detailed logs |
| max_loops | int | 1 | Maximum number of execution loops |
| random_models | bool | True | Whether to use random models for agents |

## Core Methods

### run(task: str, *args, **kwargs)

Executes the swarm on a given task.

**Parameters:**

- `task` (str): The task to execute

- `*args`: Additional positional arguments

- `**kwargs`: Additional keyword arguments

**Returns:**

- The result of the swarm execution

### create_agents(task: str)

Creates specialized agents for a given task.

**Parameters:**

- `task` (str): The task to create agents for

**Returns:**

- List[Agent]: List of created agents

### build_agent(agent_name: str, agent_description: str, agent_system_prompt: str)
Builds a single agent with specified parameters.

**Parameters:**
- `agent_name` (str): Name of the agent

- `agent_description` (str): Description of the agent

- `agent_system_prompt` (str): System prompt for the agent


**Returns:**

- Agent: The constructed agent

### batch_run(tasks: List[str])

Executes the swarm on multiple tasks.

**Parameters:**

- `tasks` (List[str]): List of tasks to execute

**Returns:**

- List[Any]: Results from each task execution

## Examples

### Example 1: Content Creation Swarm

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize the swarm builder
swarm = AutoSwarmBuilder(
    name="Content Creation Swarm",
    description="A swarm specialized in creating high-quality content"
)

# Run the swarm on a content creation task
result = swarm.run(
    "Create a comprehensive blog post about artificial intelligence in healthcare, "
    "including current applications, future trends, and ethical considerations."
)
```

### Example 2: Data Analysis Swarm

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize the swarm builder
swarm = AutoSwarmBuilder(
    name="Data Analysis Swarm",
    description="A swarm specialized in data analysis and visualization"
)

# Run the swarm on a data analysis task
result = swarm.run(
    "Analyze the provided sales data and create a detailed report with visualizations "
    "showing trends, patterns, and recommendations for improvement."
)
```

### Example 3: Batch Processing Multiple Tasks

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize the swarm builder
swarm = AutoSwarmBuilder(
    name="Multi-Task Swarm",
    description="A swarm capable of handling multiple diverse tasks"
)

# Define multiple tasks
tasks = [
    "Create a marketing strategy for a new product launch",
    "Analyze customer feedback and generate improvement suggestions",
    "Develop a project timeline for the next quarter"
]

# Run the swarm on all tasks
results = swarm.batch_run(tasks)
```

## Best Practices

!!! tip "Task Definition"
    - Provide clear, specific task descriptions
    
    - Include any relevant context or constraints
    
    - Specify expected output format if needed

!!! note "Configuration"
    
    - Set appropriate `max_loops` based on task complexity
    
    - Use `verbose=True` during development for debugging
    
    - Consider using `random_models=True` for diverse agent capabilities

!!! warning "Error Handling"
    - The class includes comprehensive error handling
    
    - All methods include try-catch blocks with detailed logging
    
    - Errors are propagated with full stack traces for debugging

## Notes

!!! info "Architecture"
    
    - The AutoSwarmBuilder uses a sophisticated boss agent system to coordinate tasks
    
    - Agents are created dynamically based on task requirements
    
    - The system includes built-in logging and error handling
    
    - Results are returned in a structured format for easy processing
