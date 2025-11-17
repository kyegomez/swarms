# Simple BatchedGridWorkflow Example

This example demonstrates the basic usage of `BatchedGridWorkflow` with minimal configuration for easy understanding.

## Basic Example

```python
from swarms import Agent
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow

# Create two basic agents
agent1 = Agent(model="gpt-4")
agent2 = Agent(model="gpt-4")

# Create workflow with default settings
workflow = BatchedGridWorkflow(
    agents=[agent1, agent2]
)

# Define simple tasks
tasks = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms"
]

# Run the workflow
result = workflow.run(tasks)
```

## Named Workflow Example

```python
# Create agents
writer = Agent(model="gpt-4")
analyst = Agent(model="gpt-4")

# Create named workflow
workflow = BatchedGridWorkflow(
    name="Content Analysis Workflow",
    description="Analyze and write content in parallel",
    agents=[writer, analyst]
)

# Content tasks
tasks = [
    "Write a short paragraph about renewable energy",
    "Analyze the benefits of solar power"
]

# Execute workflow
result = workflow.run(tasks)
```

## Multi-Loop Example

```python
# Create agents
agent1 = Agent(model="gpt-4")
agent2 = Agent(model="gpt-4")

# Create workflow with multiple loops
workflow = BatchedGridWorkflow(
    agents=[agent1, agent2],
    max_loops=3
)

# Tasks for iterative processing
tasks = [
    "Generate ideas for a mobile app",
    "Evaluate the feasibility of each idea"
]

# Run with multiple loops
result = workflow.run(tasks)
```

## Three Agent Example

```python
# Create three agents
researcher = Agent(model="gpt-4")
writer = Agent(model="gpt-4")
editor = Agent(model="gpt-4")

# Create workflow
workflow = BatchedGridWorkflow(
    name="Research and Writing Pipeline",
    agents=[researcher, writer, editor]
)

# Three different tasks
tasks = [
    "Research the history of artificial intelligence",
    "Write a summary of the research findings",
    "Review and edit the summary for clarity"
]

# Execute workflow
result = workflow.run(tasks)
```

## Key Points

- **Simple Setup**: Minimal configuration required for basic usage
- **Parallel Execution**: Tasks run simultaneously across agents
- **Flexible Configuration**: Easy to customize names, descriptions, and loop counts
- **Error Handling**: Built-in error handling and logging
- **Scalable**: Works with any number of agents and tasks

## Use Cases

- **Content Creation**: Multiple writers working on different topics
- **Research Tasks**: Different researchers investigating various aspects
- **Analysis Work**: Multiple analysts processing different datasets
- **Educational Content**: Different instructors creating materials for various subjects
