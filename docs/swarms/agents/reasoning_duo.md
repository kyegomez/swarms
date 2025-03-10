# ReasoningDuo

The ReasoningDuo class implements a dual-agent reasoning system that combines a reasoning agent and a main agent to provide well-thought-out responses to complex tasks. This architecture enables more robust and reliable outputs by separating the reasoning process from the final response generation.


## Class Overview

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_name | str | "reasoning-agent-01" | Name identifier for the reasoning agent |
| description | str | "A highly intelligent..." | Description of the reasoning agent's capabilities |
| model_names | list[str] | ["gpt-4o-mini", "gpt-4o"] | Model names for reasoning and main agents |
| system_prompt | str | "You are a helpful..." | System prompt for the main agent |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| run | task: str | str | Processes a single task through both agents |
| batched_run | tasks: List[str] | List[str] | Processes multiple tasks sequentially |



## Quick Start

```python
from swarms.agents.reasoning_duo import ReasoningDuo

# Initialize the ReasoningDuo
duo = ReasoningDuo(
    model_name="reasoning-agent-01",
    model_names=["gpt-4o-mini", "gpt-4o"]
)

# Run a single task
result = duo.run("Explain the concept of gravitational waves")

# Run multiple tasks
tasks = [
    "Calculate compound interest for $1000 over 5 years",
    "Explain quantum entanglement"
]
results = duo.batched_run(tasks)
```

## Examples

### 1. Mathematical Analysis

```python
duo = ReasoningDuo()

# Complex mathematical problem
math_task = """
Solve the following differential equation:
dy/dx + 2y = x^2, y(0) = 1
"""

solution = duo.run(math_task)
```

### 2. Physics Problem

```python
# Quantum mechanics problem
physics_task = """
Calculate the wavelength of an electron with kinetic energy of 50 eV 
using the de Broglie relationship.
"""

result = duo.run(physics_task)
```

### 3. Financial Analysis

```python
# Complex financial analysis
finance_task = """
Calculate the Net Present Value (NPV) of a project with:
- Initial investment: $100,000
- Annual cash flows: $25,000 for 5 years
- Discount rate: 8%
"""

analysis = duo.run(finance_task)
```

## Advanced Usage

### Customizing Agent Behavior

You can customize both agents by modifying their initialization parameters:

```python
duo = ReasoningDuo(
    model_name="custom-reasoning-agent",
    description="Specialized financial analysis agent",
    model_names=["gpt-4o-mini", "gpt-4o"],
    system_prompt="You are a financial expert AI assistant..."
)
```

### Batch Processing with Progress Tracking

```python
tasks = [
    "Analyze market trends for tech stocks",
    "Calculate risk metrics for a portfolio",
    "Forecast revenue growth"
]

# Process multiple tasks with logging
results = duo.batched_run(tasks)
```

## Implementation Details

The ReasoningDuo uses a two-stage process:

1. **Reasoning Stage**: The reasoning agent analyzes the task and develops a structured approach
2. **Execution Stage**: The main agent uses the reasoning output to generate the final response

### Internal Architecture

```
Task Input → Reasoning Agent → Structured Analysis → Main Agent → Final Output
```

## Best Practices

1. **Task Formulation**
   - Be specific and clear in task descriptions
   - Include relevant context and constraints
   - Break complex problems into smaller subtasks

2. **Performance Optimization**
   - Use batched_run for multiple related tasks
   - Monitor agent outputs for consistency
   - Adjust model parameters based on task complexity

## Error Handling

The ReasoningDuo includes built-in logging using the `loguru` library:

```python
from loguru import logger

# Logs are automatically generated for each task
logger.info("Task processing started")
```

## Limitations

- Processing time may vary based on task complexity
- Model response quality depends on input clarity
- Resource usage scales with batch size
