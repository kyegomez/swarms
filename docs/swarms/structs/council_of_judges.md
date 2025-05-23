# CouncilAsAJudge

The `CouncilAsAJudge` is a sophisticated evaluation system that employs multiple AI agents to assess model responses across various dimensions. It provides comprehensive, multi-dimensional analysis of AI model outputs through parallel evaluation and aggregation.

## Overview

The `CouncilAsAJudge` implements a council of specialized AI agents that evaluate different aspects of a model's response. Each agent focuses on a specific dimension of evaluation, and their findings are aggregated into a comprehensive report.

## Key Features

- Parallel evaluation across multiple dimensions
- Caching system for improved performance
- Dynamic model selection
- Comprehensive evaluation metrics
- Thread-safe execution
- Detailed technical analysis

## Installation

```bash
pip install swarms
```

## Basic Usage

```python
from swarms import Agent, CouncilAsAJudge

# Create a base agent
base_agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt="You are a financial expert helping users understand and establish ROTH IRAs.",
    model_name="claude-opus-4-20250514",
    max_loops=1,
)

# Run the base agent
user_query = "How can I establish a ROTH IRA to buy stocks and get a tax break?"
model_output = base_agent.run(user_query)

# Create and run the council
panel = CouncilAsAJudge()
results = panel.run(user_query, model_output)
print(results)
```

## Advanced Usage

### Custom Model Configuration

```python
from swarms import CouncilAsAJudge

# Initialize with custom model
council = CouncilAsAJudge(
    model_name="anthropic/claude-3-sonnet-20240229",
    output_type="all",
    cache_size=256,
    max_workers=4,
    random_model_name=False
)
```

### Parallel Processing Configuration

```python
from swarms import CouncilAsAJudge

# Configure parallel processing
council = CouncilAsAJudge(
    max_workers=8,  # Custom number of worker threads
    random_model_name=True  # Enable dynamic model selection
)
```

## Evaluation Dimensions

The council evaluates responses across six key dimensions:

| Dimension | Evaluation Criteria |
|-----------|-------------------|
| **Accuracy** | • Factual correctness<br>• Source credibility<br>• Temporal consistency<br>• Technical accuracy |
| **Helpfulness** | • Problem-solving efficacy<br>• Solution feasibility<br>• Context inclusion<br>• Proactive addressing of follow-ups |
| **Harmlessness** | • Safety assessment<br>• Ethical considerations<br>• Age-appropriateness<br>• Content sensitivity |
| **Coherence** | • Structural integrity<br>• Logical flow<br>• Information hierarchy<br>• Transition effectiveness |
| **Conciseness** | • Communication efficiency<br>• Information density<br>• Redundancy elimination<br>• Focus maintenance |
| **Instruction Adherence** | • Requirement coverage<br>• Constraint compliance<br>• Format matching<br>• Scope appropriateness |

## API Reference

### CouncilAsAJudge

```python
class CouncilAsAJudge:
    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "CouncilAsAJudge",
        description: str = "Evaluates the model's response across multiple dimensions",
        model_name: str = "gpt-4o-mini",
        output_type: str = "all",
        cache_size: int = 128,
        max_workers: int = None,
        random_model_name: bool = True,
    )
```

#### Parameters

- `id` (str): Unique identifier for the council
- `name` (str): Display name of the council
- `description` (str): Description of the council's purpose
- `model_name` (str): Name of the model to use for evaluations
- `output_type` (str): Type of output to return
- `cache_size` (int): Size of the LRU cache for prompts
- `max_workers` (int): Maximum number of worker threads
- `random_model_name` (bool): Whether to use random model selection

### Methods

#### run

```python
def run(self, task: str, model_response: str) -> None
```

Evaluates a model response across all dimensions.

##### Parameters

- `task` (str): Original user prompt
- `model_response` (str): Model's response to evaluate

##### Returns

- Comprehensive evaluation report

## Examples

### Financial Analysis Example

```python
from swarms import Agent, CouncilAsAJudge

# Create financial analysis agent
financial_agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt="You are a financial expert helping users understand and establish ROTH IRAs.",
    model_name="claude-opus-4-20250514",
    max_loops=1,
)

# Run analysis
query = "How can I establish a ROTH IRA to buy stocks and get a tax break?"
response = financial_agent.run(query)

# Evaluate response
council = CouncilAsAJudge()
evaluation = council.run(query, response)
print(evaluation)
```

### Technical Documentation Example

```python
from swarms import Agent, CouncilAsAJudge

# Create documentation agent
doc_agent = Agent(
    agent_name="Documentation-Agent",
    system_prompt="You are a technical documentation expert.",
    model_name="gpt-4",
    max_loops=1,
)

# Generate documentation
query = "Explain how to implement a REST API using FastAPI"
response = doc_agent.run(query)

# Evaluate documentation quality
council = CouncilAsAJudge(
    model_name="anthropic/claude-3-sonnet-20240229",
    output_type="all"
)
evaluation = council.run(query, response)
print(evaluation)
```

## Best Practices

### Model Selection

!!! tip "Model Selection Best Practices"
    - Choose appropriate models for your use case
    - Consider using random model selection for diverse evaluations
    - Match model capabilities to evaluation requirements

### Performance Optimization

!!! note "Performance Tips"
    - Adjust cache size based on memory constraints
    - Configure worker threads based on CPU cores
    - Monitor memory usage with large responses

### Error Handling

!!! warning "Error Handling Guidelines"
    - Implement proper exception handling
    - Monitor evaluation failures
    - Log evaluation results for analysis

### Resource Management

!!! info "Resource Management"
    - Clean up resources after evaluation
    - Monitor thread pool usage
    - Implement proper shutdown procedures

## Troubleshooting

### Memory Issues

!!! danger "Memory Problems"
    If you encounter memory-related problems:

    - Reduce cache size
    - Decrease number of worker threads
    - Process smaller chunks of text

### Performance Problems

!!! warning "Performance Issues"
    To improve performance:

    - Increase cache size
    - Adjust worker thread count
    - Use more efficient models

### Evaluation Failures

!!! danger "Evaluation Issues"
    When evaluations fail:

    - Check model availability
    - Verify input format
    - Monitor error logs

## Contributing

!!! success "Contributing"
    Contributions are welcome! Please feel free to submit a Pull Request.

## License

!!! info "License"
    This project is licensed under the MIT License - see the LICENSE file for details.