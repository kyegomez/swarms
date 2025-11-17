# Consistency Agent Documentation

The `SelfConsistencyAgent` is a specialized agent designed for generating multiple independent responses to a given task and aggregating them into a single, consistent final answer. It leverages concurrent processing to enhance efficiency and employs a majority voting mechanism to ensure the reliability of the aggregated response.

## Purpose

The primary objective of the `SelfConsistencyAgent` is to provide a robust mechanism for decision-making and problem-solving by generating diverse responses and synthesizing them into a coherent final answer. This approach is particularly useful in scenarios where consistency and reliability are critical.

## Class: `SelfConsistencyAgent`

### Initialization

- **`__init__`**: Initializes the `SelfConsistencyAgent` with specified parameters.

#### Arguments

| Argument               | Type    | Default | Description                                                                 |
|------------------------|---------|---------|-----------------------------------------------------------------------------|
| `name`                 | `str`   | `"Self-Consistency-Agent"` | Name of the agent.                                                         |
| `description`          | `str`   | `"An agent that uses self consistency to generate a final answer."` | Description of the agent's purpose.                                        |
| `system_prompt`        | `str`   | `CONSISTENCY_SYSTEM_PROMPT` | System prompt for the reasoning agent.                                     |
| `model_name`           | `str`   | Required | The underlying language model to use.                                      |
| `num_samples`          | `int`   | `5`     | Number of independent responses to generate.                               |
| `max_loops`            | `int`   | `1`     | Maximum number of reasoning loops per sample.                              |
| `majority_voting_prompt` | `Optional[str]` | `majority_voting_prompt` | Custom prompt for majority voting aggregation.                             |
| `eval`                 | `bool`  | `False` | Enable evaluation mode for answer validation.                              |
| `output_type`          | `OutputType` | `"dict"` | Format of the output.                                                      |
| `random_models_on`     | `bool`  | `False` | Enable random model selection for diversity.                               |

### Methods

- **`run`**: Generates multiple responses for the given task and aggregates them.
  - **Arguments**:
    - `task` (`str`): The input prompt.
    - `img` (`Optional[str]`, optional): Image input for vision tasks.
    - `answer` (`Optional[str]`, optional): Expected answer for validation (if eval=True).
  - **Returns**: `Union[str, Dict[str, Any]]` - The aggregated final answer.

- **`aggregation_agent`**: Aggregates a list of responses into a single final answer using majority voting.
  - **Arguments**:
    - `responses` (`List[str]`): The list of responses.
    - `prompt` (`str`, optional): Custom prompt for the aggregation agent.
    - `model_name` (`str`, optional): Model to use for aggregation.
  - **Returns**: `str` - The aggregated answer.

- **`check_responses_for_answer`**: Checks if a specified answer is present in any of the provided responses.
  - **Arguments**:
    - `responses` (`List[str]`): A list of responses to check.
    - `answer` (`str`): The answer to look for in the responses.
  - **Returns**: `bool` - `True` if the answer is found, `False` otherwise.

- **`batched_run`**: Run the agent on multiple tasks in batch.
  - **Arguments**:
    - `tasks` (`List[str]`): List of tasks to be processed.
  - **Returns**: `List[Union[str, Dict[str, Any]]]` - List of results for each task.

### Examples

#### Example 1: Basic Usage

```python
from swarms.agents.consistency_agent import SelfConsistencyAgent

# Initialize the agent
agent = SelfConsistencyAgent(
    name="Math-Reasoning-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    num_samples=5
)

# Define a task
task = "What is the 40th prime number?"

# Run the agent
final_answer = agent.run(task)

# Print the final aggregated answer
print("Final aggregated answer:", final_answer)
```

#### Example 2: Using Custom Majority Voting Prompt

```python
from swarms.agents.consistency_agent import SelfConsistencyAgent

# Initialize the agent with a custom majority voting prompt
agent = SelfConsistencyAgent(
    name="Reasoning-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    num_samples=5,
    majority_voting_prompt="Please provide the most common response."
)

# Define a task
task = "Explain the theory of relativity in simple terms."

# Run the agent
final_answer = agent.run(task)

# Print the final aggregated answer
print("Final aggregated answer:", final_answer)
```

#### Example 3: Evaluation Mode

```python
from swarms.agents.consistency_agent import SelfConsistencyAgent

# Initialize the agent with evaluation mode
agent = SelfConsistencyAgent(
    name="Validation-Agent",
    model_name="gpt-4o-mini",
    num_samples=3,
    eval=True
)

# Run with expected answer for validation
result = agent.run("What is 2 + 2?", answer="4", eval=True)
if result is not None:
    print("Validation passed:", result)
else:
    print("Validation failed - expected answer not found")
```

#### Example 4: Random Models for Diversity

```python
from swarms.agents.consistency_agent import SelfConsistencyAgent

# Initialize the agent with random model selection
agent = SelfConsistencyAgent(
    name="Diverse-Reasoning-Agent",
    model_name="gpt-4o-mini",
    num_samples=5,
    random_models_on=True
)

# Run the agent
result = agent.run("What are the benefits of renewable energy?")
print("Diverse reasoning result:", result)
```

#### Example 5: Batch Processing

```python
from swarms.agents.consistency_agent import SelfConsistencyAgent

# Initialize the agent
agent = SelfConsistencyAgent(
    name="Batch-Processing-Agent",
    model_name="gpt-4o-mini",
    num_samples=3
)

# Define multiple tasks
tasks = [
    "What is the capital of France?",
    "What is 15 * 23?",
    "Explain photosynthesis in simple terms."
]

# Process all tasks
results = agent.batched_run(tasks)

# Print results
for i, result in enumerate(results):
    print(f"Task {i+1} result: {result}")
```

## Key Features

### Self-Consistency Technique
The agent implements the self-consistency approach based on the research paper "Self-Consistency Improves Chain of Thought Reasoning in Language Models" by Wang et al. (2022). This technique:

1. **Generates Multiple Independent Responses**: Creates several reasoning paths for the same problem
2. **Analyzes Consistency**: Examines agreement among different reasoning approaches
3. **Aggregates Results**: Uses majority voting or consensus building
4. **Produces Reliable Output**: Delivers a final answer reflecting the most reliable consensus

### Benefits
- **Mitigates Random Errors**: Multiple reasoning paths reduce individual path errors
- **Reduces Bias**: Diverse approaches minimize single-method biases
- **Improves Reliability**: Consensus-based results are more trustworthy
- **Handles Complexity**: Better performance on complex problem-solving tasks

### Use Cases
- **Mathematical Problem Solving**: Where accuracy is critical
- **Decision Making**: When reliability is paramount
- **Validation Tasks**: When answers need verification
- **Complex Reasoning**: Multi-step problem solving
- **Research Questions**: Where multiple perspectives are valuable

## Technical Details

### Concurrent Execution
The agent uses `ThreadPoolExecutor` to generate multiple responses concurrently, improving performance while maintaining independence between reasoning paths.

### Aggregation Process
The aggregation uses an AI-powered agent that:
- Identifies dominant responses
- Analyzes disparities and disagreements
- Evaluates consensus strength
- Synthesizes minority insights
- Provides comprehensive recommendations

### Output Formats
The agent supports various output types:
- `"dict"`: Dictionary format with conversation history
- `"str"`: Simple string output
- `"list"`: List format
- `"json"`: JSON formatted output

## Limitations

1. **Computational Cost**: Higher `num_samples` increases processing time and cost
2. **Model Dependencies**: Performance depends on the underlying model capabilities
3. **Consensus Challenges**: May struggle with tasks where multiple valid approaches exist
4. **Memory Usage**: Concurrent execution requires more memory resources

## Best Practices

1. **Sample Size**: Use 3-7 samples for most tasks; increase for critical decisions
2. **Model Selection**: Choose models with strong reasoning capabilities
3. **Evaluation Mode**: Enable for tasks with known correct answers
4. **Custom Prompts**: Tailor majority voting prompts for specific domains
5. **Batch Processing**: Use `batched_run` for multiple related tasks

---
