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
| `num_samples`          | `int`   | `5`     | Number of independent responses to sample.                                  |
| `return_list`          | `bool`  | `False` | Whether to return the conversation as a list.                               |
| `max_loops`            | `int`   | `1`     | Maximum number of loops for the agent to run.                               |
| `return_dict`          | `bool`  | `False` | Whether to return the conversation as a dictionary.                         |
| `return_json`          | `bool`  | `False` | Whether to return the conversation as JSON.                                 |
| `majority_voting_prompt` | `str` | `None`  | Custom prompt for majority voting.                                          |

### Methods

- **`run`**: Generates multiple responses for the given task and aggregates them.
  - **Arguments**:
    - `task` (`str`): The input prompt.
    - `answer` (`str`, optional): The expected answer to validate responses against.
  - **Returns**: `str` - The aggregated final answer.

- **`aggregate`**: Aggregates a list of responses into a single final answer using majority voting.
  - **Arguments**:
    - `responses` (`List[str]`): The list of responses.
  - **Returns**: `str` - The aggregated answer.

- **`check_responses_for_answer`**: Checks if a specified answer is present in any of the provided responses.
  - **Arguments**:
    - `responses` (`List[str]`): A list of responses to check.
    - `answer` (`str`): The answer to look for in the responses.
  - **Returns**: `bool` - `True` if the answer is found, `False` otherwise.

### Examples

#### Example 1: Basic Usage

```python
from swarms.agents.consistency_agent import SelfConsistencyAgent

# Initialize the agent
agent = SelfConsistencyAgent(
    agent_name="Reasoning-Agent",
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
    agent_name="Reasoning-Agent",
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

---
