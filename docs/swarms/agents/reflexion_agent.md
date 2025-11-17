# ReflexionAgent

The ReflexionAgent is an advanced AI agent that implements the Reflexion framework to improve through self-reflection. It follows a process of acting on tasks, evaluating its performance, generating self-reflections, and using these reflections to improve future responses.

## Overview

The ReflexionAgent consists of three specialized sub-agents:
- **Actor**: Generates initial responses to tasks
- **Evaluator**: Critically assesses responses against quality criteria
- **Reflector**: Generates self-reflections to improve future responses

## Initialization

```python
from swarms.agents import ReflexionAgent

agent = ReflexionAgent(
    agent_name="reflexion-agent",
    system_prompt="...",  # Optional custom system prompt
    model_name="openai/o1",
    max_loops=3,
    memory_capacity=100
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | `"reflexion-agent"` | Name of the agent |
| `system_prompt` | `str` | `REFLEXION_PROMPT` | System prompt for the agent |
| `model_name` | `str` | `"openai/o1"` | Model name for generating responses |
| `max_loops` | `int` | `3` | Maximum number of reflection iterations per task |
| `memory_capacity` | `int` | `100` | Maximum capacity of long-term memory |

## Methods

### act

Generates a response to the given task using the actor agent.

```python
response = agent.act(task: str, relevant_memories: List[Dict[str, Any]] = None) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The task to respond to |
| `relevant_memories` | `List[Dict[str, Any]]` | Optional relevant past memories to consider |

### evaluate

Evaluates the quality of a response to a task.

```python
evaluation, score = agent.evaluate(task: str, response: str) -> Tuple[str, float]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The original task |
| `response` | `str` | The response to evaluate |

Returns:
- `evaluation`: Detailed feedback on the response
- `score`: Numerical score between 0 and 1

### reflect

Generates a self-reflection based on the task, response, and evaluation.

```python
reflection = agent.reflect(task: str, response: str, evaluation: str) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The original task |
| `response` | `str` | The generated response |
| `evaluation` | `str` | The evaluation feedback |

### refine

Refines the original response based on evaluation and reflection.

```python
refined_response = agent.refine(
    task: str,
    original_response: str,
    evaluation: str,
    reflection: str
) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The original task |
| `original_response` | `str` | The original response |
| `evaluation` | `str` | The evaluation feedback |
| `reflection` | `str` | The self-reflection |

### step

Processes a single task through one iteration of the Reflexion process.

```python
result = agent.step(
    task: str,
    iteration: int = 0,
    previous_response: str = None
) -> Dict[str, Any]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | The task to process |
| `iteration` | `int` | Current iteration number |
| `previous_response` | `str` | Response from previous iteration |

Returns a dictionary containing:
- `task`: The original task
- `response`: The generated response
- `evaluation`: The evaluation feedback
- `reflection`: The self-reflection
- `score`: Numerical score
- `iteration`: Current iteration number

### run

Executes the Reflexion process for a list of tasks.

```python
results = agent.run(
    tasks: List[str],
    include_intermediates: bool = False
) -> List[Any]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tasks` | `List[str]` | List of tasks to process |
| `include_intermediates` | `bool` | Whether to include intermediate iterations in results |

Returns:
- If `include_intermediates=False`: List of final responses
- If `include_intermediates=True`: List of complete iteration histories

## Example Usage

```python
from swarms.agents import ReflexionAgent

# Initialize the Reflexion Agent
agent = ReflexionAgent(
    agent_name="reflexion-agent",
    model_name="openai/o1",
    max_loops=3
)

# Example tasks
tasks = [
    "Explain quantum computing to a beginner.",
    "Write a Python function to sort a list of dictionaries by a specific key."
]

# Run the agent
results = agent.run(tasks)

# Print results
for i, result in enumerate(results):
    print(f"\nTask {i+1}: {tasks[i]}")
    print(f"Response: {result}")
```

## Memory System

The ReflexionAgent includes a sophisticated memory system (`ReflexionMemory`) that maintains both short-term and long-term memories of past experiences, reflections, and feedback. This system helps the agent learn from past interactions and improve its responses over time.

### Memory Features
- Short-term memory for recent interactions
- Long-term memory for important reflections and patterns
- Automatic memory management with capacity limits
- Relevance-based memory retrieval
- Similarity-based deduplication

## Best Practices

1. **Task Clarity**: Provide clear, specific tasks to get the best results
2. **Iteration Count**: Adjust `max_loops` based on task complexity (more complex tasks may benefit from more iterations)
3. **Memory Management**: Monitor memory usage and adjust `memory_capacity` as needed
4. **Model Selection**: Choose an appropriate model based on your specific use case and requirements
5. **Error Handling**: Implement proper error handling when using the agent in production
