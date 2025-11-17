# uvloop Examples

This directory contains examples demonstrating the use of uvloop for running multiple agents concurrently with improved performance.

## Files

- `utils.py`: Utility functions for creating example agents
- `same_task_example.py`: Example of running multiple agents with the same task
- `different_tasks_example.py`: Example of running agents with different tasks
- `performance_info.py`: Information about uvloop performance benefits
- `run_all_examples.py`: Script to run all examples

## Prerequisites

Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Individual Examples

Run a specific example:
```python
from same_task_example import run_same_task_example
results = run_same_task_example()
```

### Run All Examples

```python
from run_all_examples import run_all_uvloop_examples
all_results = run_all_uvloop_examples()
```

## Performance Benefits

uvloop provides:
- ~2-4x faster execution compared to standard asyncio
- Better performance for I/O-bound operations
- Lower latency and higher throughput
- Automatic fallback to asyncio if uvloop is unavailable

## Functions Used

- `run_agents_concurrently_uvloop`: For running multiple agents with the same task
- `run_agents_with_tasks_uvloop`: For running agents with different tasks
