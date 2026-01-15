# Multi-Agent Execution API Reference

This comprehensive documentation covers all functions in the `multi_agent_exec.py` module for running multiple agents using various execution strategies. The module provides synchronous and asynchronous execution methods, and uses stdlib `asyncio` event-loop policies by default (optional `uvloop`/`winloop` can be installed separately), plus utility functions for information retrieval.

## Function Overview

| Function                               | Signature                                                                                                                               | Category             | Description                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `run_single_agent`                     | `run_single_agent(agent, task, *args, **kwargs) -> Any`                                                                                 | Single Agent         | Runs a single agent synchronously                                                                                      |
| `run_agent_async`                      | `run_agent_async(agent, task) -> Any`                                                                                                   | Single Agent         | Runs a single agent asynchronously using asyncio                                                                       |
| `run_agents_concurrently_async`        | `run_agents_concurrently_async(agents, task) -> List[Any]`                                                                              | Concurrent Execution | Runs multiple agents concurrently using asyncio                                                                        |
| `run_agents_concurrently`              | `run_agents_concurrently(agents, task, img=None, max_workers=None, return_agent_output_dict=False) -> Union[List[Any], Dict[str, Any]]` | Concurrent Execution | Optimized concurrent agent runner with image support and flexible output formats                                       |
| `run_agents_concurrently_multiprocess` | `run_agents_concurrently_multiprocess(agents, task, batch_size=None) -> List[Any]`                                                      | Concurrent Execution | Manages agents concurrently in batches with optimized performance                                                      |
| `batched_grid_agent_execution`         | `batched_grid_agent_execution(agents, tasks, max_workers=None) -> List[Any]`                                                            | Batched & Grid       | Runs multiple agents with different tasks concurrently                                                                 |
| `run_agents_with_different_tasks`      | `run_agents_with_different_tasks(agent_task_pairs, batch_size=10, max_workers=None) -> List[Any]`                                       | Batched & Grid       | Runs agents with different tasks concurrently in batches                                                               |
| `run_agents_concurrently_uvloop`       | `run_agents_concurrently_uvloop(agents, task, max_workers=None) -> List[Any]`                                                           | Platform Optimized   | Runs agents concurrently using stdlib `asyncio` policies by default (optional `uvloop`/`winloop` for users who opt-in) |
| `run_agents_with_tasks_uvloop`         | `run_agents_with_tasks_uvloop(agents, tasks, max_workers=None) -> List[Any]`                                                            | Platform Optimized   | Runs agents with different tasks using platform-specific optimizations                                                 |
| `get_swarms_info`                      | `get_swarms_info(swarms) -> str`                                                                                                        | Utility              | Fetches and formats information about available swarms                                                                 |
| `get_agents_info`                      | `get_agents_info(agents, team_name=None) -> str`                                                                                        | Utility              | Fetches and formats information about available agents                                                                 |

## Single Agent Functions

### `run_single_agent(agent, task, *args, **kwargs)`

Runs a single agent synchronously.

#### Signature

```python
def run_single_agent(
    agent: AgentType,
    task: str,
    *args,
    **kwargs
) -> Any
```

#### Parameters

| Parameter  | Type        | Required | Description                     |
| ---------- | ----------- | -------- | ------------------------------- |
| `agent`    | `AgentType` | Yes      | Agent instance to run           |
| `task`     | `str`       | Yes      | Task string to execute          |
| `*args`    | `Any`       | No       | Additional positional arguments |
| `**kwargs` | `Any`       | No       | Additional keyword arguments    |

#### Returns

- `Any`: Agent execution result

#### Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_single_agent

agent = Agent(
    agent_name="Financial-Analyst",
    system_prompt="You are a financial analysis expert",
    model_name="gpt-4o-mini",
    max_loops=1
)

result = run_single_agent(agent, "Analyze the current stock market trends")
print(result)
```

### `run_agent_async(agent, task)`

Runs a single agent asynchronously using asyncio.

#### Signature

```python
async def run_agent_async(agent: AgentType, task: str) -> Any
```

#### Parameters

| Parameter | Type        | Required | Description            |
| --------- | ----------- | -------- | ---------------------- |
| `agent`   | `AgentType` | Yes      | Agent instance to run  |
| `task`    | `str`       | Yes      | Task string to execute |

#### Returns

- `Any`: Agent execution result

#### Example

```python
import asyncio
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agent_async

async def main():
    agent = Agent(
        agent_name="Researcher",
        system_prompt="You are a research assistant",
        model_name="gpt-4o-mini",
        max_loops=1
    )

    result = await run_agent_async(agent, "Research AI advancements in 2024")
    print(result)

asyncio.run(main())
```

## Concurrent Execution Functions

### `run_agents_concurrently_async(agents, task)`

Runs multiple agents concurrently using asyncio.

#### Signature

```python
async def run_agents_concurrently_async(
    agents: List[AgentType],
    task: str
) -> List[Any]
```

#### Parameters

| Parameter | Type              | Required | Description                                 |
| --------- | ----------------- | -------- | ------------------------------------------- |
| `agents`  | `List[AgentType]` | Yes      | List of Agent instances to run concurrently |
| `task`    | `str`             | Yes      | Task string to execute by all agents        |

#### Returns

- `List[Any]`: List of outputs from each agent

#### Example

```python
import asyncio
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently_async

async def main():
    agents = [
        Agent(
            agent_name=f"Analyst-{i}",
            system_prompt="You are a market analyst",
            model_name="gpt-4o-mini",
            max_loops=1
        )
        for i in range(3)
    ]

    task = "Analyze the impact of AI on job markets"
    results = await run_agents_concurrently_async(agents, task)

    for i, result in enumerate(results):
        print(f"Agent {i+1} result: {result}")

asyncio.run(main())
```

### `run_agents_concurrently(agents, task, img=None, max_workers=None, return_agent_output_dict=False)`

Optimized concurrent agent runner using ThreadPoolExecutor with image support and flexible output formats.

#### Signature

```python
def run_agents_concurrently(
    agents: List[AgentType],
    task: str,
    img: Optional[str] = None,
    max_workers: Optional[int] = None,
    return_agent_output_dict: bool = False,
) -> Union[List[Any], Dict[str, Any]]
```

#### Parameters

| Parameter                  | Type              | Required | Default          | Description                                             |
| -------------------------- | ----------------- | -------- | ---------------- | ------------------------------------------------------- |
| `agents`                   | `List[AgentType]` | Yes      | -                | List of Agent instances to run concurrently             |
| `task`                     | `str`             | Yes      | -                | Task string to execute                                  |
| `img`                      | `Optional[str]`   | No       | None             | Optional image data to pass to agent run() if supported |
| `max_workers`              | `Optional[int]`   | No       | 95% of CPU cores | Maximum number of threads in the executor               |
| `return_agent_output_dict` | `bool`            | No       | False            | If True, returns a dict mapping agent names to outputs  |

#### Returns

- `Union[List[Any], Dict[str, Any]]`:
  - If `return_agent_output_dict=False`: List of outputs from each agent in completion order (exceptions included if agents fail)
  - If `return_agent_output_dict=True`: Dictionary mapping agent names to outputs, preserving agent input order

#### Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently

# Create multiple agents
agents = [
    Agent(
        agent_name="Tech-Analyst",
        system_prompt="You are a technology analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="Finance-Analyst",
        system_prompt="You are a financial analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="Market-Strategist",
        system_prompt="You are a market strategist",
        model_name="gpt-4o-mini",
        max_loops=1
    )
]

task = "Analyze the future of electric vehicles in 2025"

# Example 1: Basic concurrent execution
results = run_agents_concurrently(agents, task, max_workers=4)
for i, result in enumerate(results):
    print(f"Agent {i+1} ({agents[i].agent_name}): {result}")

# Example 2: With image support (if agents support it)
# image_data = "base64_encoded_image_string"
# results_with_img = run_agents_concurrently(agents, task, img=image_data)

# Example 3: Return results as dictionary with agent names as keys
results_dict = run_agents_concurrently(
    agents, task, return_agent_output_dict=True
)
for agent_name, result in results_dict.items():
    print(f"{agent_name}: {result}")
```

### `run_agents_concurrently_multiprocess(agents, task, batch_size=None)`

Manages and runs multiple agents concurrently in batches with optimized performance.

#### Signature

```python
def run_agents_concurrently_multiprocess(
    agents: List[Agent],
    task: str,
    batch_size: int = os.cpu_count()
) -> List[Any]
```

#### Parameters

| Parameter    | Type          | Required | Default   | Description                                       |
| ------------ | ------------- | -------- | --------- | ------------------------------------------------- |
| `agents`     | `List[Agent]` | Yes      | -         | List of Agent instances to run concurrently       |
| `task`       | `str`         | Yes      | -         | Task string to execute by all agents              |
| `batch_size` | `int`         | No       | CPU count | Number of agents to run in parallel in each batch |

#### Returns

- `List[Any]`: List of outputs from each agent

#### Example

```python
import os
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently_multiprocess

agents = [
    Agent(
        agent_name=f"Research-Agent-{i}",
        system_prompt="You are a research specialist",
        model_name="gpt-4o-mini",
        max_loops=1
    )
    for i in range(5)
]

task = "Research the benefits of renewable energy"
batch_size = os.cpu_count()  # Use all CPU cores
results = run_agents_concurrently_multiprocess(agents, task, batch_size)

print(f"Completed {len(results)} agent executions")
```

## Batched and Grid Execution

### `batched_grid_agent_execution(agents, tasks, max_workers=None)`

Runs multiple agents with different tasks concurrently using batched grid execution.

#### Signature

```python
def batched_grid_agent_execution(
    agents: List["AgentType"],
    tasks: List[str],
    max_workers: int = None,
) -> List[Any]
```

#### Parameters

| Parameter     | Type              | Required | Default          | Description                       |
| ------------- | ----------------- | -------- | ---------------- | --------------------------------- |
| `agents`      | `List[AgentType]` | Yes      | -                | List of agent instances           |
| `tasks`       | `List[str]`       | Yes      | -                | List of tasks, one for each agent |
| `max_workers` | `int`             | No       | 90% of CPU cores | Maximum number of threads to use  |

#### Returns

- `List[Any]`: List of results from each agent

#### Raises

- `ValueError`: If number of agents doesn't match number of tasks

#### Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import batched_grid_agent_execution

agents = [
    Agent(
        agent_name="Data-Scientist",
        system_prompt="You are a data science expert",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="ML-Engineer",
        system_prompt="You are a machine learning engineer",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="AI-Researcher",
        system_prompt="You are an AI researcher",
        model_name="gpt-4o-mini",
        max_loops=1
    )
]

tasks = [
    "Analyze machine learning algorithms performance",
    "Design a neural network architecture",
    "Research latest AI breakthroughs"
]

results = batched_grid_agent_execution(agents, tasks, max_workers=3)

for i, result in enumerate(results):
    print(f"Task {i+1}: {tasks[i]}")
    print(f"Result: {result}\n")
```

### `run_agents_with_different_tasks(agent_task_pairs, batch_size=10, max_workers=None)`

Runs multiple agents with different tasks concurrently, processing them in batches.

#### Signature

```python
def run_agents_with_different_tasks(
    agent_task_pairs: List[tuple["AgentType", str]],
    batch_size: int = 10,
    max_workers: int = None,
) -> List[Any]
```

#### Parameters

| Parameter          | Type                          | Required | Default | Description                                       |
| ------------------ | ----------------------------- | -------- | ------- | ------------------------------------------------- |
| `agent_task_pairs` | `List[tuple[AgentType, str]]` | Yes      | -       | List of (agent, task) tuples                      |
| `batch_size`       | `int`                         | No       | 10      | Number of agents to run in parallel in each batch |
| `max_workers`      | `int`                         | No       | None    | Maximum number of threads                         |

#### Returns

- `List[Any]`: List of outputs from each agent, in the same order as input pairs

#### Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_with_different_tasks

# Create agents
agents = [
    Agent(
        agent_name="Content-Writer",
        system_prompt="You are a content writer",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="Editor",
        system_prompt="You are an editor",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="SEO-Specialist",
        system_prompt="You are an SEO specialist",
        model_name="gpt-4o-mini",
        max_loops=1
    )
]

# Create agent-task pairs
agent_task_pairs = [
    (agents[0], "Write a blog post about sustainable living"),
    (agents[1], "Edit and improve this article draft"),
    (agents[2], "Optimize this content for SEO")
]

results = run_agents_with_different_tasks(agent_task_pairs, batch_size=2)

for i, result in enumerate(results):
    agent, task = agent_task_pairs[i]
    print(f"{agent.agent_name} - {task}: {result}")
```

## Platform Optimized Functions

### `run_agents_concurrently_uvloop(agents, task, max_workers=None)`

Runs multiple agents concurrently using platform-specific optimized event loops for enhanced performance.

This function uses stdlib `asyncio` event-loop policies by default and will continue to run without optional third-party loops:

- **Unix/Linux/macOS**: Uses stdlib `asyncio` policies by default (you may opt-in to `uvloop` for potential gains)
- **Windows**: Uses stdlib `asyncio` policies by default (you may opt-in to `winloop` on supported systems)
- **Fallback**: Continues with the existing stdlib `asyncio` policy if optional loops are not installed

#### Signature

```python
def run_agents_concurrently_uvloop(
    agents: List[AgentType],
    task: str,
    max_workers: Optional[int] = None,
) -> List[Any]
```

#### Parameters

| Parameter     | Type              | Required | Default          | Description                                 |
| ------------- | ----------------- | -------- | ---------------- | ------------------------------------------- |
| `agents`      | `List[AgentType]` | Yes      | -                | List of Agent instances to run concurrently |
| `task`        | `str`             | Yes      | -                | Task string to execute by all agents        |
| `max_workers` | `Optional[int]`   | No       | 95% of CPU cores | Maximum number of threads in the executor   |

#### Returns

- `List[Any]`: List of outputs from each agent. If an agent fails, the exception is included in the results.

#### Raises

- `ImportError`: Optional third-party event loops are not required — the function will continue with stdlib `asyncio` if they are not installed
- `RuntimeError`: If event loop policy cannot be set (falls back to standard asyncio)

#### Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently_uvloop

# Note: Platform-specific optimizations are available as an opt-in.
# - Optional: Unix/Linux/macOS: `pip install uvloop` to opt-in to uvloop
# - Optional: Windows: `pip install winloop` to opt-in to winloop
# - The function will use stdlib `asyncio` policies by default

agents = [
    Agent(
        agent_name="Performance-Analyst",
        system_prompt="You are a performance analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    )
    for _ in range(5)
]

task = "Analyze system performance metrics"
results = run_agents_concurrently_uvloop(agents, task)

print(f"Processed {len(results)} agents with platform-optimized event loop")
```

### `run_agents_with_tasks_uvloop(agents, tasks, max_workers=None)`

Runs multiple agents with different tasks concurrently using platform-specific optimized event loops.

This function uses stdlib `asyncio` event-loop policies by default and will continue to run without optional third-party loops:

- **Unix/Linux/macOS**: Uses stdlib `asyncio` policies by default (you may opt-in to `uvloop` for potential gains)
- **Windows**: Uses stdlib `asyncio` policies by default (you may opt-in to `winloop` on supported systems)
- **Fallback**: Continues with the existing stdlib `asyncio` policy if optional loops are not installed

#### Signature

```python
def run_agents_with_tasks_uvloop(
    agents: List[AgentType],
    tasks: List[str],
    max_workers: Optional[int] = None,
) -> List[Any]
```

#### Parameters

| Parameter     | Type              | Required | Default          | Description                                        |
| ------------- | ----------------- | -------- | ---------------- | -------------------------------------------------- |
| `agents`      | `List[AgentType]` | Yes      | -                | List of Agent instances to run                     |
| `tasks`       | `List[str]`       | Yes      | -                | List of task strings (must match number of agents) |
| `max_workers` | `Optional[int]`   | No       | 95% of CPU cores | Maximum number of threads                          |

#### Returns

- `List[Any]`: List of outputs from each agent in the same order as input agents. If an agent fails, the exception is included in the results.

#### Raises

- `ValueError`: If number of agents doesn't match number of tasks
- `ImportError`: If neither uvloop nor winloop is available (falls back to standard asyncio)
- `RuntimeError`: If event loop policy cannot be set (falls back to standard asyncio)

#### Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_with_tasks_uvloop

# Note: Platform-specific optimizations are automatically selected
# - Unix/Linux/macOS: Install uvloop with 'pip install uvloop'
# - Windows: Install winloop with 'pip install winloop'
# - Falls back to standard asyncio if neither is available

agents = [
    Agent(
        agent_name="Data-Analyst-1",
        system_prompt="You are a data analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="Data-Analyst-2",
        system_prompt="You are a data analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    )
]

tasks = [
    "Analyze sales data from Q1 2024",
    "Analyze customer satisfaction metrics"
]

results = run_agents_with_tasks_uvloop(agents, tasks)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Agent {i+1} with {tasks[i]} failed: {result}")
    else:
        print(f"Task: {tasks[i]}")
        print(f"Result: {result}\n")
```

## Utility Functions

### `get_swarms_info(swarms)`

Fetches and formats information about all available swarms in the system.

#### Signature

```python
def get_swarms_info(swarms: List[Callable]) -> str
```

#### Parameters

| Parameter | Type             | Required | Description                                    |
| --------- | ---------------- | -------- | ---------------------------------------------- |
| `swarms`  | `List[Callable]` | Yes      | List of swarm objects to get information about |

#### Returns

- `str`: Formatted string containing names and descriptions of all swarms

#### Example

```python
from swarms.structs.multi_agent_exec import get_swarms_info

# Assuming you have swarm objects
swarms = [
    # Your swarm objects here
]

info = get_swarms_info(swarms)
print(info)
# Output:
# Available Swarms:
#
# [Swarm 1]
# Name: ResearchSwarm
# Description: A swarm for research tasks
# Length of Agents: 3
# Swarm Type: hierarchical
```

### `get_agents_info(agents, team_name=None)`

Fetches and formats information about all available agents in the system.

#### Signature

```python
def get_agents_info(
    agents: List[Union[Agent, Callable]],
    team_name: str = None
) -> str
```

#### Parameters

| Parameter   | Type                           | Required | Default | Description                                    |
| ----------- | ------------------------------ | -------- | ------- | ---------------------------------------------- |
| `agents`    | `List[Union[Agent, Callable]]` | Yes      | -       | List of agent objects to get information about |
| `team_name` | `str`                          | No       | None    | Optional team name to display                  |

#### Returns

- `str`: Formatted string containing names and descriptions of all agents

#### Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import get_agents_info

agents = [
    Agent(
        agent_name="Research-Agent",
        system_prompt="You are a research assistant",
        model_name="gpt-4o-mini",
        max_loops=2,
        role="Researcher"
    ),
    Agent(
        agent_name="Analysis-Agent",
        system_prompt="You are a data analyst",
        model_name="gpt-4o-mini",
        max_loops=1,
        role="Analyst"
    )
]

info = get_agents_info(agents, team_name="Data Team")
print(info)
# Output:
# Available Agents for Team: Data Team
#
# [Agent 1]
# Name: Research-Agent
# Description: You are a research assistant
# Role: Researcher
# Model: gpt-4o-mini
# Max Loops: 2
#
# [Agent 2]
# Name: Analysis-Agent
# Description: You are a data analyst
# Role: Analyst
# Model: gpt-4o-mini
# Max Loops: 1
```

## Complete Usage Examples

### Advanced Multi-Agent Workflow Example

```python
import asyncio
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently,
    run_agents_with_different_tasks,
    batched_grid_agent_execution,
    get_agents_info
)

# Create specialized agents
agents = [
    Agent(
        agent_name="Market-Researcher",
        system_prompt="You are a market research expert specializing in consumer behavior",
        model_name="gpt-4o-mini",
        max_loops=1,
        role="Researcher"
    ),
    Agent(
        agent_name="Data-Analyst",
        system_prompt="You are a data analyst expert in statistical analysis",
        model_name="gpt-4o-mini",
        max_loops=1,
        role="Analyst"
    ),
    Agent(
        agent_name="Strategy-Consultant",
        system_prompt="You are a strategy consultant specializing in business development",
        model_name="gpt-4o-mini",
        max_loops=1,
        role="Consultant"
    ),
    Agent(
        agent_name="Financial-Advisor",
        system_prompt="You are a financial advisor specializing in investment strategies",
        model_name="gpt-4o-mini",
        max_loops=1,
        role="Advisor"
    )
]

# Display agent information
print("=== Agent Information ===")
print(get_agents_info(agents, "Business Intelligence Team"))
print("\n" + "="*50 + "\n")

# Example 1: Same task for all agents (basic concurrent execution)
print("=== Example 1: Concurrent Execution with Same Task ===")
task = "Analyze the impact of remote work trends on commercial real estate market in 2024"
results = run_agents_concurrently(agents, task, max_workers=4)

for i, result in enumerate(results):
    print(f"\n{agents[i].agent_name} Analysis:")
    print(f"Result: {result}")

# Example 1b: Same task with dictionary output format
print("\n=== Example 1b: Dictionary Output Format ===")
results_dict = run_agents_concurrently(
    agents, task, return_agent_output_dict=True, max_workers=4
)
for agent_name, result in results_dict.items():
    print(f"\n{agent_name} Analysis:")
    print(f"Result: {result}")

# Example 1c: With image support (if agents support it)
print("\n=== Example 1c: With Image Support ===")
# image_data = "base64_encoded_image_string"  # Uncomment if you have image data
# results_with_img = run_agents_concurrently(agents, task, img=image_data, max_workers=4)

print("\n" + "="*50 + "\n")

# Example 2: Different tasks for different agents
print("=== Example 2: Different Tasks for Different Agents ===")
agent_task_pairs = [
    (agents[0], "Research consumer preferences for electric vehicles"),
    (agents[1], "Analyze sales data for EV market penetration"),
    (agents[2], "Develop marketing strategy for EV adoption"),
    (agents[3], "Assess financial viability of EV charging infrastructure")
]

results = run_agents_with_different_tasks(agent_task_pairs, batch_size=2)

for i, result in enumerate(results):
    agent, task = agent_task_pairs[i]
    print(f"\n{agent.agent_name} - Task: {task}")
    print(f"Result: {result}")

print("\n" + "="*50 + "\n")

# Example 3: Grid execution with matched agents and tasks
print("=== Example 3: Batched Grid Execution ===")
grid_agents = agents[:3]  # Use first 3 agents
grid_tasks = [
    "Forecast market trends for renewable energy",
    "Evaluate risk factors in green technology investments",
    "Compare traditional vs sustainable investment portfolios"
]

grid_results = batched_grid_agent_execution(grid_agents, grid_tasks, max_workers=3)

for i, result in enumerate(grid_results):
    print(f"\nTask {i+1}: {grid_tasks[i]}")
    print(f"Agent: {grid_agents[i].agent_name}")
    print(f"Result: {result}")

print("\n=== Workflow Complete ===")
```

### Platform-Optimized Execution Example

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently_uvloop,
    run_agents_with_tasks_uvloop
)

# Create agents for high-performance execution
agents = [
    Agent(
        agent_name="High-Perf-Analyst-1",
        system_prompt="You are a high-performance data analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="High-Perf-Analyst-2",
        system_prompt="You are a high-performance data analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    ),
    Agent(
        agent_name="High-Perf-Analyst-3",
        system_prompt="You are a high-performance data analyst",
        model_name="gpt-4o-mini",
        max_loops=1
    )
]

# Example 1: Platform-optimized concurrent execution
print("=== Platform-Optimized Concurrent Execution ===")
task = "Perform high-speed data analysis on market trends"
results = run_agents_concurrently_uvloop(agents, task)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Agent {i+1} failed: {result}")
    else:
        print(f"Agent {i+1} result: {result}")

# Example 2: Platform-optimized execution with different tasks
print("\n=== Platform-Optimized Different Tasks ===")
tasks = [
    "Analyze Q1 financial performance",
    "Evaluate market volatility patterns",
    "Assess competitive landscape changes"
]

results = run_agents_with_tasks_uvloop(agents, tasks)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Agent {i+1} with {tasks[i]} failed: {result}")
    else:
        print(f"Task: {tasks[i]}")
        print(f"Result: {result}\n")

print("=== Platform-Optimized Execution Complete ===")
```

### Error Handling and Best Practices

```python
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create agents with error handling
agents = [
    Agent(
        agent_name=f"Agent-{i}",
        system_prompt="You are a helpful assistant",
        model_name="gpt-4o-mini",
        max_loops=1
    )
    for i in range(5)
]

task = "Perform a complex analysis task"

try:
    results = run_agents_concurrently(agents, task, max_workers=4)

    # Handle results (some may be exceptions)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Agent {i+1} failed with error: {result}")
        else:
            print(f"Agent {i+1} succeeded: {result}")

except Exception as e:
    print(f"Execution failed: {e}")

# Best practices:
# 1. Always handle exceptions in results
# 2. Use appropriate max_workers based on system resources
# 3. Monitor memory usage for large agent counts
# 4. Consider batch processing for very large numbers of agents
# 5. Use platform-optimized functions (uvloop/winloop) for I/O intensive tasks
# 6. Use return_agent_output_dict=True for structured, named results
# 7. Pass image data to agents that support multimodal processing
# 8. Leverage platform-specific optimizations automatically
```

## Performance Considerations

| Technique                         | Best Use Case / Description                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **ThreadPoolExecutor**            | Best for CPU-bound tasks with moderate I/O, supports image processing and flexible output formats    |
| **Platform-Specific Event Loops** | **uvloop** (Unix/Linux/macOS) and **winloop** (Windows) for significantly improved async performance |
| **Batch Processing**              | Prevents system overload with large numbers of agents, maintains order with grid execution           |
| **Resource Monitoring**           | Adjust worker counts based on system capabilities (defaults to 95% of CPU cores)                     |
| **Async/Await**                   | Use async functions for better concurrency control and platform optimizations                        |
| **Image Support**                 | Pass image data to agents that support multimodal processing for enhanced capabilities               |
| **Dictionary Output**             | Use `return_agent_output_dict=True` for structured results with agent name mapping                   |
| **Error Handling**                | All functions include comprehensive exception handling with graceful fallbacks                       |
