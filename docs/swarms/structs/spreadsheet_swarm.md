# SpreadSheetSwarm Documentation

## Overview

The `SpreadSheetSwarm` is a concurrent processing system that manages multiple agents to execute tasks simultaneously. It supports both pre-configured agents and CSV-based agent loading, with automatic metadata tracking and file output capabilities.


## Full Path

```python
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm
```


## Constructor

### `__init__`

```python
def __init__(
    self,
    name: str = "Spreadsheet-Swarm",
    description: str = "A swarm that processes tasks concurrently using multiple agents and saves the metadata to a CSV file.",
    agents: List[AgentType] = None,
    autosave: bool = True,
    save_file_path: str = None,
    max_loops: int = 1,
    load_path: str = None,
    verbose: bool = False,
    *args,
    **kwargs,
):
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"Spreadsheet-Swarm"` | The name of the swarm |
| `description` | `str` | `"A swarm that processes tasks concurrently using multiple agents and saves the metadata to a CSV file."` | Description of the swarm's purpose |
| `agents` | `List[AgentType]` | `None` | List of agents participating in the swarm. If `None`, agents will be loaded from `load_path` |
| `autosave` | `bool` | `True` | Whether to enable autosave of swarm metadata |
| `save_file_path` | `str` | `None` | File path to save swarm metadata as CSV. If `None`, auto-generated based on workspace_dir |
| `max_loops` | `int` | `1` | Number of times to repeat the swarm tasks |
| `load_path` | `str` | `None` | Path to CSV file containing agent configurations. Required if `agents` is `None` |
| `verbose` | `bool` | `False` | Whether to enable verbose logging |
| `*args` | `Any` | - | Additional positional arguments |
| `**kwargs` | `Any` | - | Additional keyword arguments |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | The name of the swarm |
| `description` | `str` | Description of the swarm's purpose |
| `agents` | `List[AgentType]` | List of agents participating in the swarm |
| `autosave` | `bool` | Whether autosave is enabled |
| `save_file_path` | `str` | File path where swarm metadata is saved |
| `max_loops` | `int` | Number of times to repeat tasks |
| `load_path` | `str` | Path to CSV file for agent configurations |
| `verbose` | `bool` | Whether verbose logging is enabled |
| `outputs` | `List[Dict]` | List of completed task outputs |
| `tasks_completed` | `int` | Counter for completed tasks |
| `agent_tasks` | `Dict[str, str]` | Mapping of agent names to their tasks |

#### Note

Either `agents` or `load_path` must be provided. If both are provided, `agents` will be used.

**Workspace Directory**: The `workspace_dir` is automatically set from the `WORKSPACE_DIR` environment variable. Ensure this environment variable is set before initializing the swarm.

---

## Methods

### `reliability_check`

```python
def reliability_check(self) -> None
```

#### Description

Performs reliability checks to ensure the swarm is properly configured before processing tasks. Verifies that agents are provided and max_loops is set.

#### Raises

- **`ValueError`**: If no agents are provided or if max_loops is not provided.

#### Example

```python
swarm = SpreadSheetSwarm(agents=[agent1, agent2])
swarm.reliability_check()
```

---

### `run`

```python
def run(self, task: str = None, *args, **kwargs) -> Dict[str, Any]
```

#### Run Description

Main method to run the swarm with a specified task or using configured tasks. Handles both single task execution and CSV-based configuration.

#### Run Parameters

- **`task`** (`str`, optional): The task to be executed by all agents. If `None`, uses tasks from config.
- **`*args`**: Additional positional arguments to pass to the agents.
- **`**kwargs`**: Additional keyword arguments to pass to the agents.

#### Run Returns

- **`Dict[str, Any]`**: Summary of the swarm execution containing run_id, name, description, start_time, end_time, tasks_completed, number_of_agents, and outputs.

#### Run Example

```python
swarm = SpreadSheetSwarm(agents=[agent1, agent2])
result = swarm.run("Process Data")
print(result)
```

---

### `run_from_config`

```python
def run_from_config(self) -> Dict[str, Any]
```

#### Run From Config Description

Runs all agents with their configured tasks concurrently. Loads agents from CSV if needed and executes tasks based on the agent-task mapping.

#### Run From Config Returns

- **`Dict[str, Any]`**: Summary of the swarm execution.

#### Run From Config Example

```python
swarm = SpreadSheetSwarm(load_path="agents.csv")
result = swarm.run_from_config()
```

---

### `load_from_csv`

```python
def load_from_csv(self) -> None
```

#### Load From CSV Description

Loads agent configurations from a CSV file. Expected CSV format includes columns: agent_name, description, system_prompt, task, model_name, docs, max_loops, user_name, stopping_token.

#### Load From CSV Example

```python
swarm = SpreadSheetSwarm(load_path="agents.csv")
swarm.load_from_csv()
```

---

### `export_to_json`

```python
def export_to_json(self) -> str
```

#### Export To JSON Description

Exports the swarm outputs to JSON format. Useful for external system integration or logging purposes.

#### Export To JSON Returns

- **`str`**: JSON representation of the swarm's metadata.

#### Export To JSON Example

```python
json_data = swarm.export_to_json()
print(json_data)
```

---

### `data_to_json_file`

```python
def data_to_json_file(self) -> None
```

#### Data To JSON File Description

Saves the swarm's metadata as a JSON file in the specified workspace directory. File name is generated using the swarm's name and run ID.

#### Data To JSON File Example

```python
swarm.data_to_json_file()
```

---

### `_track_output`

```python
def _track_output(self, agent_name: str, task: str, result: str) -> None
```

#### Track Output Description

Internal method to track the output of a completed task. Updates the outputs list and increments the tasks_completed counter.

#### Track Output Parameters

- **`agent_name`** (`str`): The name of the agent that completed the task.
- **`task`** (`str`): The task that was completed.
- **`result`** (`str`): The result of the completed task.

#### Track Output Example

```python
swarm._track_output("Agent1", "Process Data", "Success")
```

---

### `_save_to_csv`

```python
def _save_to_csv(self) -> None
```

#### Save To CSV Description

Saves the swarm's metadata to a CSV file. Creates the file with headers if it doesn't exist, then appends task results.

#### Save To CSV Example

```python
swarm._save_to_csv()
```

---

## Usage Examples

### Example 1: Basic Financial Analysis Swarm

```python
from swarms import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Example 1: Using pre-configured agents
agents = [
    Agent(
        agent_name="Research-Agent",
        agent_description="Specialized in market research and analysis",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        streaming_on=False,
    ),
    Agent(
        agent_name="Technical-Agent",
        agent_description="Expert in technical analysis and trading strategies",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        streaming_on=False,
    ),
    Agent(
        agent_name="Risk-Agent",
        agent_description="Focused on risk assessment and portfolio management",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        streaming_on=False,
    ),
]

# Initialize the SpreadSheetSwarm with agents
swarm = SpreadSheetSwarm(
    name="Financial-Analysis-Swarm",
    description="A swarm of specialized financial analysis agents",
    agents=agents,
    max_loops=1,
    autosave=False,
)

# Run all agents with the same task
task = "What are the top 3 energy stocks to invest in for 2024? Provide detailed analysis."
result = swarm.run(task=task)

print(result)
```

### Example 2: CSV-Based Agent Configuration

```python
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Create a CSV file with agent configurations
csv_content = """agent_name,description,system_prompt,task,model_name
Research-Agent,Market research specialist,You are a market research expert.,Analyze market trends,claude-sonnet-4-20250514
Technical-Agent,Technical analysis expert,You are a technical analysis expert.,Perform technical analysis,claude-sonnet-4-20250514
Risk-Agent,Risk assessment specialist,You are a risk assessment expert.,Evaluate investment risks,claude-sonnet-4-20250514"""

with open("agents.csv", "w") as f:
    f.write(csv_content)

# Initialize swarm with CSV configuration
swarm = SpreadSheetSwarm(
    name="CSV-Configured-Swarm",
    description="A swarm loaded from CSV configuration",
    load_path="agents.csv",
    max_loops=1,
    autosave=True,
)

# Run agents with their configured tasks
result = swarm.run_from_config()
print(result)
```

### Example 3: Multi-Loop Task Execution

```python
from swarms import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Create specialized agents
agents = [
    Agent(
        agent_name="Content-Agent",
        agent_description="Content creation specialist",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
    ),
    Agent(
        agent_name="SEO-Agent",
        agent_description="SEO optimization expert",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
    ),
]

# Initialize swarm with multiple loops
swarm = SpreadSheetSwarm(
    name="Content-Creation-Swarm",
    description="A swarm for content creation and optimization",
    agents=agents,
    max_loops=3,  # Each agent will run the task 3 times
    autosave=True,
)

# Run the same task multiple times
task = "Create a blog post about AI trends in 2024"
result = swarm.run(task=task)

print(f"Tasks completed: {result['tasks_completed']}")
print(f"Number of agents: {result['number_of_agents']}")
```

### Example 4: JSON Export and Metadata Tracking

```python
from swarms import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

agents = [
    Agent(
        agent_name="Data-Analyst",
        agent_description="Data analysis specialist",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
    ),
]

swarm = SpreadSheetSwarm(
    name="Data-Analysis-Swarm",
    description="A swarm for data analysis tasks",
    agents=agents,
    max_loops=1,
    autosave=True,
)

# Run the task
result = swarm.run("Analyze the provided dataset and generate insights")

# Export to JSON
json_data = swarm.export_to_json()
print("JSON Export:")
print(json_data)

# Save metadata to JSON file
swarm.data_to_json_file()
print("Metadata saved to JSON file")
```

---

## Additional Information and Tips

| Tip/Feature            | Description                                                                                                                                                                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Concurrent Execution** | The swarm uses `run_agents_with_different_tasks` for concurrent execution of multiple agents. This provides better performance compared to sequential execution.                                                              |
| **Autosave Feature**       | When `autosave=True`, the swarm automatically saves metadata to both CSV and JSON files. CSV files are saved with unique run IDs to prevent conflicts.                                                                      |
| **CSV Configuration**      | You can load agent configurations from CSV files with columns: agent_name, description, system_prompt, task, model_name, docs, max_loops, user_name, stopping_token.                                                      |
| **Workspace Management**   | The swarm automatically creates workspace directories and generates unique file names using UUIDs and timestamps to prevent file conflicts.                                                                                |
| **Error Handling**         | The `run` method includes try-catch error handling and logging. Check the logs for detailed error information if execution fails.                                                                                        |
| **Metadata Tracking**      | All task executions are tracked with timestamps, agent names, tasks, and results. This data is available in both the return value and saved files.                                                                        |
| **Flexible Task Assignment** | You can either provide a single task for all agents or use CSV configuration to assign different tasks to different agents.                                                                                            |

## CSV File Format

When using CSV-based agent configuration, the file should have the following columns:

| Column | Required | Description | Default Value |
|--------|----------|-------------|---------------|
| `agent_name` | Yes | Unique name for the agent | - |
| `description` | Yes | Description of the agent's purpose | - |
| `system_prompt` | Yes | System prompt for the agent | - |
| `task` | Yes | Task to be executed by the agent | - |
| `model_name` | No | Model to use for the agent | `"openai/gpt-4o"` |
| `docs` | No | Documentation for the agent | `""` |
| `max_loops` | No | Maximum loops for the agent | `1` |
| `user_name` | No | Username for the agent | `"user"` |
| `stopping_token` | No | Token to stop agent execution | `None` |

## Return Value Structure

The `run` method returns a dictionary with the following structure:

```python
{
    "run_id": str,           # Unique identifier for this run
    "name": str,             # Name of the swarm
    "description": str,      # Description of the swarm
    "start_time": str,       # ISO timestamp of start time
    "end_time": str,         # ISO timestamp of end time
    "tasks_completed": int,  # Number of tasks completed
    "number_of_agents": int, # Number of agents in the swarm
    "outputs": List[Dict]    # List of task outputs with agent_name, task, result, timestamp
}
```

---
