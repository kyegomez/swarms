# CronJob

A wrapper class that turns any callable (including Swarms agents) into a scheduled cron job using the schedule library with cron-style scheduling.

Full Path `from swarms.structs.cron_job`

## Class Definition

```python
class CronJob:
    def __init__(
        self,
        agent: Optional[Union[Any, Callable]] = None,
        interval: Optional[str] = None,
        job_id: Optional[str] = None,
        callback: Optional[Callable[[Any, str, dict], Any]] = None,
    ) -> None
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Optional[Union[Any, Callable]]` | `None` | The Swarms Agent instance or callable to be scheduled |
| `interval` | `Optional[str]` | `None` | Interval string in format "Xunit" (e.g., "5seconds", "10minutes", "1hour") |
| `job_id` | `Optional[str]` | `None` | Unique identifier for the job. Auto-generated if not provided |
| `callback` | `Optional[Callable[[Any, str, dict], Any]]` | `None` | Function to customize output processing |

## Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `agent` | `Union[Any, Callable]` | The scheduled agent or callable |
| `interval` | `str` | The scheduling interval string |
| `job_id` | `str` | Unique job identifier |
| `is_running` | `bool` | Current execution status |
| `thread` | `Optional[threading.Thread]` | Background execution thread |
| `schedule` | `schedule.Scheduler` | Internal scheduler instance |
| `callback` | `Optional[Callable[[Any, str, dict], Any]]` | Output customization function |
| `execution_count` | `int` | Number of executions completed |
| `start_time` | `Optional[float]` | Job start timestamp |

## Methods

### `run(task: str, **kwargs) -> Any`

Schedules and starts the cron job execution.

**Parameters:**
- `task` (`str`): Task string to be executed by the agent
- `**kwargs` (`dict`): Additional parameters passed to agent's run method

**Returns:**
- `Any`: Result of the cron job execution

**Raises:**
- `CronJobConfigError`: If agent or interval is not configured
- `CronJobExecutionError`: If task execution fails

### `__call__(task: str, **kwargs) -> Any`

Callable interface for the CronJob instance.

**Parameters:**
- `task` (`str`): Task string to be executed
- `**kwargs` (`dict`): Additional parameters passed to agent's run method

**Returns:**
- `Any`: Result of the task execution

### `start() -> None`

Starts the scheduled job in a separate thread.

**Raises:**
- `CronJobExecutionError`: If the job fails to start

### `stop() -> None`

Stops the scheduled job gracefully.

**Raises:**
- `CronJobExecutionError`: If the job fails to stop properly

### `set_callback(callback: Callable[[Any, str, dict], Any]) -> None`

Sets or updates the callback function for output customization.

**Parameters:**
- `callback` (`Callable[[Any, str, dict], Any]`): Function to customize output processing

### `get_execution_stats() -> Dict[str, Any]`

Retrieves execution statistics for the cron job.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `job_id` (`str`): Unique job identifier
  - `is_running` (`bool`): Current execution status
  - `execution_count` (`int`): Number of executions completed
  - `start_time` (`Optional[float]`): Job start timestamp
  - `uptime` (`float`): Seconds since job started
  - `interval` (`str`): Scheduled execution interval

## Callback Function Signature

```python
def callback_function(
    output: Any,           # Original output from the agent
    task: str,             # Task that was executed
    metadata: dict         # Job execution metadata
) -> Any:                  # Customized output (any type)
    pass
```

### Callback Metadata Dictionary

| Key | Type | Description |
|-----|------|-------------|
| `job_id` | `str` | Unique job identifier |
| `timestamp` | `float` | Execution timestamp (Unix time) |
| `execution_count` | `int` | Sequential execution number |
| `task` | `str` | The task string that was executed |
| `kwargs` | `dict` | Additional parameters passed to agent |
| `start_time` | `Optional[float]` | Job start timestamp |
| `is_running` | `bool` | Current job status |

## Interval Format

The `interval` parameter accepts strings in the format `"Xunit"`:

| Unit | Examples | Description |
|------|----------|-------------|
| `seconds` | `"5seconds"`, `"30seconds"` | Execute every X seconds |
| `minutes` | `"1minute"`, `"15minutes"` | Execute every X minutes |
| `hours` | `"1hour"`, `"6hours"` | Execute every X hours |

## Exceptions

### `CronJobError`
Base exception class for all CronJob errors.

### `CronJobConfigError`
Raised for configuration errors (invalid agent, interval format, etc.).

### `CronJobScheduleError`
Raised for scheduling-related errors.

### `CronJobExecutionError`
Raised for execution-related errors (start/stop failures, task execution failures).

## Type Definitions

```python
from typing import Any, Callable, Dict, Optional, Union

# Agent type can be any callable or object with run method
AgentType = Union[Any, Callable]

# Callback function signature
CallbackType = Callable[[Any, str, Dict[str, Any]], Any]

# Execution statistics return type
StatsType = Dict[str, Any]
```

## Quick Start Examples

### Basic Usage

```python
from swarms import Agent, CronJob

# Simple agent cron job
agent = Agent(agent_name="MyAgent", ...)
cron_job = CronJob(agent=agent, interval="30seconds")
cron_job.run("Analyze market trends")
```

### With Custom Function

```python
def my_task(task: str) -> str:
    return f"Completed: {task}"

cron_job = CronJob(agent=my_task, interval="1minute")
cron_job.run("Process data")
```

### With Callback

```python
def callback(output, task, metadata):
    return {"result": output, "count": metadata["execution_count"]}

cron_job = CronJob(
    agent=agent, 
    interval="30seconds", 
    callback=callback
)
```


## Full Examples

### Complete Agent with Callback

```python
from swarms import Agent, CronJob
from datetime import datetime
import json

# Create agent
agent = Agent(
    agent_name="Financial-Analyst",
    system_prompt="You are a financial analyst. Analyze market data and provide insights.",
    model_name="gpt-4o-mini",
    max_loops=1
)

# Advanced callback with monitoring
class AdvancedCallback:
    def __init__(self):
        self.history = []
        self.error_count = 0
        
    def __call__(self, output, task, metadata):
        # Track execution
        execution_data = {
            "output": output,
            "execution_id": metadata["execution_count"],
            "timestamp": datetime.fromtimestamp(metadata["timestamp"]).isoformat(),
            "task": task,
            "job_id": metadata["job_id"],
            "success": bool(output and "error" not in str(output).lower())
        }
        
        if not execution_data["success"]:
            self.error_count += 1
            
        self.history.append(execution_data)
        
        # Keep only last 100 executions
        if len(self.history) > 100:
            self.history.pop(0)
            
        return execution_data
    
    def get_stats(self):
        return {
            "total_executions": len(self.history),
            "error_count": self.error_count,
            "success_rate": (len(self.history) - self.error_count) / len(self.history) if self.history else 0
        }

# Use advanced callback
callback = AdvancedCallback()
cron_job = CronJob(
    agent=agent,
    interval="2minutes",
    job_id="financial_analysis_job",
    callback=callback
)

# Run the cron job
try:
    cron_job.run("Analyze current market conditions and provide investment recommendations")
except KeyboardInterrupt:
    cron_job.stop()
    print("Final stats:", json.dumps(callback.get_stats(), indent=2))
```

### Multi-Agent Workflow with CronJob

```python
from swarms import Agent, CronJob, ConcurrentWorkflow
import json

# Create specialized agents
bitcoin_agent = Agent(
    agent_name="Bitcoin-Analyst",
    system_prompt="You are a Bitcoin specialist. Focus only on Bitcoin analysis.",
    model_name="gpt-4o-mini",
    max_loops=1
)

ethereum_agent = Agent(
    agent_name="Ethereum-Analyst", 
    system_prompt="You are an Ethereum specialist. Focus only on Ethereum analysis.",
    model_name="gpt-4o-mini",
    max_loops=1
)

# Create concurrent workflow
workflow = ConcurrentWorkflow(
    name="Crypto-Analysis-Workflow",
    agents=[bitcoin_agent, ethereum_agent],
    max_loops=1
)

# Workflow callback
def workflow_callback(output, task, metadata):
    """Process multi-agent workflow output."""
    return {
        "workflow_results": output,
        "execution_id": metadata["execution_count"],
        "timestamp": metadata["timestamp"],
        "agents_count": len(workflow.agents),
        "task": task,
        "metadata": {
            "job_id": metadata["job_id"],
            "uptime": metadata.get("uptime", 0)
        }
    }

# Create workflow cron job
workflow_cron = CronJob(
    agent=workflow,
    interval="5minutes",
    job_id="crypto_workflow_job",
    callback=workflow_callback
)

# Run workflow cron job
workflow_cron.run("Analyze your assigned cryptocurrency and provide market insights")
```

### API Integration Example

```python
import requests
from swarms import Agent, CronJob
import json

# Create agent
agent = Agent(
    agent_name="News-Analyst",
    system_prompt="Analyze news and provide summaries.",
    model_name="gpt-4o-mini",
    max_loops=1
)

# API webhook callback
def api_callback(output, task, metadata):
    """Send results to external API."""
    payload = {
        "data": output,
        "source": "swarms_cronjob",
        "job_id": metadata["job_id"],
        "execution_id": metadata["execution_count"],
        "timestamp": metadata["timestamp"],
        "task": task
    }
    
    try:
        # Send to webhook (replace with your URL)
        response = requests.post(
            "https://api.example.com/webhook",
            json=payload,
            timeout=30
        )
        
        return {
            "original_output": output,
            "api_status": "sent",
            "api_response_code": response.status_code,
            "execution_id": metadata["execution_count"]
        }
    except requests.RequestException as e:
        return {
            "original_output": output,
            "api_status": "failed",
            "error": str(e),
            "execution_id": metadata["execution_count"]
        }

# Database logging callback
def db_callback(output, task, metadata):
    """Log to database (pseudo-code)."""
    # db.execute(
    #     "INSERT INTO cron_results (job_id, output, timestamp) VALUES (?, ?, ?)",
    #     (metadata["job_id"], output, metadata["timestamp"])
    # )
    
    return {
        "output": output,
        "logged_to_db": True,
        "execution_id": metadata["execution_count"]
    }

# Create cron job with API integration
api_cron_job = CronJob(
    agent=agent,
    interval="10minutes",
    job_id="news_analysis_api_job",
    callback=api_callback
)

# Dynamic callback switching example
db_cron_job = CronJob(
    agent=agent,
    interval="1hour",
    job_id="news_analysis_db_job"
)

# Start with API callback
db_cron_job.set_callback(api_callback)

# Later switch to database callback
# db_cron_job.set_callback(db_callback)

# Get execution statistics
stats = db_cron_job.get_execution_stats()
print(f"Job statistics: {json.dumps(stats, indent=2)}")
``` 