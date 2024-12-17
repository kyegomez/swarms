# AsyncWorkflow Documentation

The `AsyncWorkflow` class represents an asynchronous workflow that executes tasks concurrently using multiple agents. It allows for efficient task management, leveraging Python's `asyncio` for concurrent execution.

## Key Features
- **Concurrent Task Execution**: Distribute tasks across multiple agents asynchronously.
- **Configurable Workers**: Limit the number of concurrent workers (agents) for better resource management.
- **Autosave Results**: Optionally save the task execution results automatically.
- **Verbose Logging**: Enable detailed logging to monitor task execution.
- **Error Handling**: Gracefully handles exceptions raised by agents during task execution.

---

## Attributes
| Attribute         | Type                | Description                                                                 |
|-------------------|---------------------|-----------------------------------------------------------------------------|
| `name`            | `str`               | The name of the workflow.                                                   |
| `agents`          | `List[Agent]`       | A list of agents participating in the workflow.                             |
| `max_workers`     | `int`               | The maximum number of concurrent workers (default: 5).                      |
| `dashboard`       | `bool`              | Whether to display a dashboard (currently not implemented).                 |
| `autosave`        | `bool`              | Whether to autosave task results (default: `False`).                        |
| `verbose`         | `bool`              | Whether to enable detailed logging (default: `False`).                      |
| `task_pool`       | `List`              | A pool of tasks to be executed.                                             |
| `results`         | `List`              | A list to store results of executed tasks.                                  |
| `loop`            | `asyncio.EventLoop` | The event loop for asynchronous execution.                                  |

---

**Description**:
Initializes the `AsyncWorkflow` with specified agents, configuration, and options.

**Parameters**:
- `name` (`str`): Name of the workflow. Default: "AsyncWorkflow".
- `agents` (`List[Agent]`): A list of agents. Default: `None`.
- `max_workers` (`int`): The maximum number of workers. Default: `5`.
- `dashboard` (`bool`): Enable dashboard visualization (placeholder for future implementation).
- `autosave` (`bool`): Enable autosave of task results. Default: `False`.
- `verbose` (`bool`): Enable detailed logging. Default: `False`.
- `**kwargs`: Additional parameters for `BaseWorkflow`.

---

### `_execute_agent_task`
```python
async def _execute_agent_task(self, agent: Agent, task: str) -> Any:
```
**Description**:
Executes a single task asynchronously using a given agent.

**Parameters**:
- `agent` (`Agent`): The agent responsible for executing the task.
- `task` (`str`): The task to be executed.

**Returns**:
- `Any`: The result of the task execution or an error message in case of an exception.

**Example**:
```python
result = await workflow._execute_agent_task(agent, "Sample Task")
```

---

### `run`
```python
async def run(self, task: str) -> List[Any]:
```
**Description**:
Executes the specified task concurrently across all agents.

**Parameters**:
- `task` (`str`): The task to be executed by all agents.

**Returns**:
- `List[Any]`: A list of results or error messages returned by the agents.

**Raises**:
- `ValueError`: If no agents are provided in the workflow.

**Example**:
```python
import asyncio

agents = [Agent("Agent1"), Agent("Agent2")]
workflow = AsyncWorkflow(agents=agents, verbose=True)

results = asyncio.run(workflow.run("Process Data"))
print(results)
```

---

## Production-Grade Financial Example: Multiple Agents
### Example: Stock Analysis and Investment Strategy
```python
import asyncio
from swarms import Agent, AsyncWorkflow
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT

# Initialize multiple Financial Agents
portfolio_analysis_agent = Agent(
    agent_name="Portfolio-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    autosave=True,
    verbose=True,
)

stock_strategy_agent = Agent(
    agent_name="Stock-Strategy-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    autosave=True,
    verbose=True,
)

risk_management_agent = Agent(
    agent_name="Risk-Management-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    autosave=True,
    verbose=True,
)

# Create a workflow with multiple agents
workflow = AsyncWorkflow(
    name="Financial-Workflow",
    agents=[portfolio_analysis_agent, stock_strategy_agent, risk_management_agent],
    verbose=True,
)

# Run the workflow
async def main():
    task = "Analyze the current stock market trends and provide an investment strategy with risk assessment."
    results = await workflow.run(task)
    for agent_result in results:
        print(agent_result)

asyncio.run(main())
```

**Output**:
```
INFO: Agent Portfolio-Analysis-Agent processing task: Analyze the current stock market trends and provide an investment strategy with risk assessment.
INFO: Agent Stock-Strategy-Agent processing task: Analyze the current stock market trends and provide an investment strategy with risk assessment.
INFO: Agent Risk-Management-Agent processing task: Analyze the current stock market trends and provide an investment strategy with risk assessment.
INFO: Agent Portfolio-Analysis-Agent completed task
INFO: Agent Stock-Strategy-Agent completed task
INFO: Agent Risk-Management-Agent completed task
Results:
- Detailed portfolio analysis...
- Stock investment strategies...
- Risk assessment insights...
```

---

## Notes
1. **Autosave**: The autosave functionality is a placeholder. Users can implement custom logic to save `self.results`.
2. **Error Handling**: Exceptions raised by agents are logged and returned as part of the results.
3. **Dashboard**: The `dashboard` feature is currently not implemented but can be extended for visualization.

---

## Dependencies
- `asyncio`: Python's asynchronous I/O framework.
- `loguru`: Logging utility for better log management.
- `swarms`: Base components (`BaseWorkflow`, `Agent`).

---

## Future Extensions
- **Dashboard**: Implement a real-time dashboard for monitoring agent performance.
- **Autosave**: Add persistent storage support for task results.
- **Task Management**: Extend task pooling and scheduling logic to support dynamic workloads.

---

## License
This class is part of the `swarms` framework and follows the framework's licensing terms.
