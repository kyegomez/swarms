# `AgentRearrange` Class

The `AgentRearrange` class represents a swarm of agents for task rearrangement and orchestration. It enables complex workflows where multiple agents can work sequentially or concurrently based on a defined flow pattern. The class supports both sequential execution (using '->') and concurrent execution (using ',') within the same workflow, and includes **sequential awareness** features that allow agents to know about the agents ahead and behind them in sequential flows.

## Key Features

- Sequential and concurrent agent execution
- Custom flow patterns with arrow (->) and comma (,) syntax
- Team awareness and sequential flow information
- Human-in-the-loop integration
- Memory system support
- Batch and concurrent processing capabilities
- Comprehensive error handling and logging

## Flow Syntax

- Use '->' to define sequential execution: `"agent1 -> agent2 -> agent3"`
- Use ',' to define concurrent execution: `"agent1, agent2 -> agent3"`
- Combine both: `"agent1 -> agent2, agent3 -> agent4"`
- Use 'H' for human-in-the-loop: `"agent1 -> H -> agent2"`

## Attributes
----------

| Attribute | Type | Description |
| --- | --- | --- |
| `id` | `str` | Unique identifier for the swarm |
| `name` | `str` | Name of the swarm |
| `description` | `str` | Description of the swarm's purpose |
| `agents` | `dict` | Dictionary mapping agent names to Agent objects |
| `flow` | `str` | Flow pattern defining task execution order |
| `max_loops` | `int` | Maximum number of execution loops |
| `verbose` | `bool` | Whether to enable verbose logging |
| `memory_system` | `Any` | Optional memory system for persistence |
| `human_in_the_loop` | `bool` | Whether human intervention is enabled |
| `custom_human_in_the_loop` | `Callable` | Custom function for human intervention |
| `output_type` | `OutputType` | Format of output ("all", "final", "list", or "dict") |
| `autosave` | `bool` | Whether to automatically save agent data |
| `rules` | `str` | Custom rules to add to the conversation |
| `team_awareness` | `bool` | Whether to enable team awareness and sequential flow information |
| `time_enabled` | `bool` | Whether to enable timestamps in conversation |
| `message_id_on` | `bool` | Whether to enable message IDs in conversation |
| `conversation` | `Conversation` | Conversation history management |

## Methods
-------

### `__init__(self, id: str = swarm_id(), name: str = "AgentRearrange", description: str = "A swarm of agents for rearranging tasks.", agents: List[Union[Agent, Callable]] = None, flow: str = None, max_loops: int = 1, verbose: bool = True, memory_system: Any = None, human_in_the_loop: bool = False, custom_human_in_the_loop: Optional[Callable[[str], str]] = None, output_type: OutputType = "all", autosave: bool = True, rules: str = None, team_awareness: bool = False, time_enabled: bool = False, message_id_on: bool = False, *args, **kwargs)`

Initializes the `AgentRearrange` object with enhanced sequential awareness capabilities.

**Note:**
-   The `reliability_check()` method is automatically called during initialization to validate all critical parameters (agents, max_loops, flow, output_type).
-   If validation fails, a `ValueError` will be raised before the object is created.

| Parameter | Type | Description |
| --- | --- | --- |
| `id` | `str` (optional) | Unique identifier for the swarm. Defaults to auto-generated UUID. |
| `name` | `str` (optional) | Name of the swarm. Defaults to "AgentRearrange". |
| `description` | `str` (optional) | Description of the swarm's purpose. Defaults to "A swarm of agents for rearranging tasks.". |
| `agents` | `List[Union[Agent, Callable]]` (optional) | A list of `Agent` objects or callables. Defaults to `None`. |
| `flow` | `str` (optional) | The flow pattern of the tasks. Defaults to `None`. |
| `max_loops` | `int` (optional) | The maximum number of loops for the agents to run. Defaults to `1`. |
| `verbose` | `bool` (optional) | Whether to enable verbose logging or not. Defaults to `True`. |
| `memory_system` | `Any` (optional) | Memory system for storing agent interactions. Defaults to `None`. |
| `human_in_the_loop` | `bool` (optional) | Whether human intervention is enabled. Defaults to `False`. |
| `custom_human_in_the_loop` | `Callable[[str], str]` (optional) | Custom function for human intervention. Defaults to `None`. |
| `output_type` | `OutputType` (optional) | Format of output. Defaults to `"all"`. |
| `autosave` | `bool` (optional) | Whether to automatically save agent data. Defaults to `True`. |
| `rules` | `str` (optional) | Custom rules to add to the conversation. Defaults to `None`. |
| `team_awareness` | `bool` (optional) | Whether to enable team awareness and sequential flow information. Defaults to `False`. |
| `time_enabled` | `bool` (optional) | Whether to enable timestamps in conversation. Defaults to `False`. |
| `message_id_on` | `bool` (optional) | Whether to enable message IDs in conversation. Defaults to `False`. |

### `add_agent(self, agent: Agent)`

Adds an agent to the swarm.

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `Agent` | The agent to be added. |

### `remove_agent(self, agent_name: str)`

Removes an agent from the swarm.

| Parameter | Type | Description |
| --- | --- | --- |
| `agent_name` | `str` | The name of the agent to be removed. |

### `add_agents(self, agents: List[Agent])`

Adds multiple agents to the swarm.

| Parameter | Type | Description |
| --- | --- | --- |
| `agents` | `List[Agent]` | A list of `Agent` objects. |

### `reliability_check(self)`

Validates the configuration parameters to ensure the system can run properly. This method is called automatically during initialization to ensure the system is properly configured before execution.

**Raises:**

-   `ValueError`: If any of the following conditions are met:
    - agents list is None or empty
    - max_loops is 0
    - flow is None or empty string
    - output_type is None or empty string

**Note:**
-   This method is called automatically during `__init__` to prevent runtime errors.
-   You can call this method manually to validate configuration after making changes.

### `set_custom_flow(self, flow: str)`

Sets a custom flow pattern for agent execution. Allows dynamic modification of the execution flow after initialization.

| Parameter | Type | Description |
| --- | --- | --- |
| `flow` | `str` | The new flow pattern to use for agent execution. Must follow the syntax: "agent1 -> agent2, agent3 -> agent4" |

**Example:**
```python
rearrange_system.set_custom_flow("researcher -> writer, editor")
```

**Note:**
-   The flow will be validated on the next execution. If invalid, a `ValueError` will be raised during the `run()` method.

### `track_history(self, agent_name: str, result: str)`

Tracks the execution history for a specific agent. Records the result of an agent's execution in the swarm history for later analysis or debugging purposes.

| Parameter | Type | Description |
| --- | --- | --- |
| `agent_name` | `str` | The name of the agent whose result to track. |
| `result` | `str` | The result/output from the agent's execution. |

**Note:**
-   This method is typically called internally during agent execution to maintain a complete history of all agent activities.
-   Can be called manually to add custom history entries.

### `validate_flow(self)`

Validates the flow pattern.

**Raises:**

-   `ValueError`: If the flow pattern is incorrectly formatted or contains duplicate agent names.

**Returns:**

-   `bool`: `True` if the flow pattern is valid.

### **Sequential Awareness Methods**

#### `get_agent_sequential_awareness(self, agent_name: str) -> str`

Gets the sequential awareness information for a specific agent, showing which agents come before and after in the sequence.

| Parameter | Type | Description |
| --- | --- | --- |
| `agent_name` | `str` | The name of the agent to get awareness for. |

**Returns:**

-   `str`: A string describing the agents ahead and behind in the sequence.

**Example:**
```python
awareness = agent_system.get_agent_sequential_awareness("Agent2")
# Returns: "Sequential awareness: Agent ahead: Agent1 | Agent behind: Agent3"
```

#### `get_sequential_flow_structure(self) -> str`

Gets the overall sequential flow structure information showing the complete workflow with relationships between agents.

**Returns:**

-   `str`: A string describing the complete sequential flow structure.

**Example:**
```python
flow_structure = agent_system.get_sequential_flow_structure()
# Returns: "Sequential Flow Structure:
# Step 1: Agent1
# Step 2: Agent2 (follows: Agent1) (leads to: Agent3)
# Step 3: Agent3 (follows: Agent2)"
```

### `run(self, task: str = None, img: str = None, *args, **kwargs)`

Executes the agent rearrangement task with comprehensive logging and error handling. This is the main public method for executing tasks through the agent rearrange system.

| Parameter | Type | Description |
| --- | --- | --- |
| `task` | `str` (optional) | The task to execute through the agent workflow. Defaults to `None`. |
| `img` | `str` (optional) | Path to input image if required by any agents. Defaults to `None`. |
| `*args` | - | Additional positional arguments passed to the internal `_run()` method. |
| `**kwargs` | - | Additional keyword arguments passed to the internal `_run()` method. |

**Returns:**

-   The result from executing the task through the agent rearrange system. The format depends on the configured `output_type`.

**Note:**
-   This method automatically logs agent data before and after execution for telemetry and debugging purposes.
-   Any exceptions are caught and handled by the `_catch_error()` method.

### `__call__(self, task: str, *args, **kwargs)`

Makes the class callable by executing the `run()` method. Enables the `AgentRearrange` instance to be called directly as a function.

| Parameter | Type | Description |
| --- | --- | --- |
| `task` | `str` | The task to execute through the agent workflow. |
| `*args` | - | Additional positional arguments passed to `run()`. |
| `**kwargs` | - | Additional keyword arguments passed to `run()`. |

**Returns:**

-   The result from executing the task through the agent rearrange system.

**Example:**
```python
rearrange_system = AgentRearrange(agents=[agent1, agent2], flow="agent1 -> agent2")
result = rearrange_system("Process this data")
```

### `batch_run(self, tasks: List[str], img: Optional[List[str]] = None, batch_size: int = 10, *args, **kwargs)`

Process multiple tasks in batches.

| Parameter | Type | Description |
| --- | --- | --- |
| `tasks` | `List[str]` | List of tasks to process |
| `img` | `List[str]` (optional) | Optional list of images corresponding to tasks |
| `batch_size` | `int` | Number of tasks to process simultaneously |
| `*args` | - | Additional positional arguments |
| `**kwargs` | - | Additional keyword arguments |

**Returns:**

-   `List[str]`: List of results corresponding to input tasks

### `concurrent_run(self, tasks: List[str], img: Optional[List[str]] = None, max_workers: Optional[int] = None, *args, **kwargs)`

Process multiple tasks concurrently using ThreadPoolExecutor. This method enables true parallel processing of multiple tasks by using Python's ThreadPoolExecutor to run tasks simultaneously across multiple threads.

| Parameter | Type | Description |
| --- | --- | --- |
| `tasks` | `List[str]` | List of tasks to process through the agent workflow |
| `img` | `List[str]` (optional) | Optional list of images corresponding to tasks. Must be the same length as tasks list. Defaults to `None`. |
| `max_workers` | `int` (optional) | Maximum number of worker threads to use. If `None`, uses the default ThreadPoolExecutor behavior. Defaults to `None`. |
| `*args` | - | Additional positional arguments passed to individual task execution |
| `**kwargs` | - | Additional keyword arguments passed to individual task execution |

**Returns:**

-   `List[str]`: List of results corresponding to input tasks in the same order

**Note:**
-   This method uses ThreadPoolExecutor for true parallel execution.
-   The number of concurrent executions is limited by `max_workers` parameter.
-   Each task runs independently through the full agent workflow.

### `to_dict(self) -> Dict[str, Any]`

Converts all attributes of the class, including callables, into a dictionary. This method provides a comprehensive serialization of the `AgentRearrange` instance, converting all attributes into a dictionary format suitable for storage, logging, or transmission.

**Returns:**

-   `Dict[str, Any]`: A dictionary representation of all class attributes. Non-serializable objects are converted to string representations or serialized using their `to_dict()` method if available.

**Note:**
-   This method is used for telemetry logging and state persistence.
-   It recursively handles nested objects and provides fallback handling for objects that cannot be directly serialized.

## **Sequential Awareness Feature**

The `AgentRearrange` class now includes a **sequential awareness** feature that enhances agent collaboration in sequential workflows. When agents are executed sequentially, they automatically receive information about:

- **Agent ahead**: The agent that completed their task before them
- **Agent behind**: The agent that will receive their output next

This feature is automatically enabled when using sequential flows and provides agents with context about their position in the workflow, improving coordination and task understanding.

### How It Works

1. **Automatic Detection**: The system automatically detects when agents are running sequentially vs. in parallel
2. **Context Injection**: Before each sequential agent runs, awareness information is added to the conversation
3. **Enhanced Collaboration**: Agents can reference previous agents' work and prepare output for the next agent

### Example with Sequential Awareness

```python
from swarms import Agent, AgentRearrange

# Create agents
agent1 = Agent(agent_name="Researcher", system_prompt="Research the topic")
agent2 = Agent(agent_name="Writer", system_prompt="Write based on research")
agent3 = Agent(agent_name="Editor", system_prompt="Edit the written content")

# Create sequential workflow
workflow = AgentRearrange(
    agents=[agent1, agent2, agent3],
    flow="Researcher -> Writer -> Editor",
    team_awareness=True  # Enables sequential awareness
)

# Run the workflow
result = workflow.run("Research and write about artificial intelligence")
```

**What happens automatically:**
- **Researcher** runs first (no awareness info needed)
- **Writer** receives: "Sequential awareness: Agent ahead: Researcher | Agent behind: Editor"
- **Editor** receives: "Sequential awareness: Agent ahead: Writer"

## Documentation for `rearrange` Function
======================================

The rearrange function is a helper function that rearranges the given list of agents based on the specified flow.

## Parameters
----------

| Parameter | Type | Description |
| --- | --- | --- |
| `name` | `str` (optional) | Name for the agent rearrange system. Defaults to `None` (uses AgentRearrange default). |
| `description` | `str` (optional) | Description of the system. Defaults to `None` (uses AgentRearrange default). |
| `agents` | `List[Agent]` | The list of agents to be included in the system. |
| `flow` | `str` | The flow pattern defining agent execution order. Uses '->' for sequential and ',' for concurrent execution. |
| `task` | `str` (optional) | The task to be performed during rearrangement. Defaults to `None`. |
| `img` | `str` (optional) | Image input for agents that support it. Defaults to `None`. |
| `*args` | - | Additional positional arguments passed to `AgentRearrange` constructor. |
| `**kwargs` | - | Additional keyword arguments passed to `AgentRearrange` constructor. |

## Returns
-------

The result of running the agent system with the specified task.

### Example
-------

```python
from swarms import Agent, rearrange

# Create agents
agent1 = Agent(agent_name="researcher", system_prompt="Research topics")
agent2 = Agent(agent_name="writer", system_prompt="Write content")
agent3 = Agent(agent_name="reviewer", system_prompt="Review content")

# Execute task with flow
result = rearrange(
    agents=[agent1, agent2, agent3],
    flow="researcher -> writer, reviewer",
    task="Research and write a report"
)
```

### Example Usage
-------------

Here's an example of how to use the `AgentRearrange` class and the `rearrange` function with the new sequential awareness features:

```python
from swarms import Agent, AgentRearrange

# Create agents
agent1 = Agent(agent_name="researcher", system_prompt="Research the topic")
agent2 = Agent(agent_name="writer", system_prompt="Write based on research")
agent3 = Agent(agent_name="reviewer", system_prompt="Review the written content")

# Create sequential workflow
workflow = AgentRearrange(
    agents=[agent1, agent2, agent3],
    flow="researcher -> writer -> reviewer",
    team_awareness=True,  # Enables sequential awareness
    time_enabled=True,     # Enable timestamps
    message_id_on=True     # Enable message IDs
)

# Get sequential flow information
flow_structure = workflow.get_sequential_flow_structure()
print("Flow Structure:", flow_structure)

# Get awareness for specific agents
writer_awareness = workflow.get_agent_sequential_awareness("writer")
print("Writer Awareness:", writer_awareness)

# Run the workflow
output = workflow.run("Research and write about artificial intelligence")
print(output)

# Or use the callable interface
result = workflow("Research and write about machine learning")
```

In this example, we create three agents and define a sequential flow pattern. The sequential awareness features provide:
- **Automatic context**: Each agent knows who came before and who comes after
- **Better coordination**: Agents can reference previous work and prepare for next steps
- **Flow visualization**: You can see the complete workflow structure
- **Enhanced logging**: Better tracking of agent interactions

## Error Handling
--------------

The `AgentRearrange` class includes comprehensive error handling mechanisms. During initialization, the `reliability_check()` method validates critical parameters:

- **Agents validation**: Raises `ValueError` if agents list is None or empty
- **Max loops validation**: Raises `ValueError` if max_loops is 0
- **Flow validation**: Raises `ValueError` if flow is None or empty
- **Output type validation**: Raises `ValueError` if output_type is None or empty

The `validate_flow()` method checks the flow pattern format:
- Raises `ValueError` if the flow pattern doesn't include '->' to denote direction
- Raises `ValueError` if any agent in the flow is not registered

### Example:

```python
# Invalid flow pattern - missing agent
invalid_flow = "agent1 -> agent2 -> agent3"
agent_system = AgentRearrange(agents=[agent1, agent2], flow=invalid_flow)
output = agent_system.run("Some task")
```

This will raise a `ValueError` with the message `"Agent 'agent3' is not registered."`.

All errors are automatically logged and, if `autosave` is enabled, the current state is saved before error reporting.

## Parallel and Sequential Processing
----------------------------------

The `AgentRearrange` class supports both parallel and sequential processing of tasks based on the specified flow pattern. The flow syntax determines execution mode:

### Parallel Processing
When multiple agents are separated by commas, they execute concurrently:
```python
parallel_flow = "agent1, agent2 -> agent3"
```
In this example, `agent1` and `agent2` will be executed in parallel using `run_agents_concurrently()`, and their outputs will be collected and passed to `agent3`.

### Sequential Processing with Awareness
When agents are connected by arrows, they execute sequentially:
```python
sequential_flow = "agent1 -> agent2 -> agent3"
```
In this example, `agent1` runs first, then `agent2` receives awareness that `agent1` came before and `agent3` comes after, and finally `agent3` receives awareness that `agent2` came before.

### Combined Processing
You can combine both patterns:
```python
combined_flow = "agent1 -> agent2, agent3 -> agent4"
```
This executes `agent1` first, then `agent2` and `agent3` run concurrently, and finally `agent4` receives both outputs.

## Logging and Monitoring
-------

The `AgentRearrange` class includes comprehensive logging capabilities using the `loguru` library. Logs are stored in a dedicated "rearrange" folder. The sequential awareness features add enhanced logging:

- Flow validation messages
- Agent execution start/completion
- Sequential awareness information injection
- Concurrent execution coordination
- Error handling and recovery

All agent data is automatically logged via telemetry before and after execution (if `autosave=True`), providing comprehensive monitoring and debugging capabilities.

## Additional Parameters
---------------------

The `AgentRearrange` class accepts comprehensive parameters for enhanced functionality:

```python
agent_system = AgentRearrange(
    agents=agents, 
    flow=flow,
    max_loops=1,              # Maximum execution loops
    team_awareness=True,      # Enable sequential awareness
    time_enabled=True,        # Enable conversation timestamps
    message_id_on=True,       # Enable message IDs
    verbose=True,             # Enable detailed logging
    autosave=True,            # Automatically save execution data
    output_type="all",        # Output format: "all", "final", "list", or "dict"
    memory_system=None,       # Optional memory system for persistence
    human_in_the_loop=False,  # Enable human interaction points
    rules=None                # Custom system rules and constraints
)
```

## Customization
-------------

The `AgentRearrange` class and the `rearrange` function can be customized and extended to suit specific use cases. The new sequential awareness features provide a foundation for building more sophisticated agent coordination systems.

## Internal Methods

The following methods are used internally but may be useful to understand the system architecture:

- `_run(task, img, custom_tasks, *args, **kwargs)`: Core execution method that orchestrates the workflow. The `custom_tasks` parameter (Dict[str, str]) allows overriding the main task for specific agents in the flow, enabling per-agent task customization.
- `_run_sequential_workflow()`: Handles sequential agent execution with awareness
- `_run_concurrent_workflow()`: Handles parallel agent execution using `run_agents_concurrently()`
- `_get_sequential_awareness()`: Generates awareness information for agents based on their position in the flow
- `_get_sequential_flow_info()`: Generates overall flow structure information showing the complete workflow
- `_catch_error()`: Comprehensive error handling with logging and state saving. Automatically called when exceptions occur.
- `_serialize_callable()`: Helper method for serializing callable attributes in `to_dict()`
- `_serialize_attr()`: Helper method for serializing individual attributes, handling non-serializable objects

## Limitations
-----------

It's important to note that the `AgentRearrange` class and the `rearrange` function rely on the individual agents to process tasks correctly. The quality of the output will depend on the capabilities and configurations of the agents used in the swarm. 

The sequential awareness feature works best with agents that can understand and utilize context about their position in the workflow.

Flow patterns must include at least one '->' to denote direction. Agents referenced in the flow must be registered in the agents dictionary.

## Conclusion
----------

The `AgentRearrange` class and the `rearrange` function provide a flexible and extensible framework for orchestrating swarms of agents to process tasks based on a specified flow pattern. The new **sequential awareness** features significantly enhance agent collaboration by providing context about workflow relationships.

By combining the capabilities of individual agents with enhanced awareness of their position in the workflow, you can create more intelligent and coordinated multi-agent systems that understand not just their individual tasks, but also their role in the larger workflow.

Whether you're working on natural language processing tasks, data analysis, or any other domain where agent-based systems can be beneficial, the enhanced `AgentRearrange` class provides a solid foundation for building sophisticated swarm-based solutions with improved coordination and context awareness.

