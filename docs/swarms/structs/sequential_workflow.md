# SequentialWorkflow Documentation

The `SequentialWorkflow` class is designed to manage and execute a sequence of tasks through a dynamic arrangement of agents. This class allows for the orchestration of multiple agents in a predefined order, facilitating complex workflows where tasks are processed sequentially by different agents.

## Attributes

| Attribute        | Type          | Description                                      |
|------------------|---------------|--------------------------------------------------|
| `agents`         | `List[Agent]` | The list of agents in the workflow.              |
| `flow`           | `str`         | A string representing the order of agents.       |
| `agent_rearrange`| `AgentRearrange` | Manages the dynamic execution of agents.        |

## Methods

### `__init__(self, agents: List[Agent] = None, max_loops: int = 1, *args, **kwargs)`

The constructor initializes the `SequentialWorkflow` object.

- **Parameters:**
  - `agents` (`List[Agent]`, optional): The list of agents in the workflow. Defaults to `None`.
  - `max_loops` (`int`, optional): The maximum number of loops to execute the workflow. Defaults to `1`.
  - `*args`: Variable length argument list.
  - `**kwargs`: Arbitrary keyword arguments.

### `run(self, task: str) -> str`

Runs the specified task through the agents in the dynamically constructed flow.

- **Parameters:**
  - `task` (`str`): The task for the agents to execute.

- **Returns:**
  - `str`: The final result after processing through all agents.

- **Usage Example:**
  ```python
  from swarms import Agent, SequentialWorkflow, Anthropic


    # Initialize the language model agent (e.g., GPT-3)
    llm = Anthropic()

    # Place your key in .env

    # Initialize agents for individual tasks
    agent1 = Agent(
        agent_name="Blog generator",
        system_prompt="Generate a blog post like stephen king",
        llm=llm,
        max_loops=1,
        dashboard=False,
        tools=[],
    )
    agent2 = Agent(
        agent_name="summarizer",
        system_prompt="Sumamrize the blog post",
        llm=llm,
        max_loops=1,
        dashboard=False,
        tools=[],
    )

    # Create the Sequential workflow
    workflow = SequentialWorkflow(
        agents=[agent1, agent2], max_loops=1, verbose=False
    )

    # Run the workflow
    workflow.run(
        "Generate a blog post on how swarms of agents can help businesses grow."
    )

  ```

  This example initializes a `SequentialWorkflow` with three agents and executes a task, printing the final result.

- **Notes:**
  - Logs the task execution process and handles any exceptions that occur during the task execution.

### Logging and Error Handling

The `run` method includes logging to track the execution flow and captures errors to provide detailed information in case of failures. This is crucial for debugging and ensuring smooth operation of the workflow.

## Additional Tips

- Ensure that the agents provided to the `SequentialWorkflow` are properly initialized and configured to handle the tasks they will receive.

- The `max_loops` parameter can be used to control how many times the workflow should be executed, which is useful for iterative processes.
- Utilize the logging information to monitor and debug the task execution process.
