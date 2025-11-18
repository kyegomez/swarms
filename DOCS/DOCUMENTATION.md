# Swarms Documentation

## Overview
The Swarm module includes the implementation of two classes, `WorkerNode` and `BossNode`, which respectively represent a worker agent and a boss agent. A worker agent is responsible for completing given tasks, while a boss agent is responsible for creating and managing tasks for the worker agent(s).

## Key Classes

### WorkerNode
```python
class WorkerNode:
```

The WorkerNode class represents an autonomous worker agent that can perform a range of tasks.

__Methods__:

- `create_agent(ai_name: str, ai_role: str, human_in_the_loop: bool, search_kwargs: dict) -> None`:

    This method creates a new autonomous agent that can complete tasks. The agent utilizes several tools such as search engines, a file writer/reader, and a multi-modal visual tool. 
    The agent's configuration is customizable through the method parameters. 

    ```python
    # Example usage
    worker_node = WorkerNode(llm, tools, vectorstore)
    worker_node.create_agent('test_agent', 'test_role', False, {})
    ```

- `run_agent(prompt: str) -> None`:

    This method runs the agent on a given task, defined by the `prompt` parameter.

    ```python
    # Example usage
    worker_node = WorkerNode(llm, tools, vectorstore)
    worker_node.create_agent('test_agent', 'test_role', False, {})
    worker_node.run_agent('Calculate the square root of 144.')
    ```

### BossNode
```python
class BossNode:
```

The BossNode class represents a manager agent that can create tasks and control the execution of these tasks.

__Methods__:

- `create_task(objective: str) -> dict`:

    This method creates a new task based on the provided `objective`. The created task is a dictionary with the objective as its value.

    ```python
    # Example usage
    boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
    task = boss_node.create_task('Find the square root of 144.')
    ```

- `execute_task(task: dict) -> None`:

    This method triggers the execution of a given task.

    ```python
    # Example usage
    boss_node = BossNode(llm, vectorstore, task_execution_chain, False, 3)
    task = boss_node.create_task('Find the square root of 144.')
    boss_node.execute_task(task)
    ```

### Note

Before creating the WorkerNode and BossNode, make sure to initialize the lower level model (llm), tools, and vectorstore which are used as parameters in the constructors of the two classes.

In addition, the WorkerNode class uses the MultiModalVisualAgentTool which is a custom tool that enables the worker agent to run multi-modal visual tasks. Ensure that this tool is correctly initialized before running the WorkerNode.

This documentation provides an overview of the main functionalities of the Swarm module. For additional details and advanced functionalities, please review the source code and the accompanying comments.
