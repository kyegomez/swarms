
# `TaskQueueBase`

## Introduction
The `swarms.structs` library is a key component of a multi-agent system's task management infrastructure. It provides the necessary classes and methods to create and manage queues of tasks that can be distributed among a swarm of agents. The purpose of this documentation is to guide users through the proper use of the `TaskQueueBase` class, which serves as an abstract base class for implementing task queues.

## TaskQueueBase Class

```python
import threading
from abc import ABC, abstractmethod

# Include any additional imports that are relevant to decorators and other classes such as Task and Agent if needed

# Definition of the synchronized_queue decorator (if necessary)


class TaskQueueBase(ABC):
    def __init__(self):
        self.lock = threading.Lock()

    @synchronized_queue
    @abstractmethod
    def add_task(self, task: Task) -> bool:
        pass

    @synchronized_queue
    @abstractmethod
    def get_task(self, agent: Agent) -> Task:
        pass

    @synchronized_queue
    @abstractmethod
    def complete_task(self, task_id: str):
        pass

    @synchronized_queue
    @abstractmethod
    def reset_task(self, task_id: str):
        pass
```

### Architecture and Purpose
The `TaskQueueBase` class provides an abstract interface for task queue implementations. This class uses the `threading.Lock` to ensure mutual exclusion, making it suitable for concurrent environments. The `@synchronized_queue` decorator implies that each method should be synchronized to prevent race conditions.

Tasks are generally represented by the `Task` class, and agents by the `Agent` class. Implementations of the `TaskQueueBase` will provide the logic to store tasks, distribute them to agents, and manage their lifecycles.

#### Methods and Their Arguments

Here's an overview of each method and its arguments:

| Method         | Arguments      | Return Type | Description                                                                                   |
|----------------|----------------|-------------|-----------------------------------------------------------------------------------------------|
| add_task       | task (Task)    | bool        | Adds a task to the queue and returns True if successfully added, False otherwise.             |
| get_task       | agent (Agent)  | Task        | Retrieves the next task for the given agent.                                                  |
| complete_task  | task_id (str)  | None        | Marks the task identified by task_id as completed.                                            |
| reset_task     | task_id (str)  | None        | Resets the task identified by task_id, typically done if an agent fails to complete the task. |

### Example Usage

Below are three examples of how the `TaskQueueBase` class can be implemented and used.

**Note:** The actual code for decorators, Task, Agent, and concrete implementations of `TaskQueueBase` is not provided and should be created as per specific requirements.

#### Example 1: Basic Implementation

```python
# file: basic_queue.py

# Assume synchronized_queue decorator is defined elsewhere
from decorators import synchronized_queue

from swarms.structs import Agent, Task, TaskQueueBase


class BasicTaskQueue(TaskQueueBase):
    def __init__(self):
        super().__init__()
        self.tasks = []

    @synchronized_queue
    def add_task(self, task: Task) -> bool:
        self.tasks.append(task)
        return True

    @synchronized_queue
    def get_task(self, agent: Agent) -> Task:
        return self.tasks.pop(0)

    @synchronized_queue
    def complete_task(self, task_id: str):
        # Logic to mark task as completed
        pass

    @synchronized_queue
    def reset_task(self, task_id: str):
        # Logic to reset the task
        pass


# Usage
queue = BasicTaskQueue()
# Add task, assuming Task object is created
queue.add_task(someTask)
# Get task for an agent, assuming Agent object is created
task = queue.get_task(someAgent)
```

#### Example 2: Priority Queue Implementation

```python
# file: priority_queue.py
# Similar to example 1, but tasks are managed based on priority within add_task and get_task methods
```

#### Example 3: Persistent Queue Implementation

```python
# file: persistent_queue.py
# An example demonstrating tasks being saved to a database or filesystem. Methods would include logic for persistence.
```

### Additional Information and Common Issues

This section would provide insights on thread safety, error handling, and best practices in working with task queues in a multi-agent system.

### References

Links to further resources and any academic papers or external documentation related to task queues and multi-agent systems would be included here.

