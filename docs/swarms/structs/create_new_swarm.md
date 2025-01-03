# How to Add a New Swarm Class

This guide provides comprehensive step-by-step instructions for developers to create and add a new swarm. It emphasizes the importance of adhering to best practices, using proper type hints, and documenting code thoroughly to ensure maintainability, scalability, and clarity in your implementations.

## Overview

A Swarm class enables developers to manage and coordinate multiple agents working together to accomplish complex tasks efficiently. Each Swarm must:

- Contain a `run(task: str, img: str, *args, **kwargs)` method, which serves as the primary execution method for tasks.
- Include `name`, `description`, and `agents` parameters.
- Ensure `agents` is a callable function that adheres to specific requirements for dynamic agent behavior.
- Follow type-hinting and documentation best practices to maintain code clarity and reliability.

Each Agent within the swarm must:

- Contain `agent_name`, `system_prompt`, and a `run` method.
- Follow similar type hinting and documentation standards to ensure consistency and readability.

By adhering to these requirements, you can create robust, reusable, and modular swarms that streamline task management and enhance collaborative functionality.

---

## Creating a Swarm Class

Below is a detailed template for creating a Swarm class. Ensure that all elements are documented and clearly defined:

```python
from typing import Callable, Any

class MySwarm:
    """
    A custom swarm class to manage and execute tasks with multiple agents.

    Attributes:
        name (str): The name of the swarm.
        description (str): A brief description of the swarm's purpose.
        agents (Callable): A callable that returns the list of agents to be utilized.
    """

    def __init__(self, name: str, description: str, agents: Callable):
        """
        Initialize the Swarm with its name, description, and agents.

        Args:
            name (str): The name of the swarm.
            description (str): A description of the swarm.
            agents (Callable): A callable that provides the agents for the swarm.
        """
        self.name = name
        self.description = description
        self.agents = agents

    def run(self, task: str, img: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a task using the swarm and its agents.

        Args:
            task (str): The task description.
            img (str): The image input.
            *args: Additional positional arguments for customization.
            **kwargs: Additional keyword arguments for fine-tuning behavior.

        Returns:
            Any: The result of the task execution, aggregated from all agents.
        """
        results = []
        for agent in self.agents():
            result = agent.run(task, img, *args, **kwargs)
            results.append(result)
        return results
```

This Swarm class serves as the main orchestrator for coordinating agents and running tasks dynamically and flexibly.

---

## Creating an Agent Class

Each agent must follow a well-defined structure to ensure compatibility with the swarm. Below is an example of an agent class:

```python
class Agent:
    """
    A single agent class to handle specific tasks assigned by the swarm.

    Attributes:
        agent_name (str): The name of the agent.
        system_prompt (str): The system prompt guiding the agent's behavior and purpose.
    """

    def __init__(self, agent_name: str, system_prompt: str):
        """
        Initialize the agent with its name and system prompt.

        Args:
            agent_name (str): The name of the agent.
            system_prompt (str): The guiding prompt for the agent.
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt

    def run(self, task: str, img: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a specific task assigned to the agent.

        Args:
            task (str): The task description.
            img (str): The image input for processing.
            *args: Additional positional arguments for task details.
            **kwargs: Additional keyword arguments for extended functionality.

        Returns:
            Any: The result of the task execution, which can be customized.
        """
        # Example implementation (to be customized by developer)
        return f"Agent {self.agent_name} executed task: {task}"
```

This structure ensures that each agent can independently handle tasks and integrate seamlessly into a swarm.

---

## Adding Your Swarm to a Project

### Step 1: Define Your Agents
Create one or more instances of the `Agent` class to serve as components of your swarm. For example:

```python
def create_agents():
    return [
        Agent(agent_name="Agent1", system_prompt="Analyze the image and summarize results."),
        Agent(agent_name="Agent2", system_prompt="Detect objects and highlight key features."),
    ]
```

### Step 2: Implement Your Swarm
Create an instance of your Swarm class, defining its name, description, and associated agents:

```python
my_swarm = MySwarm(
    name="Image Analysis Swarm",
    description="A swarm designed to analyze images and perform a range of related tasks.",
    agents=create_agents
)
```

### Step 3: Execute Tasks
Call the `run` method of your swarm, passing in the required parameters for execution:

```python
results = my_swarm.run(task="Analyze image content", img="path/to/image.jpg")
print(results)
```

This simple flow allows you to dynamically utilize agents for diverse operations and ensures efficient task execution.

---

## Best Practices

To ensure your swarm implementation is efficient and maintainable, follow these best practices:

1. **Type Annotations:**
   Use precise type hints for parameters and return types to improve code readability and support static analysis tools.

2. **Comprehensive Documentation:**
   Include clear and detailed docstrings for all classes, methods, and attributes to ensure your code is understandable.

3. **Thorough Testing:**
   Test your swarm and agents with various tasks to verify correctness and identify potential edge cases.

4. **Modular Design:**
   Keep your swarm and agent logic modular, enabling reuse and easy extensions for future enhancements.

5. **Error Handling:**
   Implement robust error handling in the `run` methods to gracefully manage unexpected inputs or issues during execution.

6. **Code Review:**
   Regularly review and refactor your code to align with the latest best practices and maintain high quality.

7. **Scalability:**
   Design your swarm with scalability in mind, ensuring it can handle a large number of agents and complex tasks.

8. **Logging and Monitoring:**
   Include comprehensive logging to track task execution and monitor performance, enabling easier debugging and optimization.

---

## Example Output

Given the implementation above, executing a task might produce output such as:

```plaintext
[
    "Agent Agent1 executed task: Analyze image content",
    "Agent Agent2 executed task: Analyze image content"
]
```

The modular design ensures that each agent contributes to the overall functionality of the swarm, allowing seamless scalability and dynamic task management.

---

## Conclusion

By following these guidelines, you can create swarms that are powerful, flexible, and maintainable. Leveraging the provided templates and best practices enables you to build efficient multi-agent systems capable of handling diverse and complex tasks. Proper structuring, thorough testing, and adherence to best practices will ensure your swarm integrates effectively into any project, delivering robust and reliable performance. Furthermore, maintaining clear documentation and emphasizing modularity will help your implementation adapt to future needs and use cases. Empower your projects with a well-designed swarm architecture today.

