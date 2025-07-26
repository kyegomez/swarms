# How to Create Good Agents

This guide will walk you through the steps to build high-quality agents by extending the `Agent` class. It emphasizes best practices, the use of type annotations, comprehensive documentation, and modular design to ensure maintainability and scalability. Additionally, you will learn how to incorporate a callable `llm` parameter or specify a `model_name` attribute to enhance flexibility and functionality. These principles ensure that agents are not only functional but also robust and adaptable to future requirements.

## Overview

A good agent is a modular and reusable component designed to perform specific tasks efficiently. By inheriting from the base `Agent` class, developers can extend its functionality while adhering to standardized principles. Each custom agent should:

- Inherit from the `Agent` class to maintain compatibility with swarms.
- Define a `run(task: str, img: str)` method to execute tasks effectively.
- Include descriptive attributes such as `name`, `system_prompt`, and `description` to enhance clarity.
- Optionally, include an `llm` parameter (callable) or a `model_name` to enable seamless integration with language models.
- Emphasize modularity, allowing the agent to be reused across various contexts and tasks.

By following these guidelines, you can create agents that integrate well with broader systems and exhibit high reliability in real-world applications.

---

## Creating a Custom Agent

Here is a detailed template for creating a custom agent by inheriting the `Agent` class. This template demonstrates how to structure an agent with extendable and reusable features:

```python
from typing import Callable, Any
from swarms import Agent

class MyNewAgent(Agent):
    """
    A custom agent class for specialized tasks.

    Attributes:
        name (str): The name of the agent.
        system_prompt (str): The prompt guiding the agent's behavior.
        description (str): A brief description of the agent's purpose.
        llm (Callable, optional): A callable representing the language model to use.
    """

    def __init__(self, name: str, system_prompt: str, model_name: str = None, description: str, llm: Callable = None):
        """
        Initialize the custom agent.

        Args:
            name (str): The name of the agent.
            system_prompt (str): The prompt guiding the agent.
            model_name (str): The name of your model can use litellm [openai/gpt-4o]
            description (str): A description of the agent's purpose.
            llm (Callable, optional): A callable representing the language model to use.
        """
        super().__init__(agent_name=name, system_prompt=system_prompt, model_name=model_name)
        self.agent_name = agent_name
        self.system_prompt system_prompt
        self.description = description
        self.model_name = model_name

    def run(self, task: str, img: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the task assigned to the agent.

        Args:
            task (str): The task description.
            img (str): The image input for processing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of the task execution.
        """
        # Your custom logic 
        ...
```

This design ensures a seamless extension of functionality while maintaining clear and maintainable code.

---

## Key Considerations

### 1. **Type Annotations**
Always use type hints for method parameters and return values. This improves code readability, supports static analysis tools, and reduces bugs, ensuring long-term reliability.

### 2. **Comprehensive Documentation**
Provide detailed docstrings for all classes, methods, and attributes. Clear documentation ensures that your agent's functionality is understandable to both current and future collaborators.

### 3. **Modular Design**
Keep the agent logic modular and reusable. Modularity simplifies debugging, testing, and extending functionalities, making the code more adaptable to diverse scenarios.

### 4. **Flexible Model Integration**
Use either an `llm` callable or `model_name` attribute for integrating language models. This flexibility ensures your agent can adapt to various tasks, environments, and system requirements.

### 5. **Error Handling**
Incorporate robust error handling to manage unexpected inputs or issues during execution. This not only ensures reliability but also builds user trust in your system.

### 6. **Scalability Considerations**
Ensure your agent design can scale to accommodate increased complexity or a larger number of tasks without compromising performance.

---

## Example Usage

Here is an example of how to use your custom agent effectively:

```python
# Example LLM callable
class MockLLM:
    """
    A mock language model class for simulating LLM behavior.

    Methods:
        run(task: str, img: str, *args: Any, **kwargs: Any) -> str:
            Processes the task and image input to return a simulated response.
    """

    def run(self, task: str, img: str, *args: Any, **kwargs: Any) -> str:
        return f"Processed task '{task}' with image '{img}'"

# Create an instance of MyNewAgent
agent = MyNewAgent(
    name="ImageProcessor",
    system_prompt="Process images and extract relevant details.",
    description="An agent specialized in processing images and extracting insights.",
    llm=MockLLM().run
)

# Run a task
result = agent.run(task="Analyze content", img="path/to/image.jpg")
print(result)
```

This example showcases the practical application of the `MyNewAgent` class and highlights its extensibility.


## Production-Grade Example with **Griptape Agent Integration Example**

In this example, we will create a **Griptape** agent by inheriting from the Swarms `Agent` class and implementing the `run` method.

### **Griptape Integration Steps**:

1. **Inherit from Swarms Agent**: Inherit from the `SwarmsAgent` class.
2. **Create Griptape Agent**: Initialize the **Griptape** agent inside your class and provide it with the necessary tools.
3. **Override the `run()` method**: Implement logic to process a task string and execute the Griptape agent.

## **Griptape Example Code**:

```python
from swarms import (
    Agent as SwarmsAgent,
)  # Import the base Agent class from Swarms
from griptape.structures import Agent as GriptapeAgent
from griptape.tools import (
    WebScraperTool,
    FileManagerTool,
    PromptSummaryTool,
)

# Create a custom agent class that inherits from SwarmsAgent
class GriptapeSwarmsAgent(SwarmsAgent):
    def __init__(self, name: str, system_prompt: str: str, *args, **kwargs):
        super().__init__(agent_name=name, system_prompt=system_prompt)
        # Initialize the Griptape agent with its tools
        self.agent = GriptapeAgent(
            input="Load {{ args[0] }}, summarize it, and store it in a file called {{ args[1] }}.",
            tools=[
                WebScraperTool(off_prompt=True),
                PromptSummaryTool(off_prompt=True),
                FileManagerTool(),
            ],
            *args,
            **kwargs,
        )

    # Override the run method to take a task and execute it using the Griptape agent
    def run(self, task: str) -> str:
        # Extract URL and filename from task
        url, filename = task.split(",")  # Example task string: "https://example.com, output.txt"
        # Execute the Griptape agent
        result = self.agent.run(url.strip(), filename.strip())
        # Return the final result as a string
        return str(result)


# Example usage:
griptape_swarms_agent = GriptapeSwarmsAgent()
output = griptape_swarms_agent.run("https://griptape.ai, griptape.txt")
print(output)
```


---

## Best Practices

1. **Test Extensively:**
   Validate your agent with various task inputs to ensure it performs as expected under different conditions.

2. **Follow the Single Responsibility Principle:**
   Design each agent to focus on a specific task or role, ensuring clarity and modularity in implementation.

3. **Log Actions:**
   Include detailed logging within the `run` method to capture key actions, inputs, and results for debugging and monitoring.

4. **Use Open-Source Contributions:**
   Contribute your custom agents to the Swarms repository at [https://github.com/kyegomez/swarms](https://github.com/kyegomez/swarms). Sharing your innovations helps advance the ecosystem and encourages collaboration.

5. **Iterate and Refactor:**
   Continuously improve your agents based on feedback, performance evaluations, and new requirements to maintain relevance and functionality.

---

## Conclusion

By following these guidelines, you can create powerful and flexible agents tailored to specific tasks. Leveraging inheritance from the `Agent` class ensures compatibility and standardization across swarms. Emphasize modularity, thorough testing, and clear documentation to build agents that are robust, scalable, and easy to integrate. Collaborate with the community by submitting your innovative agents to the Swarms repository, contributing to a growing ecosystem of intelligent solutions. With a well-designed agent, you are equipped to tackle diverse challenges efficiently and effectively.

