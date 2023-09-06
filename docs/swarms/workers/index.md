# Module Name: Worker

The `Worker` class encapsulates the idea of a semi-autonomous agent that utilizes a large language model to execute tasks. The module provides a unified interface for AI-driven task execution while combining a series of tools and utilities. It sets up memory storage and retrieval mechanisms for contextual recall and offers an option for human involvement, making it a versatile and adaptive agent for diverse applications.

## **Class Definition**:

```python
class Worker:
```

### **Parameters**:

- `model_name` (str, default: "gpt-4"): Name of the language model.
- `openai_api_key` (str, Optional): API key for accessing OpenAI's models.
- `ai_name` (str, default: "Autobot Swarm Worker"): Name of the AI agent.
- `ai_role` (str, default: "Worker in a swarm"): Role description of the AI agent.
- `external_tools` (list, Optional): A list of external tool objects to be used.
- `human_in_the_loop` (bool, default: False): If set to `True`, it indicates that human intervention may be required.
- `temperature` (float, default: 0.5): Sampling temperature for the language model's output. Higher values make the output more random, and lower values make it more deterministic.

### **Methods**:

#### `__init__`:

Initializes the Worker class.

#### `setup_tools`:

Sets up the tools available to the worker. Default tools include reading and writing files, processing CSV data, querying websites, and taking human input. Additional tools can be appended through the `external_tools` parameter.

#### `setup_memory`:

Initializes memory systems using embeddings and a vector store for the worker.

#### `setup_agent`:

Sets up the primary agent using the initialized tools, memory, and language model.

#### `run`:

Executes a given task using the agent.

#### `__call__`:

Makes the Worker class callable. When an instance of the class is called, it will execute the provided task using the agent.

## **Usage Examples**:

### **Example 1**: Basic usage with default parameters:

```python
from swarms import Worker

worker = Worker(model_name="gpt-4", openai_api_key="YOUR_API_KEY")
result = worker.run("Summarize the document.")
```

### **Example 2**: Usage with custom tools:

```python
from swarms import Worker, MyCustomTool

worker = Worker(model_name="gpt-4", openai_api_key="YOUR_API_KEY", external_tools=[MyCustomTool()])
result = worker.run("Perform a custom operation on the document.")
```

### **Example 3**: Usage with human in the loop:

```python
from swarms import Worker

worker = Worker(model_name="gpt-4", openai_api_key="YOUR_API_KEY", human_in_the_loop=True)
result = worker.run("Translate this complex document, and ask for help if needed.")
```

## **Mathematical Description**:

Conceptually, the `Worker` class can be seen as a function:

\[ W(t, M, K, T, H, \theta) \rightarrow R \]

Where:

- \( W \) = Worker function
- \( t \) = task to be performed
- \( M \) = Model (e.g., "gpt-4")
- \( K \) = OpenAI API key
- \( T \) = Set of Tools available
- \( H \) = Human involvement flag (True/False)
- \( \theta \) = Temperature parameter
- \( R \) = Result of the task

This mathematical abstraction provides a simple view of the `Worker` class's capability to transform a task input into a desired output using a combination of AI and toolsets.

## **Notes**:

The Worker class acts as a bridge between raw tasks and the tools & AI required to accomplish them. The setup ensures flexibility and versatility. The decorators used in the methods (e.g., log_decorator, error_decorator) emphasize the importance of logging, error handling, and performance measurement, essential for real-world applications.