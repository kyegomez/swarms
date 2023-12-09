# Swarms Documentation

## ClassName

Swarms

## Purpose

The Swarms module provides a powerful framework for creating and managing swarms of autonomous agents to accomplish complex tasks. It consists of the `WorkerNode` and `BossNode` classes, along with the `LLM` utility class, which allow you to easily set up and run a swarm of agents to tackle any objective. The module is highly configurable and extensible, providing flexibility to accommodate various use cases.

## Usage example

```python
from swarms import Swarms

api_key = "your_openai_api_key"

# Initialize Swarms with your API key
swarm = Swarms(api_key=api_key)

# Define an objective
objective = "Please make a web GUI for using HTTP API server..."

# Run Swarms
result = swarm.run(objective)

print(result)
```

## Constructor

```python
def __init__(self, openai_api_key)
```

- `openai_api_key` (required): The API key for OpenAI's models.

## Methods

### run(objective)

Runs the swarm with the given objective by initializing the worker and boss nodes.

- `objective` (required): The objective or task to be accomplished by the swarm.

Returns the result of the swarm execution.

## Example Usage

```python
from swarms import Swarms

api_key = "your_openai_api_key"

# Initialize Swarms with your API key
swarm = Swarms(api_key=api_key)

# Define an objective
objective = "Please make a web GUI for using HTTP API server..."

# Run Swarms
result = swarm.run(objective)

print(result)
```

## WorkerNode

The `WorkerNode` class represents an autonomous agent instance that functions as a worker to accomplish complex tasks. It has the ability to search the internet, process and generate images, text, audio, and more.

### Constructor

```python
def __init__(self, llm, tools, vectorstore)
```

- `llm` (required): The language model used by the worker node.
- `tools` (required): A list of tools available to the worker node.
- `vectorstore` (required): The vector store used by the worker node.

### Methods

- `create_agent(ai_name, ai_role, human_in_the_loop, search_kwargs)`: Creates an agent within the worker node.
- `add_tool(tool)`: Adds a tool to the worker node.
- `run(prompt)`: Runs the worker node to complete a task specified by the prompt.

### Example Usage

```python
from swarms import worker_node

# Your OpenAI API key
api_key = "your_openai_api_key"

# Initialize a WorkerNode with your API key
node = worker_node(api_key)

# Define an objective
objective = "Please make a web GUI for using HTTP API server..."

# Run the task
task = node.run(objective)

print(task)
```

## BossNode

The `BossNode` class represents an agent responsible for creating and managing tasks for the worker agent(s). It interacts with the worker node(s) to delegate tasks and monitor their progress.

### Constructor

```python
def __init__(self, llm, vectorstore, agent_executor, max_iterations)
```

- `llm` (required): The language model used by the boss node.
- `vectorstore` (required): The vector store used by the boss node.
- `agent_executor` (required): The agent executor used to execute tasks.
- `max_iterations` (required): The maximum number of iterations for task execution.

### Methods

- `create_task(objective)`: Creates a task with the given objective.
- `execute_task(task)`: Executes the given task by interacting with the worker agent(s).

## LLM

The `LLM` class is a utility class that provides an interface to different language models (LLMs) such as OpenAI's ChatGPT and Hugging Face models. It is used to initialize the language model for the worker and boss nodes.

### Constructor

```python
def __init__(self, openai_api_key=None, hf_repo_id=None, hf_api_token=None, model_kwargs=None)
```

- `openai_api_key` (optional): The API key for OpenAI's models.
- `hf_repo_id` (optional): The repository ID for the Hugging Face model.
- `hf_api_token` (optional): The API token for the Hugging Face model.
- `model_kwargs` (optional): Additional keyword arguments to pass to the language model.

### Methods

- `run(prompt)`: Runs the language model with the given prompt and returns the generated response.

## Configuration

The Swarms module can be configured by modifying the following parameters:

### WorkerNode

- `llm_class`: The language model class to use for the worker node (default: `ChatOpenAI`).
- `temperature`: The temperature parameter for the language model (default: `0.5`).

### BossNode

- `llm_class`: The language model class to use for the boss node (default: `OpenAI`).
- `max_iterations`: The maximum number of iterations for task execution (default: `5`).

### LLM

- `openai_api_key`: The API key for OpenAI's models.
- `hf_repo_id`: The repository ID for the Hugging Face model.
- `hf_api_token`: The API token for the Hugging Face model.
- `model_kwargs`: Additional keyword arguments to pass to the language model.

## Tool Configuration

The Swarms module supports various tools that can be added to the worker node for performing specific tasks. The following tools are available:

- `DuckDuckGoSearchRun`: A tool for performing web searches.
- `WriteFileTool`: A tool for writing files.
- `ReadFileTool`: A tool for reading files.
- `process_csv`: A tool for processing CSV files.
- `WebpageQATool`: A tool for performing question answering using web pages.

Additional tools can be added by extending the functionality of the `Tool` class.

## Advanced Usage

For more advanced usage, you can customize the tools and parameters according to your specific requirements. The Swarms module provides flexibility and extensibility to accommodate various use cases.

For example, you can add your own custom tools by extending the `Tool` class and adding them to the worker node. You can also modify the prompt templates used by the boss node to customize the interaction between the boss and worker agents.

Please refer to the source code and documentation of the Swarms module for more details and examples.

## Conclusion

The Swarms module provides a powerful framework for creating and managing swarms of autonomous agents to accomplish complex tasks. With the `WorkerNode` and `BossNode` classes, along with the `LLM` utility class, you can easily set up and run a swarm of agents to tackle any objective. The module is highly configurable and extensible, allowing you to tailor it to your specific needs.


## LLM
### Purpose
The `LLM` class provides an interface to different language models (LLMs) such as OpenAI's ChatGPT and Hugging Face models. It allows you to initialize and run a language model with a given prompt and obtain the generated response.

### Systems Understanding
The `LLM` class takes an OpenAI API key or Hugging Face repository ID and API token as input. It uses these credentials to initialize the language model, either from OpenAI's models or from a specific Hugging Face repository. The language model can then be run with a prompt, and the generated response is returned.

### Usage Example
```python
from swarms import LLM

# Create an instance of LLM with OpenAI API key
llm_instance = LLM(openai_api_key="your_openai_key")

# Run the language model with a prompt
result = llm_instance.run("Who won the FIFA World Cup in 1998?")
print(result)

# Create an instance of LLM with Hugging Face repository ID and API token
llm_instance = LLM(hf_repo_id="google/flan-t5-xl", hf_api_token="your_hf_api_token")

# Run the language model with a prompt
result = llm_instance.run("Who won the FIFA World Cup in 1998?")
print(result)
```

### Constructor
```python
def __init__(self, openai_api_key: Optional[str] = None,
             hf_repo_id: Optional[str] = None,
             hf_api_token: Optional[str] = None,
             model_kwargs: Optional[dict] = None)
```
- `openai_api_key` (optional): The API key for OpenAI's models.
- `hf_repo_id` (optional): The repository ID for the Hugging Face model.
- `hf_api_token` (optional): The API token for the Hugging Face model.
- `model_kwargs` (optional): Additional keyword arguments to pass to the language model.

### Methods
- `run(prompt: str) -> str`: Runs the language model with the given prompt and returns the generated response.

### Args
- `prompt` (str): The prompt to be passed to the language model.

### Returns
- `result` (str): The generated response from the language model.

## Conclusion
The `LLM` class provides a convenient way to initialize and run different language models using either OpenAI's API or Hugging Face models. By providing the necessary credentials and a prompt, you can obtain the generated response from the language model.






# `GooglePalm` class:

### Example 1: Using Dictionaries as Messages

```python
from google_palm import GooglePalm

# Initialize the GooglePalm instance
gp = GooglePalm(
    client=your_client,
    model_name="models/chat-bison-001",
    temperature=0.7,
    top_p=0.9,
    top_k=10,
    n=5
)

# Create some messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
]

# Generate a response
response = gp.generate(messages)

# Print the generated response
print(response)
```

### Example 2: Using BaseMessage and Its Subclasses as Messages

```python
from google_palm import GooglePalm
from langchain.schema.messages import SystemMessage, HumanMessage

# Initialize the GooglePalm instance
gp = GooglePalm(
    client=your_client,
    model_name="models/chat-bison-001",
    temperature=0.7,
    top_p=0.9,
    top_k=10,
    n=5
)

# Create some messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who won the world series in 2020?"),
]

# Generate a response
response = gp.generate(messages)

# Print the generated response
print(response)
```

### Example 3: Using GooglePalm with Asynchronous Function

```python
import asyncio
from google_palm import GooglePalm
from langchain.schema.messages import SystemMessage, HumanMessage

# Initialize the GooglePalm instance
gp = GooglePalm(
    client=your_client,
    model_name="models/chat-bison-001",
    temperature=0.7,
    top_p=0.9,
    top_k=10,
    n=5
)

# Create some messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who won the world series in 2020?"),
]

# Define an asynchronous function
async def generate_response():
    response = await gp._agenerate(messages)
    print(response)

# Run the asynchronous function
asyncio.run(generate_response())
```

Remember to replace `your_client` with an actual instance of your client. Also, ensure the `model_name` is the correct name of the model that you want to use.

The `temperature`, `top_p`, `top_k`, and `n` parameters control the randomness and diversity of the generated responses. You can adjust these parameters based on your application's requirements.





## `CodeInterpreter`:

```python
tool = CodeInterpreter("Code Interpreter", "A tool to interpret code and generate useful outputs.")
tool.run("Plot the bitcoin chart of 2023 YTD")

# Or with file inputs
tool.run("Analyze this dataset and plot something interesting about it.", ["examples/assets/iris.csv"])
```

To use the asynchronous version, simply replace `run` with `arun` and ensure your calling code is in an async context:

```python
import asyncio

tool = CodeInterpreter("Code Interpreter", "A tool to interpret code and generate useful outputs.")
asyncio.run(tool.arun("Plot the bitcoin chart of 2023 YTD"))

# Or with file inputs
asyncio.run(tool.arun("Analyze this dataset and plot something interesting about it.", ["examples/assets/iris.csv"]))
```

The `CodeInterpreter` class is a flexible tool that uses the `CodeInterpreterSession` from the `codeinterpreterapi` package to run the code interpretation and return the result. It provides both synchronous and asynchronous methods for convenience, and ensures that exceptions are handled gracefully.