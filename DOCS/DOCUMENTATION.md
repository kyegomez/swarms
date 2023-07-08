# Swarms Documentation

## Overview
The Swarm module includes the implementation of two classes, `WorkerNode` and `BossNode`, which respectively represent a worker agent and a boss agent. A worker agent is responsible for completing given tasks, while a boss agent is responsible for creating and managing tasks for the worker agent(s).

## Key Classes

### WorkerNode
The `WorkerNode` class represents an autonomous agent instance that functions as a worker to accomplish complex tasks. It has the ability to search the internet, process and generate images, text, audio, and more.

#### Constructor
```python
def __init__(self, llm, tools, vectorstore)
```
- `llm` (required): The language model used by the worker node.
- `tools` (required): A list of tools available to the worker node.
- `vectorstore` (required): The vector store used by the worker node.

#### Methods
- `create_agent(ai_name, ai_role, human_in_the_loop, search_kwargs)`: Creates an agent within the worker node.
- `add_tool(tool)`: Adds a tool to the worker node.
- `run(prompt)`: Runs the worker node to complete a task specified by the prompt.


#### Example Usage

```python


from swarms import worker_node

# Your OpenAI API key
api_key = "sk-your api key"

# Initialize a WorkerNode with your API key
node = worker_node(api_key)

# Define an objective
objective = "Please make a web GUI for using HTTP API server..."

# Run the task
task = node.run(objective)

print(task)
```

### BossNode
The `BossNode` class represents an agent responsible for creating and managing tasks for the worker agent(s). It interacts with the worker node(s) to delegate tasks and monitor their progress.

#### Constructor
```python
def __init__(self, llm, vectorstore, agent_executor, max_iterations)
```
- `llm` (required): The language model used by the boss node.
- `vectorstore` (required): The vector store used by the boss node.
- `agent_executor` (required): The agent executor used to execute tasks.
- `max_iterations` (required): The maximum number of iterations for task execution.

#### Methods
- `create_task(objective)`: Creates a task with the given objective.
- `execute_task(task)`: Executes the given task by interacting with the worker agent(s).

### LLM
The `LLM` class is a utility class that provides an interface to different language models (LLMs) such as OpenAI's ChatGPT and Hugging Face models. It is used to initialize the language model for the worker and boss nodes.

#### Constructor
```python
def __init__(self, openai_api_key=None, hf_repo_id=None, hf_api_token=None, model_kwargs=None)
```
- `openai_api_key` (optional): The API key for OpenAI's models.
- `hf_repo_id` (optional): The repository ID for the Hugging Face model.
- `hf_api_token` (optional): The API token for the Hugging Face model.
- `model_kwargs` (optional): Additional keyword arguments to pass to the language model.

#### Methods
- `run(prompt)`: Runs the language model with the given prompt and returns the generated response.

### Swarms
The `Swarms` class is a wrapper class that encapsulates the functionality of the worker and boss nodes. It provides a convenient way to initialize and run a swarm of agents to accomplish tasks.

#### Constructor
```python
def __init__(self, openai_api_key)
```
- `openai_api_key` (required): The API key for OpenAI's models.

#### Methods
- `run_swarms(objective)`: Runs the swarm with the given objective by initializing the worker and boss nodes.

## Example Usage
```python
from swarms import Swarms

api_key = "sksdsds"

# Initialize Swarms with your API key
swarm = Swarms(openai_api_key=api_key)

# Define an objective
objective = """
Please make a web GUI for using HTTP API server. 
The name of it is Swarms. 
You can check the server code at ./main.py. 
The server is served on localhost:8000. 
Users should be able to write text input as 'query' and url array as 'files', and check the response. 
Users input form should be delivered in JSON format. 
I want it to have neumorphism-style. Serve it on port 4500.

"""

# Run Swarms
task = swarm.run_swarms(objective)

print(task)
```

This will create a swarm of agents to complete the given objective. The boss agent will create tasks and delegate them to the worker agent(s) for execution.

Please make sure to replace `"your_openai_api_key"` with your actual OpenAI API key.

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
The Swarms module provides a powerful framework for creating and managing swarms of autonomous agents to accomplish complex tasks. With the WorkerNode and BossNode classes, along with the LLM utility class, you can easily set up and run a swarm of agents to tackle any objective. The module is highly configurable and extensible, allowing you to tailor it to your specific needs.