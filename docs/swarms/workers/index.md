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
from swarms.models import OpenAIChat

llm = OpenAIChat(
    # enter your api key
    openai_api_key="",
    temperature=0.5,
)

node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    openai_api_key="",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)
```

### **Example 2**: Usage with custom tools:

```python
import os

import interpreter

from swarms.agents.hf_agents import HFAgent
from swarms.agents.omni_modal_agent import OmniModalAgent
from swarms.models import OpenAIChat
from swarms.tools.autogpt import tool
from swarms.workers import Worker

# Initialize API Key
api_key = ""


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)


# wrap a function with the tool decorator to make it a tool, then add docstrings for tool documentation
@tool
def hf_agent(task: str = None):
    """
    An tool that uses an openai model to call and respond to a task by search for a model on huggingface
    It first downloads the model then uses it.

    Rules: Don't call this model for simple tasks like generating a summary, only call this tool for multi modal tasks like generating images, videos, speech, etc

    """
    agent = HFAgent(model="text-davinci-003", api_key=api_key)
    response = agent.run(task, text="¡Este es un API muy agradable!")
    return response


# wrap a function with the tool decorator to make it a tool
@tool
def omni_agent(task: str = None):
    """
    An tool that uses an openai Model to utilize and call huggingface models and guide them to perform a task.

    Rules: Don't call this model for simple tasks like generating a summary, only call this tool for multi modal tasks like generating images, videos, speech
    The following tasks are what this tool should be used for:

    Tasks omni agent is good for:
    --------------
    document-question-answering
    image-captioning
    image-question-answering
    image-segmentation
    speech-to-text
    summarization
    text-classification
    text-question-answering
    translation
    huggingface-tools/text-to-image
    huggingface-tools/text-to-video
    text-to-speech
    huggingface-tools/text-download
    huggingface-tools/image-transformation
    """
    agent = OmniModalAgent(llm)
    response = agent.run(task)
    return response


# Code Interpreter
@tool
def compile(task: str):
    """
    Open Interpreter lets LLMs run code (Python, Javascript, Shell, and more) locally.
    You can chat with Open Interpreter through a ChatGPT-like interface in your terminal
    by running $ interpreter after installing.

    This provides a natural-language interface to your computer's general-purpose capabilities:

    Create and edit photos, videos, PDFs, etc.
    Control a Chrome browser to perform research
    Plot, clean, and analyze large datasets
    ...etc.
    ⚠️ Note: You'll be asked to approve code before it's run.

    Rules: Only use when given to generate code or an application of some kind
    """
    task = interpreter.chat(task, return_messages=True)
    interpreter.chat()
    interpreter.reset(task)

    os.environ["INTERPRETER_CLI_AUTO_RUN"] = True
    os.environ["INTERPRETER_CLI_FAST_MODE"] = True
    os.environ["INTERPRETER_CLI_DEBUG"] = True


# Append tools to an list
tools = [hf_agent, omni_agent, compile]


# Initialize a single Worker node with previously defined tools in addition to it's
# predefined tools
node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    openai_api_key=api_key,
    ai_role="Worker in a swarm",
    external_tools=tools,
    human_in_the_loop=False,
    temperature=0.5,
)

# Specify task
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."

# Run the node on the task
response = node.run(task)

# Print the response
print(response)
```

### **Example 3**: Usage with human in the loop:

```python
from swarms import Worker
from swarms.models import OpenAIChat

llm = OpenAIChat(
    # enter your api key
    openai_api_key="",
    temperature=0.5,
)

node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    openai_api_key="",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=True,
    temperature=0.5,
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)
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