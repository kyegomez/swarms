# Tutorial: Understanding and Utilizing Worker Examples

## Table of Contents
1. Introduction
2. Code Overview
   - Import Statements
   - Initializing API Key and Language Model
   - Creating Swarm Tools
   - Appending Tools to a List
   - Initializing a Worker Node
3. Understanding the `hf_agent` Tool
4. Understanding the `omni_agent` Tool
5. Understanding the `compile` Tool
6. Running a Swarm
7. Interactive Examples
   - Example 1: Initializing API Key and Language Model
   - Example 2: Using the `hf_agent` Tool
   - Example 3: Using the `omni_agent` Tool
   - Example 4: Using the `compile` Tool
8. Conclusion

## 1. Introduction
The provided code showcases a system built around a worker node that utilizes various AI models and tools to perform tasks. This tutorial will break down the code step by step, explaining its components, how they work together, and how to utilize its modularity for various tasks.

## 2. Code Overview

### Import Statements
The code begins with import statements, bringing in necessary modules and classes. Key imports include the `OpenAIChat` class, which represents a language model, and several custom agents and tools from the `swarms` package.

```python
import interpreter  # Assuming this is a custom module

from swarms.agents.hf_agents import HFAgent
from swarms.agents.omni_modal_agent import OmniModalAgent
from swarms.models import OpenAIChat
from swarms.tools.autogpt import tool
from swarms.workers import Worker
```

### Initializing API Key and Language Model
Here, an API key is initialized, and a language model (`OpenAIChat`) is created. This model is capable of generating human-like text based on the provided input.

```python
# Initialize API Key
api_key = "YOUR_OPENAI_API_KEY"

# Initialize the language model
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)
```

### Creating Swarm Tools
The code defines three tools: `hf_agent`, `omni_agent`, and `compile`. These tools encapsulate specific functionalities and can be invoked to perform tasks.

### Appending Tools to a List
All defined tools are appended to a list called `tools`. This list is later used when initializing a worker node, allowing the node to access and utilize these tools.

```python
# Append tools to a list
tools = [hf_agent, omni_agent, compile]
```

### Initializing a Worker Node
A worker node is initialized using the `Worker` class. The worker node is equipped with the language model, a name, API key, and the list of tools. It's set up to perform tasks without human intervention.

```python
# Initialize a single Worker node with previously defined tools in addition to its predefined tools
node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    openai_api_key=api_key,
    ai_role="Worker in a swarm",
    external_tools=tools,
    human_in_the_loop=False,
    temperature=0.5,
)
```

## 3. Understanding the `hf_agent` Tool
The `hf_agent` tool utilizes an OpenAI model (`text-davinci-003`) to perform tasks. It takes a task as input and returns a response. This tool is suitable for multi-modal tasks like generating images, videos, speech, etc. The tool's primary rule is not to be used for simple tasks like generating summaries.

```python
@tool
def hf_agent(task: str = None):
    # Create an HFAgent instance with the specified model and API key
    agent = HFAgent(model="text-davinci-003", api_key=api_key)
    # Run the agent with the provided task and optional text input
    response = agent.run(task, text="¡Este es un API muy agradable!")
    return response
```

## 4. Understanding the `omni_agent` Tool
The `omni_agent` tool is more versatile and leverages the `llm` (language model) to interact with Huggingface models for various tasks. It's intended for multi-modal tasks such as document-question-answering, image-captioning, summarization, and more. The tool's rule is also not to be used for simple tasks.

```python
@tool
def omni_agent(task: str = None):
    # Create an OmniModalAgent instance with the provided language model
    agent = OmniModalAgent(llm)
    # Run the agent with the provided task
    response = agent.run(task)
    return response
```

## 5. Understanding the `compile` Tool
The `compile` tool allows the execution of code locally, supporting various programming languages like Python, JavaScript, and Shell. It provides a natural language interface to your computer's capabilities. Users can chat with this tool in a terminal-like interface to perform tasks such as creating and editing files, controlling a browser, and more.

```python
@tool
def compile(task: str):
    # Use the interpreter module to chat with the local interpreter
    task = interpreter.chat(task, return_messages=True)
    interpreter.chat()
    interpreter.reset(task)

    # Set environment variables for the interpreter
    os.environ["INTERPRETER_CLI_AUTO_RUN"] = True
    os.environ["INTERPRETER_CLI_FAST_MODE"] = True
    os.environ["INTERPRETER_CLI_DEBUG"] = True
```

## 6. Running a Swarm
After defining tools and initializing the worker node, a specific task is provided as input to the worker node. The node then runs the task, and the response is printed to the console.

```python
# Specify the task
task = "What were the winning Boston Marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."

# Run the node on the task
response = node.run(task)

# Print the response
print(response)
```


## Full Code
- The full code example of stacked swarms

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


## 8. Conclusion
In this extensive tutorial, we've embarked on a journey to explore a sophisticated system designed to harness the power of AI models and tools for a myriad of tasks. We've peeled back the layers of code, dissected its various components, and gained a profound understanding of how these elements come together to create a versatile, modular, and powerful swarm-based AI system.

## What We've Learned

Throughout this tutorial, we've covered the following key aspects:

### Code Structure and Components
We dissected the code into its fundamental building blocks:
- **Import Statements:** We imported necessary modules and libraries, setting the stage for our system's functionality.
- **Initializing API Key and Language Model:** We learned how to set up the essential API key and initialize the language model, a core component for text generation and understanding.
- **Creating Swarm Tools:** We explored how to define tools, encapsulating specific functionalities that our system can leverage.
- **Appending Tools to a List:** We aggregated our tools into a list, making them readily available for use.
- **Initializing a Worker Node:** We created a worker node equipped with tools, a name, and configuration settings.

### Tools and Their Functions
We dove deep into the purpose and functionality of three crucial tools:
- **`hf_agent`:** We understood how this tool employs an OpenAI model for multi-modal tasks, and its use cases beyond simple summarization.
- **`omni_agent`:** We explored the versatility of this tool, guiding Huggingface models to perform a wide range of multi-modal tasks.
- **`compile`:** We saw how this tool allows the execution of code in multiple languages, providing a natural language interface for various computational tasks.

### Interactive Examples
We brought the code to life through interactive examples, showcasing how to initialize the language model, generate text, perform document-question-answering, and execute code—all with practical, real-world scenarios.

## A Recap: The Worker Node's Role

At the heart of this system lies the "Worker Node," a versatile entity capable of wielding the power of AI models and tools to accomplish tasks. The Worker Node's role is pivotal in the following ways:

1. **Task Execution:** It is responsible for executing tasks, harnessing the capabilities of the defined tools to generate responses or perform actions.

2. **Modularity:** The Worker Node benefits from the modularity of the system. It can easily access and utilize a variety of tools, allowing it to adapt to diverse tasks and requirements.

3. **Human in the Loop:** While the example here is configured to operate without human intervention, the Worker Node can be customized to incorporate human input or approval when needed.

4. **Integration:** It can be extended to integrate with other AI models, APIs, or services, expanding its functionality and versatility.

## The Road Ahead: Future Features and Enhancements

As we conclude this tutorial, let's peek into the future of this system. While the current implementation is already powerful, there is always room for growth and improvement. Here are some potential future features and enhancements to consider:

### 1. Enhanced Natural Language Understanding
   - **Semantic Understanding:** Improve the system's ability to understand context and nuances in natural language, enabling more accurate responses.

### 2. Multimodal Capabilities
   - **Extended Multimodal Support:** Expand the `omni_agent` tool to support additional types of multimodal tasks, such as video generation or audio processing.

### 3. Customization and Integration
   - **User-defined Tools:** Allow users to define their own custom tools, opening up endless possibilities for tailoring the system to specific needs.

### 4. Collaborative Swarms
   - **Swarm Collaboration:** Enable multiple Worker Nodes to collaborate on complex tasks, creating a distributed, intelligent swarm system.

### 5. User-Friendly Interfaces
   - **Graphical User Interface (GUI):** Develop a user-friendly GUI for easier interaction and task management, appealing to a wider audience.

### 6. Continuous Learning
   - **Active Learning:** Implement mechanisms for the system to learn and adapt over time, improving its performance with each task.

### 7. Security and Privacy
   - **Enhanced Security:** Implement robust security measures to safeguard sensitive data and interactions within the system.

### 8. Community and Collaboration
   - **Open Source Community:** Foster an open-source community around the system, encouraging contributions and innovation from developers worldwide.

### 9. Integration with Emerging Technologies
   - **Integration with Emerging AI Models:** Keep the system up-to-date by seamlessly integrating with new and powerful AI models as they emerge in the industry.

## In Conclusion

In this tutorial, we've journeyed through a complex AI system, unraveling its inner workings, and understanding its potential. We've witnessed how code can transform into a powerful tool, capable of handling a vast array of tasks, from generating creative stories to executing code snippets.

As we conclude, we stand at the threshold of an exciting future for AI and technology. This system, with its modular design and the potential for continuous improvement, embodies the spirit of innovation and adaptability. Whether you're a developer, a researcher, or an enthusiast, the possibilities are boundless, and the journey is just beginning.

Embrace this knowledge, explore the system, and embark on your own quest to shape the future of AI. With each line of code, you have the power to transform ideas into reality and unlock new horizons of innovation. The future is yours to create, and the tools are at your fingertips.