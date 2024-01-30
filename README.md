![Swarming banner icon](images/swarmslogobanner.png)

<div align="center">

A modular framework that enables you to Build, Deploy, and Scale Reliable Autonomous Agents. Get started now below.

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/swarms)](https://github.com/kyegomez/swarms/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/swarms)](https://github.com/kyegomez/swarms/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/swarms)](https://github.com/kyegomez/swarms/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/swarms)](https://github.com/kyegomez/swarms/blob/main/LICENSE)[![GitHub star chart](https://img.shields.io/github/stars/kyegomez/swarms?style=social)](https://star-history.com/#kyegomez/swarms)[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/swarms)](https://libraries.io/github/kyegomez/swarms) [![Downloads](https://static.pepy.tech/badge/swarms/month)](https://pepy.tech/project/swarms)

[![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarms)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=Swarms%20-%20the%20future%20of%20AI) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&t=Swarms%20-%20the%20future%20of%20AI) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Swarms%20-%20the%20future%20of%20AI) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Swarms%20-%20the%20future%20of%20AI%20%23swarms%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)

</div>


----

## Installation
`pip3 install -U swarms`

---

## Usage
With Swarms, you can create structures, such as Agents, Swarms, and Workflows, that are composed of different types of tasks. Let's build a simple creative agent that will dynamically create a 10,000 word blog on health and wellness.

Run example in Collab: <a target="_blank" href="https://colab.research.google.com/github/kyegomez/swarms/blob/master/playground/swarms_example.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### `Agent`
A fully plug in and play Autonomous agent powered by an LLM extended by a long term memory database, and equipped with function calling for tool usage! By passing in an LLM you can create a fully autonomous agent with extreme customization and reliability ready for real-world task automation!

Features:

✅ Any LLM / Any framework

✅ Extremely customize-able with max loops, autosaving, import docs (PDFS, TXT, CSVs, etc), tool usage, etc etc

✅ Long term memory database with RAG (ChromaDB, Pinecone, Qdrant)

```python
import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import OpenAIChat, Agent

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    model_name="gpt-4",
    openai_api_key=api_key,
    max_tokens=4000
)


## Initialize the workflow
agent = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)

# Run the workflow on a task
agent.run("Generate a 10,000 word blog on health and wellness.")



```


### `ToolAgent`
ToolAgent is an agent that outputs JSON using any model from huggingface. It takes in an example schema with fields and then you provide it with a simple task and it'll output json! Perfect for function calling, parallel, and multi-step tool usage!

✅ Versatility: The ToolAgent class is designed to be flexible and adaptable. It can be used with any model and tokenizer, making it suitable for a wide range of tasks. This versatility means that you can use ToolAgent as a foundation for any tool that requires language model processing.

✅  Ease of Use: With its simple and intuitive interface, ToolAgent makes it easy to perform complex tasks. Just initialize it with your model, tokenizer, and JSON schema, and then call the run method with your task. This ease of use allows you to focus on your task, not on setting up your tools.

✅  Customizability: ToolAgent accepts variable length arguments and keyword arguments, allowing you to customize its behavior to suit your needs. Whether you need to adjust the temperature of the model's output, limit the number of tokens, or tweak any other parameter, ToolAgent has you covered. This customizability ensures that ToolAgent can adapt to your specific requirements.


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from swarms import ToolAgent


model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

task = "Generate a person's information based on the following schema:"
agent = ToolAgent(model=model, tokenizer=tokenizer, json_schema=json_schema)
generated_data = agent.run(task)

print(generated_data)

```


### `Worker`
The `Worker` is a simple all-in-one agent equipped with an LLM, tools, and RAG. Get started below:

✅ Plug in and Play LLM. Utilize any LLM from anywhere and any framework

✅ Reliable RAG: Utilizes FAISS for efficient RAG but it's modular so you can use any DB.

✅ Multi-Step Parallel Function Calling: Use any tool

```python
# Importing necessary modules
import os
from dotenv import load_dotenv
from swarms import Worker, OpenAIChat, tool

# Loading environment variables from .env file
load_dotenv()

# Retrieving the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")


# Create a tool
@tool
def search_api(query: str):
    pass


# Creating a Worker instance
worker = Worker(
    name="My Worker",
    role="Worker",
    human_in_the_loop=False,
    tools=[search_api],
    temperature=0.5,
    llm=OpenAIChat(openai_api_key=api_key),
)

# Running the worker with a prompt
out = worker.run(
    "Hello, how are you? Create an image of how your are doing!"
)

# Printing the output
print(out)


```

------

### `SequentialWorkflow`
Sequential Workflow enables you to sequentially execute tasks with `Agent` and then pass the output into the next agent and onwards until you have specified your max loops. `SequentialWorkflow` is wonderful for real-world business tasks like sending emails, summarizing documents, and analyzing data.


✅  Save and Restore Workflow states!

✅  Multi-Modal Support for Visual Chaining

✅  Utilizes Agent class

```python
import os 
from swarms import OpenAIChat, Agent, SequentialWorkflow 
from dotenv import load_dotenv

load_dotenv()

# Load the environment variables
api_key = os.getenv("OPENAI_API_KEY")


# Initialize the language agent
llm = OpenAIChat(
    temperature=0.5,
    model_name="gpt-4",
    openai_api_key=api_key,
    max_tokens=4000
)


# Initialize the agent with the language agent
agent1 = Agent(llm=llm, max_loops=1)

# Create another agent for a different task
agent2 = Agent(llm=llm, max_loops=1)

# Create another agent for a different task
agent3 = Agent(llm=llm, max_loops=1)

# Create the workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add(
    agent1, "Generate a 10,000 word blog on health and wellness.", 
)

# Suppose the next task takes the output of the first task as input
workflow.add(
    agent2, "Summarize the generated blog",
)

# Run the workflow
workflow.run()

# Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")
```



### `ConcurrentWorkflow`
`ConcurrentWorkflow` runs all the tasks all at the same time with the inputs you give it!


```python
import os
from dotenv import load_dotenv
from swarms import OpenAIChat, Task, ConcurrentWorkflow, Agent

# Load environment variables from .env file
load_dotenv()

# Load environment variables
llm = OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = Agent(llm=llm, max_loops=1)

# Create a workflow
workflow = ConcurrentWorkflow(max_workers=5)

# Create tasks
task1 = Task(agent, "What's the weather in miami")
task2 = Task(agent, "What's the weather in new york")
task3 = Task(agent, "What's the weather in london")

# Add tasks to the workflow
workflow.add(tasks=[task1, task2, task3])

# Run the workflow
workflow.run()

```

### `RecursiveWorkflow`
`RecursiveWorkflow` will keep executing the tasks until a specific token like <DONE> is located inside the text!

```python
import os 
from dotenv import load_dotenv 
from swarms import OpenAIChat, Task, RecursiveWorkflow, Agent

# Load environment variables from .env file
load_dotenv()

# Load environment variables
llm = OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = Agent(llm=llm, max_loops=1)

# Create a workflow
workflow = RecursiveWorkflow(stop_token="<DONE>")

# Create tasks
task1 = Task(agent, "What's the weather in miami")
task2 = Task(agent, "What's the weather in new york")
task3 = Task(agent, "What's the weather in london")

# Add tasks to the workflow
workflow.add(task1)
workflow.add(task2)
workflow.add(task3)

# Run the workflow
workflow.run()


```



### `ModelParallelizer`
The ModelParallelizer allows you to run multiple models concurrently, comparing their outputs. This feature enables you to easily compare the performance and results of different models, helping you make informed decisions about which model to use for your specific task.

Plug-and-Play Integration: The structure provides a seamless integration with various models, including OpenAIChat, Anthropic, Mixtral, and Gemini. You can easily plug in any of these models and start using them without the need for extensive modifications or setup.


```python
import os

from dotenv import load_dotenv

from swarms import Anthropic, Gemini, Mixtral, OpenAIChat, ModelParallelizer

load_dotenv()

# API Keys
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the models
llm = OpenAIChat(openai_api_key=openai_api_key)
anthropic = Anthropic(anthropic_api_key=anthropic_api_key)
mixtral = Mixtral()
gemini = Gemini(gemini_api_key=gemini_api_key)

# Initialize the parallelizer
llms = [llm, anthropic, mixtral, gemini]
parallelizer = ModelParallelizer(llms)

# Set the task
task = "Generate a 10,000 word blog on health and wellness."

# Run the task
out = parallelizer.run(task)

# Print the responses 1 by 1
for i in range(len(out)):
    print(f"Response from LLM {i}: {out[i]}")
```


### Simple Conversational Agent
A Plug in and play conversational agent with `GPT4`, `Mixytral`, or any of our models

- Reliable conversational structure to hold messages together with dynamic handling for long context conversations and interactions with auto chunking
- Reliable, this simple system will always provide responses you want.

```python
import os

from dotenv import load_dotenv

from swarms import (
    OpenAIChat,
    Conversation,
)

conv = Conversation(
    time_enabled=True,
)

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(openai_api_key=api_key, model_name="gpt-4")

# Run the language model in a loop
def interactive_conversation(llm):
    conv = Conversation()
    while True:
        user_input = input("User: ")
        conv.add("user", user_input)
        if user_input.lower() == "quit":
            break
        task = (
            conv.return_history_as_string()
        )  # Get the conversation history
        out = llm(task)
        conv.add("assistant", out)
        print(
            f"Assistant: {out}",
        )
    conv.display_conversation()
    conv.export_conversation("conversation.txt")


# Replace with your LLM instance
interactive_conversation(llm)

```


### `SwarmNetwork`
`SwarmNetwork` provides the infrasturcture for building extremely dense and complex multi-agent applications that span across various types of agents.

✅ Efficient Task Management: SwarmNetwork's intelligent agent pool and task queue management system ensures tasks are distributed evenly across agents. This leads to efficient use of resources and faster task completion.

✅ Scalability: SwarmNetwork can dynamically scale the number of agents based on the number of pending tasks. This means it can handle an increase in workload by adding more agents, and conserve resources when the workload is low by reducing the number of agents.

✅ Versatile Deployment Options: With SwarmNetwork, each agent can be run on its own thread, process, container, machine, or even cluster. This provides a high degree of flexibility and allows for deployment that best suits the user's needs and infrastructure.

```python
import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import OpenAIChat, Agent, SwarmNetwork

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    openai_api_key=api_key,
)

## Initialize the workflow
agent = Agent(llm=llm, max_loops=1, agent_name="Social Media Manager")
agent2 = Agent(llm=llm, max_loops=1, agent_name=" Product Manager")
agent3 = Agent(llm=llm, max_loops=1, agent_name="SEO Manager")


# Load the swarmnet with the agents
swarmnet = SwarmNetwork(
    agents=[agent, agent2, agent3],
)

# List the agents in the swarm network
out = swarmnet.list_agents()
print(out)

# Run the workflow on a task
out = swarmnet.run_single_agent(
    agent2.id, "Generate a 10,000 word blog on health and wellness."
)
print(out)


# Run all the agents in the swarm network on a task
out = swarmnet.run_many_agents(
    "Generate a 10,000 word blog on health and wellness."
)
print(out)

```


### `Task`
`Task` is a simple structure for task execution with the `Agent`. Imagine zapier for LLM-based workflow automation

✅ Task is a structure for task execution with the Agent. 

✅ Tasks can have descriptions, scheduling, triggers, actions, conditions, dependencies, priority, and a history. 

✅ The Task structure allows for efficient workflow automation with LLM-based agents.

```python
import os

from dotenv import load_dotenv

from swarms.structs import Agent, OpenAIChat, Task

# Load the environment variables
load_dotenv()


# Define a function to be used as the action
def my_action():
    print("Action executed")


# Define a function to be used as the condition
def my_condition():
    print("Condition checked")
    return True


# Create an agent
agent = Agent(
    llm=OpenAIChat(openai_api_key=os.environ["OPENAI_API_KEY"]),
    max_loops=1,
    dashboard=False,
)

# Create a task
task = Task(
    description=(
        "Generate a report on the top 3 biggest expenses for small"
        " businesses and how businesses can save 20%"
    ),
    agent=agent,
)

# Set the action and condition
task.set_action(my_action)
task.set_condition(my_condition)

# Execute the task
print("Executing task...")
task.run()

# Check if the task is completed
if task.is_completed():
    print("Task completed")
else:
    print("Task not completed")

# Output the result of the task
print(f"Task result: {task.result}")


```

---



### `BlockList`
- Modularity and Flexibility: BlocksList allows users to create custom swarms by adding or removing different classes or functions as blocks. This means users can easily tailor the functionality of their swarm to suit their specific needs.

- Ease of Management: With methods to add, remove, update, and retrieve blocks, BlocksList provides a straightforward way to manage the components of a swarm. This makes it easier to maintain and update the swarm over time.

- Enhanced Searchability: BlocksList offers methods to get blocks by various attributes such as name, type, ID, and parent-related properties. This makes it easier for users to find and work with specific blocks in a large and complex swarm.

```python
import os

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the models, structs, and telemetry modules
from swarms import (
    Gemini,
    GPT4VisionAPI,
    Mixtral,
    OpenAI,
    ToolAgent,
    BlocksList,
)

# Load the environment variables
load_dotenv()

# Get the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Tool Agent
model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-12b"
)
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {"type": "array", "items": {"type": "string"}},
    },
}
toolagent = ToolAgent(
    model=model, tokenizer=tokenizer, json_schema=json_schema
)

# Blocks List which enables you to build custom swarms by adding classes or functions
swarm = BlocksList(
    "SocialMediaSwarm",
    "A swarm of social media agents",
    [
        OpenAI(openai_api_key=openai_api_key),
        Mixtral(),
        GPT4VisionAPI(openai_api_key=openai_api_key),
        Gemini(gemini_api_key=gemini_api_key),
    ],
)


# Add the new block to the swarm
swarm.add(toolagent)

# Remove a block from the swarm
swarm.remove(toolagent)

# Update a block in the swarm
swarm.update(toolagent)

# Get a block at a specific index
block_at_index = swarm.get(0)

# Get all blocks in the swarm
all_blocks = swarm.get_all()

# Get blocks by name
openai_blocks = swarm.get_by_name("OpenAI")

# Get blocks by type
gpt4_blocks = swarm.get_by_type("GPT4VisionAPI")

# Get blocks by ID
block_by_id = swarm.get_by_id(toolagent.id)

# Get blocks by parent
blocks_by_parent = swarm.get_by_parent(swarm)

# Get blocks by parent ID
blocks_by_parent_id = swarm.get_by_parent_id(swarm.id)

# Get blocks by parent name
blocks_by_parent_name = swarm.get_by_parent_name(swarm.name)

# Get blocks by parent type
blocks_by_parent_type = swarm.get_by_parent_type(type(swarm).__name__)

# Get blocks by parent description
blocks_by_parent_description = swarm.get_by_parent_description(
    swarm.description
)

# Run the block in the swarm
inference = swarm.run_block(toolagent, "Hello World")
print(inference)
```


## Real-World Deployment

### Multi-Agent Swarm for Logistics
Here's a production grade swarm ready for real-world deployment in a factory and logistics settings like warehouses. This swarm can automate 3 costly and inefficient workflows, safety checks, productivity checks, and warehouse security.


```python
from swarms.structs import Agent
import os
from dotenv import load_dotenv
from swarms.models import GPT4VisionAPI
from swarms.prompts.logistics import (
    Health_Security_Agent_Prompt,
    Quality_Control_Agent_Prompt,
    Productivity_Agent_Prompt,
    Safety_Agent_Prompt,
    Security_Agent_Prompt,
    Sustainability_Agent_Prompt,
    Efficiency_Agent_Prompt,
)

# Load ENV
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# GPT4VisionAPI 
llm = GPT4VisionAPI(openai_api_key=api_key)

# Image for analysis
factory_image = "factory_image1.jpg"

# Initialize agents with respective prompts
health_security_agent = Agent(
    llm=llm,
    sop=Health_Security_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Quality control agent
quality_control_agent = Agent(
    llm=llm,
    sop=Quality_Control_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)


# Productivity Agent
productivity_agent = Agent(
    llm=llm,
    sop=Productivity_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Initiailize safety agent
safety_agent = Agent(
    llm=llm, sop=Safety_Agent_Prompt, max_loops=1, multi_modal=True
)

# Init the security agent
security_agent = Agent(
    llm=llm, sop=Security_Agent_Prompt, max_loops=1, multi_modal=True
)


# Initialize sustainability agent
sustainability_agent = Agent(
    llm=llm,
    sop=Sustainability_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)


# Initialize efficincy agent
efficiency_agent = Agent(
    llm=llm,
    sop=Efficiency_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Run agents with respective tasks on the same image
health_analysis = health_security_agent.run(
    "Analyze the safety of this factory", factory_image
)
quality_analysis = quality_control_agent.run(
    "Examine product quality in the factory", factory_image
)
productivity_analysis = productivity_agent.run(
    "Evaluate factory productivity", factory_image
)
safety_analysis = safety_agent.run(
    "Inspect the factory's adherence to safety standards",
    factory_image,
)
security_analysis = security_agent.run(
    "Assess the factory's security measures and systems",
    factory_image,
)
sustainability_analysis = sustainability_agent.run(
    "Examine the factory's sustainability practices", factory_image
)
efficiency_analysis = efficiency_agent.run(
    "Analyze the efficiency of the factory's manufacturing process",
    factory_image,
)
```
---


## `Multi Modal Autonomous Agents`
Run the agent with multiple modalities useful for various real-world tasks in manufacturing, logistics, and health.

```python
# Description: This is an example of how to use the Agent class to run a multi-modal workflow
import os
from dotenv import load_dotenv
from swarms.models.gpt4_vision_api import GPT4VisionAPI
from swarms.structs import Agent

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = GPT4VisionAPI(
    openai_api_key=api_key,
    max_tokens=500,
)

# Initialize the task
task = (
    "Analyze this image of an assembly line and identify any issues such as"
    " misaligned parts, defects, or deviations from the standard assembly"
    " process. IF there is anything unsafe in the image, explain why it is"
    " unsafe and how it could be improved."
)
img = "assembly_line.jpg"

## Initialize the workflow
agent = Agent(
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=True,
    multi_modal=True
)

# Run the workflow on a task
agent.run(task=task, img=img)


```

---

## Multi-Modal Model APIs

### `Gemini`
- Deploy Gemini from Google with utmost reliability with our visual chain of thought prompt that enables more reliable responses
```python
import os

from dotenv import load_dotenv

from swarms import Gemini
from swarms.prompts.visual_cot import VISUAL_CHAIN_OF_THOUGHT

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("GEMINI_API_KEY")

# Initialize the language model
llm = Gemini(
    gemini_api_key=api_key,
    temperature=0.5,
    max_tokens=1000,
    system_prompt=VISUAL_CHAIN_OF_THOUGHT,
)

# Initialize the task
task = "This is an eye test. What do you see?"
img = "playground/demos/multi_modal_chain_of_thought/eyetest.jpg"

# Run the workflow on a task
out = llm.run(task=task, img=img)
print(out)
```

### `GPT4Vision`
```python
from swarms import GPT4VisionAPI

# Initialize with default API key and custom max_tokens
api = GPT4VisionAPI(max_tokens=1000)

# Define the task and image URL
task = "Describe the scene in the image."
img = "https://i.imgur.com/4P4ZRxU.jpeg"

# Run the GPT-4 Vision model
response = api.run(task, img)

# Print the model's response
print(response)
```

### `QwenVLMultiModal`
A radically simple interface for QwenVLMultiModal comes complete with Quantization to turn it on just set quantize to true!

```python
from swarms import QwenVLMultiModal

# Instantiate the QwenVLMultiModal model
model = QwenVLMultiModal(
    model_name="Qwen/Qwen-VL-Chat",
    device="cuda",
    quantize=True,
)

# Run the model
response = model(
    "Hello, how are you?", "https://example.com/image.jpg"
)

# Print the response
print(response)


```


### `Kosmos`
- Multi-Modal Model from microsoft!

```python
from swarms import Kosmos

# Initialize the model
model = Kosmos()

# Generate
out = model.run("Analyze the reciepts in this image", "docs.jpg")

# Print the output
print(out)

```


### `Idefics`
- Multi-Modal model from Huggingface team!

```python
# Import the idefics model from the swarms.models module
from swarms.models import Idefics

# Create an instance of the idefics model
model = Idefics()

# Define user input with an image URL and chat with the model
user_input = (
    "User: What is in this image?"
    " https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
)
response = model.chat(user_input)
print(response)

# Define another user input with an image URL and chat with the model
user_input = (
    "User: And who is that?"
    " https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
)
response = model.chat(user_input)
print(response)

# Set the checkpoint of the model to "new_checkpoint"
model.set_checkpoint("new_checkpoint")

# Set the device of the model to "cpu"
model.set_device("cpu")

# Set the maximum length of the chat to 200
model.set_max_length(200)

# Clear the chat history of the model
model.clear_chat_history()


```

## Radically Simple AI Model APIs
We provide a vast array of language and multi-modal model APIs for you to generate text, images, music, speech, and even videos. Get started below:



-----


### `Anthropic`
```python
# Import necessary modules and classes
from swarms.models import Anthropic

# Initialize an instance of the Anthropic class
model = Anthropic(
    anthropic_api_key=""
)

# Using the run method
completion_1 = model.run("What is the capital of France?")
print(completion_1)

# Using the __call__ method
completion_2 = model("How far is the moon from the earth?", stop=["miles", "km"])
print(completion_2)

```


### `HuggingFaceLLM`
```python
from swarms.models import HuggingfaceLLM

# Initialize with custom configuration
custom_config = {
    "quantize": True,
    "quantization_config": {"load_in_4bit": True},
    "verbose": True
}
inference = HuggingfaceLLM(model_id="NousResearch/Nous-Hermes-2-Vision-Alpha", **custom_config)

# Generate text based on a prompt
prompt_text = "Create a list of known biggest risks of structural collapse with references"
generated_text = inference(prompt_text)
print(generated_text)
```

### `Mixtral`
- Utilize Mixtral in a very simple API,
- Utilize 4bit quantization for a increased speed and less memory usage
- Use Flash Attention 2.0 for increased speed and less memory usage
```python
from swarms.models import Mixtral

# Initialize the Mixtral model with 4 bit and flash attention!
mixtral = Mixtral(load_in_4bit=True, use_flash_attention_2=True)

# Generate text for a simple task
generated_text = mixtral.run("Generate a creative story.")

# Print the generated text
print(generated_text)
```


### `Dalle3`
```python
from swarms import Dalle3

# Create an instance of the Dalle3 class with high quality
dalle3 = Dalle3(quality="high")

# Define a text prompt
task = "A high-quality image of a sunset"

# Generate a high-quality image from the text prompt
image_url = dalle3(task)

# Print the generated image URL
print(image_url)
```




### Text to Video with `ZeroscopeTTV`

```python
# Import the model
from swarms import ZeroscopeTTV

# Initialize the model
zeroscope = ZeroscopeTTV()

# Specify the task
task = "A person is walking on the street."

# Generate the video!
video_path = zeroscope(task)
print(video_path)

```


<!-- ### ModelScope
```python
from swarms.models import ModelScopeAutoModel

# Initialize the model
mp = ModelScopeAutoModel(
    model_name="AI-ModelScope/Mixtral-8x7B-Instruct-v0.1",
)

mp.run("Generate a 10,000 word blog on health and wellness.")
```

```python

from swarms import CogAgent

# Initialize CogAgent
cog_agent = CogAgent()

# Run the model on the tests
cog_agent.run("Describe this scene", "images/1.jpg")

``` -->



----

## Supported Models ✅ 
Swarms supports various model providers like OpenAI, Huggingface, Anthropic, Google, Mistral and many more.

| Provider | Provided ✅  | Module Name |
|----------|-----------------------------|-------------|
| OpenAI | ✅  | OpenAIChat, OpenAITTS, GPT4VisionAPI, Dalle3 |
| Anthropic | ✅  | Anthropic |
| Mistral | ✅  | Mistral, Mixtral |
| Gemini/Palm | ✅  | Gemini |
| Huggingface | ✅  | HuggingFaceLLM |
| Modelscope | ✅  | Modelscope |
| Vllm | ✅  | vLLM |


---

# Features 🤖 
The Swarms framework is designed with a strong emphasis on reliability, performance, and production-grade readiness. 
Below are the key features that make Swarms an ideal choice for enterprise-level AI deployments.

## 🚀 Production-Grade Readiness
- **Scalable Architecture**: Built to scale effortlessly with your growing business needs.
- **Enterprise-Level Security**: Incorporates top-notch security features to safeguard your data and operations.
- **Containerization and Microservices**: Easily deployable in containerized environments, supporting microservices architecture.

## ⚙️ Reliability and Robustness
- **Fault Tolerance**: Designed to handle failures gracefully, ensuring uninterrupted operations.
- **Consistent Performance**: Maintains high performance even under heavy loads or complex computational demands.
- **Automated Backup and Recovery**: Features automatic backup and recovery processes, reducing the risk of data loss.

## 💡 Advanced AI Capabilities

The Swarms framework is equipped with a suite of advanced AI capabilities designed to cater to a wide range of applications and scenarios, ensuring versatility and cutting-edge performance.

### Multi-Modal Autonomous Agents
- **Versatile Model Support**: Seamlessly works with various AI models, including NLP, computer vision, and more, for comprehensive multi-modal capabilities.
- **Context-Aware Processing**: Employs context-aware processing techniques to ensure relevant and accurate responses from agents.

### Function Calling Models for API Execution
- **Automated API Interactions**: Function calling models that can autonomously execute API calls, enabling seamless integration with external services and data sources.
- **Dynamic Response Handling**: Capable of processing and adapting to responses from APIs for real-time decision making.

### Varied Architectures of Swarms
- **Flexible Configuration**: Supports multiple swarm architectures, from centralized to decentralized, for diverse application needs.
- **Customizable Agent Roles**: Allows customization of agent roles and behaviors within the swarm to optimize performance and efficiency.

### Generative Models
- **Advanced Generative Capabilities**: Incorporates state-of-the-art generative models to create content, simulate scenarios, or predict outcomes.
- **Creative Problem Solving**: Utilizes generative AI for innovative problem-solving approaches and idea generation.

### Enhanced Decision-Making
- **AI-Powered Decision Algorithms**: Employs advanced algorithms for swift and effective decision-making in complex scenarios.
- **Risk Assessment and Management**: Capable of assessing risks and managing uncertain situations with AI-driven insights.

### Real-Time Adaptation and Learning
- **Continuous Learning**: Agents can continuously learn and adapt from new data, improving their performance and accuracy over time.
- **Environment Adaptability**: Designed to adapt to different operational environments, enhancing robustness and reliability.


## 🔄 Efficient Workflow Automation
- **Streamlined Task Management**: Simplifies complex tasks with automated workflows, reducing manual intervention.
- **Customizable Workflows**: Offers customizable workflow options to fit specific business needs and requirements.
- **Real-Time Analytics and Reporting**: Provides real-time insights into agent performance and system health.

## 🌐 Wide-Ranging Integration
- **API-First Design**: Easily integrates with existing systems and third-party applications via robust APIs.
- **Cloud Compatibility**: Fully compatible with major cloud platforms for flexible deployment options.
- **Continuous Integration/Continuous Deployment (CI/CD)**: Supports CI/CD practices for seamless updates and deployment.

## 📊 Performance Optimization
- **Resource Management**: Efficiently manages computational resources for optimal performance.
- **Load Balancing**: Automatically balances workloads to maintain system stability and responsiveness.
- **Performance Monitoring Tools**: Includes comprehensive monitoring tools for tracking and optimizing performance.

## 🛡️ Security and Compliance
- **Data Encryption**: Implements end-to-end encryption for data at rest and in transit.
- **Compliance Standards Adherence**: Adheres to major compliance standards ensuring legal and ethical usage.
- **Regular Security Updates**: Regular updates to address emerging security threats and vulnerabilities.

## 💬 Community and Support
- **Extensive Documentation**: Detailed documentation for easy implementation and troubleshooting.
- **Active Developer Community**: A vibrant community for sharing ideas, solutions, and best practices.
- **Professional Support**: Access to professional support for enterprise-level assistance and guidance.

Swarms framework is not just a tool but a robust, scalable, and secure partner in your AI journey, ready to tackle the challenges of modern AI applications in a business environment.

---

## Documentation
Documentation is located here at: [swarms.apac.ai](https://swarms.apac.ai)

----

## 🫶 Contributions:

The easiest way to contribute is to pick any issue with the `good first issue` tag 💪. Read the Contributing guidelines [here](/CONTRIBUTING.md). Bug Report? [File here](https://github.com/swarms/gateway/issues) | Feature Request? [File here](https://github.com/swarms/gateway/issues)

Swarms is an open-source project, and contributions are VERY welcome. If you want to contribute, you can create new features, fix bugs, or improve the infrastructure. Please refer to the [CONTRIBUTING.md](https://github.com/kyegomez/swarms/blob/master/CONTRIBUTING.md) and our [contributing board](https://github.com/users/kyegomez/projects/1) to participate in Roadmap discussions!

<a href="https://github.com/kyegomez/swarms/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/swarms" />
</a>

----

## Community

Join our growing community around the world, for real-time support, ideas, and discussions on Swarms 😊 

- View our official [Blog](https://swarms.apac.ai)
- Chat live with us on [Discord](https://discord.gg/kS3rwKs3ZC)
- Follow us on [Twitter](https://twitter.com/kyegomez)
- Connect with us on [LinkedIn](https://www.linkedin.com/company/the-swarm-corporation)
- Visit us on [YouTube](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ)
- [Join the Swarms community on Discord!](https://discord.gg/AJazBmhKnr)
- Join our Swarms Community Gathering every Thursday at 1pm NYC Time to unlock the potential of autonomous agents in automating your daily tasks [Sign up here](https://lu.ma/5p2jnc2v)

---

## Discovery Call
Book a discovery call to learn how Swarms can lower your operating costs by 40% with swarms of autonomous agents in lightspeed. [Click here to book a time that works for you!](https://calendly.com/swarm-corp/30min?month=2023-11)

## Accelerate Backlog
Help us accelerate our backlog by supporting us financially! Note, we're an open source corporation and so all the revenue we generate is through donations at the moment ;)

<a href="https://polar.sh/kyegomez"><img src="https://polar.sh/embed/fund-our-backlog.svg?org=kyegomez" /></a>

## Swarm Newsletter 🤖 🤖 🤖 📧 
Sign up to the Swarm newsletter to receive  updates on the latest Autonomous agent research papers, step by step guides on creating multi-agent app, and much more Swarmie goodiness 😊


[CLICK HERE TO SIGNUP](https://docs.google.com/forms/d/e/1FAIpQLSfqxI2ktPR9jkcIwzvHL0VY6tEIuVPd-P2fOWKnd6skT9j1EQ/viewform?usp=sf_link)

# License
Apache License



