# Effortlessly coordinate swarms of agents for production-grade applications.

Individual agents currently face 5 significant challenges that hinder their deployment in production: 
- Short memory 
- Single-task threading 
- Hallucinations
- High cost
- Lack of multi-agent collaboration. 

The Swarms framework is a solution to all these issues. Swarms provides simple, reliable, and versatile tools to create your own swarm of agents tailored to your exact needs. 

Currently used in production by...
- The Royal Bank of Canada
- John Deere
- Many AI startups

----

## Installation
`$ pip3 install -U swarms`

---

## Usage


Run example in Collab: <a target="_blank" href="https://colab.research.google.com/github/kyegomez/swarms/blob/master/playground/swarms_example.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### `Agent`
A fully plug-and-play autonomous agent powered by an LLM extended by a long-term memory database, and equipped with function calling for tool usage! By passing in an LLM, you can create a fully autonomous agent with extreme customization and reliability, ready for real-world task automation!

Features:

✅ Any LLM / Any framework

✅ Extremely customize-able with max loops, autosaving, import docs (PDFS, TXT, CSVs, etc), tool usage, etc etc

✅ Long term memory database with RAG (ChromaDB, Pinecone, Qdrant)

```python
import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import Agent, OpenAIChat

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5, model_name="gpt-4", openai_api_key=api_key, max_tokens=4000
)


## Initialize the workflow
agent = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)

# Run the workflow on a task
agent.run("Generate a 10,000 word blog on health and wellness.")
```


# `Agent` with Long Term Memory
`Agent` equipped with quasi-infinite long term memory. Great for long document understanding, analysis, and retrieval.

```python
from swarms import Agent, OpenAIChat
from playground.memory.chromadb_example import ChromaDB # Copy and paste the code and put it in your own local directory.

# Making an instance of the ChromaDB class
memory = ChromaDB(
    metric="cosine",
    n_results=3,
    output_dir="results",
    docs_folder="docs",
)

# Initializing the agent with the Gemini instance and other parameters
agent = Agent(
    agent_name="Covid-19-Chat",
    agent_description=(
        "This agent provides information about COVID-19 symptoms."
    ),
    llm=OpenAIChat(),
    max_loops="auto",
    autosave=True,
    verbose=True,
    long_term_memory=memory,
    stopping_condition="finish",
)

# Defining the task and image path
task = ("What are the symptoms of COVID-19?",)

# Running the agent with the specified task and image
out = agent.run(task)
print(out)

```


# `Agent` with Long Term Memory ++ Tools!
An LLM equipped with long term memory and tools, a full stack agent capable of automating all and any digital tasks given a good prompt.

```python
from swarms import Agent, ChromaDB, OpenAIChat

# Making an instance of the ChromaDB class
memory = ChromaDB(
    metric="cosine",
    n_results=3,
    output_dir="results",
    docs_folder="docs",
)

# Initialize a tool
def search_api(query: str):
    # Add your logic here
    return query

# Initializing the agent with the Gemini instance and other parameters
agent = Agent(
    agent_name="Covid-19-Chat",
    agent_description=(
        "This agent provides information about COVID-19 symptoms."
    ),
    llm=OpenAIChat(),
    max_loops="auto",
    autosave=True,
    verbose=True,
    long_term_memory=memory,
    stopping_condition="finish",
    tools=[search_api],
)

# Defining the task and image path
task = ("What are the symptoms of COVID-19?",)

# Running the agent with the specified task and image
out = agent.run(task)
print(out)

```

### Simple Conversational Agent
A Plug in and play conversational agent with `GPT4`, `Mixytral`, or any of our models

- Reliable conversational structure to hold messages together with dynamic handling for long context conversations and interactions with auto chunking
- Reliable, this simple system will always provide responses you want.

```python
from swarms import Agent, Anthropic


## Initialize the workflow
agent = Agent(
    agent_name="Transcript Generator",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms"
        " are!"
    ),
    llm=Anthropic(),
    max_loops=3,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True, # Set to True
)

# Run the workflow on a task
agent("Generate a transcript for a youtube video on what swarms are!")
```

## Devin
Implementation of Devin in less than 90 lines of code with several tools:
terminal, browser, and edit files!

```python
from swarms import Agent, Anthropic
import subprocess

# Model
llm = Anthropic(
    temperature=0.1,
)

# Tools
def terminal(
    code: str,
):
    """
    Run code in the terminal.

    Args:
        code (str): The code to run in the terminal.

    Returns:
        str: The output of the code.
    """
    out = subprocess.run(
        code, shell=True, capture_output=True, text=True
    ).stdout
    return str(out)

def browser(query: str):
    """
    Search the query in the browser with the `browser` tool.

    Args:
        query (str): The query to search in the browser.

    Returns:
        str: The search results.
    """
    import webbrowser

    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"Searching for {query} in the browser."

def create_file(file_path: str, content: str):
    """
    Create a file using the file editor tool.

    Args:
        file_path (str): The path to the file.
        content (str): The content to write to the file.

    Returns:
        str: The result of the file creation operation.
    """
    with open(file_path, "w") as file:
        file.write(content)
    return f"File {file_path} created successfully."

def file_editor(file_path: str, mode: str, content: str):
    """
    Edit a file using the file editor tool.

    Args:
        file_path (str): The path to the file.
        mode (str): The mode to open the file in.
        content (str): The content to write to the file.

    Returns:
        str: The result of the file editing operation.
    """
    with open(file_path, mode) as file:
        file.write(content)
    return f"File {file_path} edited successfully."


# Agent
agent = Agent(
    agent_name="Devin",
    system_prompt=(
        "Autonomous agent that can interact with humans and other"
        " agents. Be Helpful and Kind. Use the tools provided to"
        " assist the user. Return all code in markdown format."
    ),
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    tools=[terminal, browser, file_editor, create_file],
    code_interpreter=True,
    # streaming=True,
)

# Run the agent
out = agent("Create a new file for a plan to take over the world.")
print(out)
```


## `Agent`with Pydantic BaseModel as Output Type
The following is an example of an agent that intakes a pydantic basemodel and outputs it at the same time:

```python
from pydantic import BaseModel, Field
from swarms import Anthropic, Agent


# Initialize the schema for the person's information
class Schema(BaseModel):
    name: str = Field(..., title="Name of the person")
    agent: int = Field(..., title="Age of the person")
    is_student: bool = Field(..., title="Whether the person is a student")
    courses: list[str] = Field(
        ..., title="List of courses the person is taking"
    )


# Convert the schema to a JSON string
tool_schema = Schema(
    name="Tool Name",
    agent=1,
    is_student=True,
    courses=["Course1", "Course2"],
)

# Define the task to generate a person's information
task = "Generate a person's information based on the following schema:"

# Initialize the agent
agent = Agent(
    agent_name="Person Information Generator",
    system_prompt=(
        "Generate a person's information based on the following schema:"
    ),
    # Set the tool schema to the JSON string -- this is the key difference
    tool_schema=tool_schema,
    llm=Anthropic(),
    max_loops=3,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    interactive=True,
    # Set the output type to the tool schema which is a BaseModel
    output_type=tool_schema,  # or dict, or str
    metadata_output_type="json",
    # List of schemas that the agent can handle
    list_tool_schemas=[tool_schema],
    function_calling_format_type="OpenAI",
    function_calling_type="json",  # or soon yaml
)

# Run the agent to generate the person's information
generated_data = agent.run(task)

# Print the generated data
print(f"Generated data: {generated_data}")


```


### `ToolAgent`
ToolAgent is an agent that can use tools through JSON function calling. It intakes any open source model from huggingface and is extremely modular and plug in and play. We need help adding general support to all models soon.


```python
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms import ToolAgent
from swarms.utils.json_utils import base_model_to_json

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-12b",
    load_in_4bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")


# Initialize the schema for the person's information
class Schema(BaseModel):
    name: str = Field(..., title="Name of the person")
    agent: int = Field(..., title="Age of the person")
    is_student: bool = Field(
        ..., title="Whether the person is a student"
    )
    courses: list[str] = Field(
        ..., title="List of courses the person is taking"
    )


# Convert the schema to a JSON string
tool_schema = base_model_to_json(Schema)

# Define the task to generate a person's information
task = (
    "Generate a person's information based on the following schema:"
)

# Create an instance of the ToolAgent class
agent = ToolAgent(
    name="dolly-function-agent",
    description="Ana gent to create a child data",
    model=model,
    tokenizer=tokenizer,
    json_schema=tool_schema,
)

# Run the agent to generate the person's information
generated_data = agent.run(task)

# Print the generated data
print(f"Generated data: {generated_data}")

```







----

### `SequentialWorkflow`
Sequential Workflow enables you to sequentially execute tasks with `Agent` and then pass the output into the next agent and onwards until you have specified your max loops. `SequentialWorkflow` is wonderful for real-world business tasks like sending emails, summarizing documents, and analyzing data.


✅  Save and Restore Workflow states!

✅  Multi-Modal Support for Visual Chaining

✅  Utilizes Agent class

```python
from swarms import Agent, SequentialWorkflow, Anthropic


# Initialize the language model agent (e.g., GPT-3)
llm = Anthropic()

# Initialize agents for individual tasks
agent1 = Agent(
    agent_name="Blog generator",
    system_prompt="Generate a blog post like stephen king",
    llm=llm,
    max_loops=1,
    dashboard=False,
    tools=[],
)
agent2 = Agent(
    agent_name="summarizer",
    system_prompt="Sumamrize the blog post",
    llm=llm,
    max_loops=1,
    dashboard=False,
    tools=[],
)

# Create the Sequential workflow
workflow = SequentialWorkflow(
    agents=[agent1, agent2], max_loops=1, verbose=False
)

# Run the workflow
workflow.run(
    "Generate a blog post on how swarms of agents can help businesses grow."
)

```



### `ConcurrentWorkflow`
`ConcurrentWorkflow` runs all the tasks all at the same time with the inputs you give it!


```python
import os

from dotenv import load_dotenv

from swarms import Agent, ConcurrentWorkflow, OpenAIChat, Task

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

from swarms import Agent, OpenAIChat, RecursiveWorkflow, Task

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



### `SwarmNetwork`
`SwarmNetwork` provides the infrasturcture for building extremely dense and complex multi-agent applications that span across various types of agents.

✅ Efficient Task Management: SwarmNetwork's intelligent agent pool and task queue management system ensures tasks are distributed evenly across agents. This leads to efficient use of resources and faster task completion.

✅ Scalability: SwarmNetwork can dynamically scale the number of agents based on the number of pending tasks. This means it can handle an increase in workload by adding more agents, and conserve resources when the workload is low by reducing the number of agents.

✅ Versatile Deployment Options: With SwarmNetwork, each agent can be run on its own thread, process, container, machine, or even cluster. This provides a high degree of flexibility and allows for deployment that best suits the user's needs and infrastructure.

```python
import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import Agent, OpenAIChat, SwarmNetwork

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
out = swarmnet.run_many_agents("Generate a 10,000 word blog on health and wellness.")
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




## Majority Voting
Multiple-agents will evaluate an idea based off of an parsing or evaluation function. From papers like "[More agents is all you need](https://arxiv.org/pdf/2402.05120.pdf)

```python
from swarms import Agent, MajorityVoting, ChromaDB, Anthropic

# Initialize the llm
llm = Anthropic()

# Agents
agent1 = Agent(
    llm = llm,
    system_prompt="You are the leader of the Progressive Party. What is your stance on healthcare?",
    agent_name="Progressive Leader",
    agent_description="Leader of the Progressive Party",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

agent2 = Agent(
    llm=llm,
    agent_name="Conservative Leader",
    agent_description="Leader of the Conservative Party",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

agent3 = Agent(
    llm=llm,
    agent_name="Libertarian Leader",
    agent_description="Leader of the Libertarian Party",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

# Initialize the majority voting
mv = MajorityVoting(
    agents=[agent1, agent2, agent3],
    output_parser=llm.majority_voting,
    autosave=False,
    verbose=True,
)


# Start the majority voting
mv.run("What is your stance on healthcare?")
```

## Real-World Deployment

### Multi-Agent Swarm for Logistics
Here's a production grade swarm ready for real-world deployment in a factory and logistics settings like warehouses. This swarm can automate 3 costly and inefficient workflows, safety checks, productivity checks, and warehouse security.


```python
import os

from dotenv import load_dotenv

from swarms.models import GPT4VisionAPI
from swarms.prompts.logistics import (
    Efficiency_Agent_Prompt,
    Health_Security_Agent_Prompt,
    Productivity_Agent_Prompt,
    Quality_Control_Agent_Prompt,
    Safety_Agent_Prompt,
    Security_Agent_Prompt,
    Sustainability_Agent_Prompt,
)
from swarms.structs import Agent

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
safety_agent = Agent(llm=llm, sop=Safety_Agent_Prompt, max_loops=1, multi_modal=True)

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
    llm=llm, max_loops="auto", autosave=True, dashboard=True, multi_modal=True
)

# Run the workflow on a task
agent.run(task=task, img=img)
```
----


## Build your own LLMs, Agents, and Swarms!

### Swarms Compliant Model Interface
```python
from swarms import BaseLLM

class vLLMLM(BaseLLM):
    def __init__(self, model_name='default_model', tensor_parallel_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        # Add any additional initialization here
    
    def run(self, task: str):
        pass

# Example
model = vLLMLM("mistral")

# Run the model
out = model("Analyze these financial documents and summarize of them")
print(out)

```


### Swarms Compliant Agent Interface

```python
from swarms import Agent


class MyCustomAgent(Agent):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Custom initialization logic

    def custom_method(self, *args, **kwargs):

        # Implement custom logic here

        pass

    def run(self, task, *args, **kwargs):

        # Customize the run method

        response = super().run(task, *args, **kwargs)

        # Additional custom logic

        return response`

# Model
agent = MyCustomAgent()

# Run the agent
out = agent("Analyze and summarize these financial documents: ")
print(out)

```


### Compliant Interface for Multi-Agent Collaboration

```python
from swarms import AutoSwarm, AutoSwarmRouter, BaseSwarm


# Build your own Swarm
class MySwarm(BaseSwarm):
    def __init__(self, name="kyegomez/myswarm", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def run(self, task: str, *args, **kwargs):
        # Add your multi-agent logic here
        # agent 1
        # agent 2
        # agent 3
        return "output of the swarm"


# Add your custom swarm to the AutoSwarmRouter
router = AutoSwarmRouter(
    swarms=[MySwarm]
)


# Create an AutoSwarm instance
autoswarm = AutoSwarm(
    name="kyegomez/myswarm",
    description="A simple API to build and run swarms",
    verbose=True,
    router=router,
)


# Run the AutoSwarm
autoswarm.run("Analyze these financial data and give me a summary")


```

## `AgentRearrange`
Inspired by Einops and einsum, this orchestration techniques enables you to map out the relationships between various agents. For example you specify linear and sequential relationships like `a -> a1 -> a2 -> a3` or concurrent relationships where the first agent will send a message to 3 agents all at once: `a -> a1, a2, a3`. You can customize your workflow to mix sequential and concurrent relationships. [Docs Available:](https://swarms.apac.ai/en/latest/swarms/structs/agent_rearrange/)

```python
from swarms import Agent, AgentRearrange, rearrange, Anthropic


# Initialize the director agent

director = Agent(
    agent_name="Director",
    system_prompt="Directs the tasks for the workers",
    llm=Anthropic(),
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="director.json",
)


# Initialize worker 1

worker1 = Agent(
    agent_name="Worker1",
    system_prompt="Generates a transcript for a youtube video on what swarms are",
    llm=Anthropic(),
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="worker1.json",
)


# Initialize worker 2
worker2 = Agent(
    agent_name="Worker2",
    system_prompt="Summarizes the transcript generated by Worker1",
    llm=Anthropic(),
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="worker2.json",
)


# Create a list of agents
agents = [director, worker1, worker2]

# Define the flow pattern
flow = "Director -> Worker1 -> Worker2"

# Using AgentRearrange class
agent_system = AgentRearrange(agents=agents, flow=flow)
output = agent_system.run(
    "Create a format to express and communicate swarms of llms in a structured manner for youtube"
)
print(output)


# Using rearrange function
output = rearrange(
    agents,
    flow,
    "Create a format to express and communicate swarms of llms in a structured manner for youtube",
)

print(output)

```

## `HierarhicalSwarm`
Coming soon...


## `AgentLoadBalancer`
Coming soon...


## `GraphSwarm`
Coming soon...


---

## Documentation
Documentation is located here at: [swarms.apac.ai](https://swarms.apac.ai)

----

## Folder Structure
The swarms package has been meticlously crafted for extreme use-ability and understanding, the swarms package is split up into various modules such as `swarms.agents` that holds pre-built agents, `swarms.structs` that holds a vast array of structures like `Agent` and multi agent structures. The 3 most important are `structs`, `models`, and `agents`.

```sh
├── __init__.py
├── agents
├── artifacts
├── memory
├── schemas
├── models
├── prompts
├── structs
├── telemetry
├── tools
├── utils
└── workers
```

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
Accelerate Bugs, Features, and Demos to implement by supporting us here:

<a href="https://polar.sh/kyegomez"><img src="https://polar.sh/embed/fund-our-backlog.svg?org=kyegomez" /></a>


## Docker Instructions
- [Learn More Here About Deployments In Docker](https://swarms.apac.ai/en/latest/docker_setup/)


## Swarm Newsletter 🤖 🤖 🤖 📧 
Sign up to the Swarm newsletter to receive  updates on the latest Autonomous agent research papers, step by step guides on creating multi-agent app, and much more Swarmie goodiness 😊

[CLICK HERE TO SIGNUP](https://docs.google.com/forms/d/e/1FAIpQLSfqxI2ktPR9jkcIwzvHL0VY6tEIuVPd-P2fOWKnd6skT9j1EQ/viewform?usp=sf_link)

# License
Apache License

# Citation
Please cite Swarms in your paper or your project if you found it beneficial in any way! Appreciate you.

```bibtex
@misc{swarms,
  author = {Gomez, Kye},
  title = {{Swarms: The Multi-Agent Collaboration Framework}},
  howpublished = {\url{https://github.com/kyegomez/swarms}},
  year = {2023},
  note = {Accessed: Date}
}
```
