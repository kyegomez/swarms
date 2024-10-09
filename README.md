<div align="center">
  <a href="https://swarms.world">
    <img src="https://github.com/kyegomez/swarms/blob/master/images/swarmslogobanner.png" style="margin: 15px; max-width: 300px" width="50%" alt="Logo">
  </a>
</div>
<p align="center">
  <em>The Enterprise-Grade Production-Ready Multi-Agent Orchestration Framework </em>
</p>

<p align="center">
    <a href="https://pypi.org/project/swarms/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/swarms?style=for-the-badge&color=3670A0">
    </a>
</p>
<p align="center">
<a href="https://twitter.com/swarms_corp/">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://discord.gg/agora-999382051935506503">📢 Discord</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://swarms.world">Swarms Platform</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://docs.swarms.world">📙 Documentation</a>
</p>




[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)



[![GitHub issues](https://img.shields.io/github/issues/kyegomez/swarms)](https://github.com/kyegomez/swarms/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/swarms)](https://github.com/kyegomez/swarms/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/swarms)](https://github.com/kyegomez/swarms/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/swarms)](https://github.com/kyegomez/swarms/blob/main/LICENSE)[![GitHub star chart](https://img.shields.io/github/stars/kyegomez/swarms?style=social)](https://star-history.com/#kyegomez/swarms)[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/swarms)](https://libraries.io/github/kyegomez/swarms) [![Downloads](https://static.pepy.tech/badge/swarms/month)](https://pepy.tech/project/swarms)

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarms)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=Swarms%20-%20the%20future%20of%20AI) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&t=Swarms%20-%20the%20future%20of%20AI) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Swarms%20-%20the%20future%20of%20AI) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Swarms%20-%20the%20future%20of%20AI%20%23swarms%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)



Swarms is an enterprise grade and production ready multi-agent collaboration framework that enables you to orchestrate many agents to work collaboratively at scale to automate real-world activities.

----

## Requirements
- `python3.10` or above!
- `$ pip install -U swarms` And, don't forget to install swarms!
- `.env` file with API keys from your providers like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
-  Set an `.env` Variable with your desired workspace dir: `WORKSPACE_DIR="agent_workspace"` or do it in your terminal with `export WORKSPACE_DIR="agent_workspace"`

## Install 💻

```bash
$ pip3 install -U swarms
```

---

# Usage Examples 🤖
Here are some simple examples but we have more comprehensive documentation at our [docs here](https://docs.swarms.world/en/latest/)

Run example in Collab: <a target="_blank" href="https://colab.research.google.com/github/kyegomez/swarms/blob/master/examples/collabs/swarms_example.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## `Agents`
A fully plug-and-play autonomous agent powered by an LLM extended by a long-term memory database, and equipped with function calling for tool usage! By passing in an LLM, you can create a fully autonomous agent with extreme customization and reliability, ready for real-world task automation!

Features:

✅ Any LLM / Any framework

✅ Extremely customize-able with max loops, autosaving, import docs (PDFS, TXT, CSVs, etc), tool usage, etc etc

✅ Long term memory database with RAG (ChromaDB, Pinecone, Qdrant)

```python
import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from dotenv import load_dotenv

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
    # output_type="json",
)


out = agent.run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria"
)
print(out)

```

-----

### Agent with RAG
`Agent` equipped with quasi-infinite long term memory. Great for long document understanding, analysis, and retrieval.

```python
import os

from swarms_memory import ChromaDB

from swarms import Agent
from swarm_models import Anthropic
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initilaize the chromadb client
chromadb = ChromaDB(
    metric="cosine",
    output_dir="fiance_agent_rag",
    # docs_folder="artifacts", # Folder of your documents
)

# Model
model = Anthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    agent_description="Agent creates ",
    llm=model,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=200000,
    long_term_memory=chromadb,
)


agent.run(
    "What are the components of a startups stock incentive equity plan"
)


```

-------

### `Agent` ++ Long Term Memory ++ Tools!
An LLM equipped with long term memory and tools, a full stack agent capable of automating all and any digital tasks given a good prompt.

```python
from swarms import Agent
from swarm_models import OpenAIChat
from swarms_memory import ChromaDB
import subprocess
import os

# Making an instance of the ChromaDB class
memory = ChromaDB(
    metric="cosine",
    n_results=3,
    output_dir="results",
    docs_folder="docs",
)

# Model
model = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.1,
)


# Tools in swarms are simple python functions and docstrings
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
    llm=model,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    tools=[terminal, browser, file_editor, create_file],
    streaming=True,
    long_term_memory=memory,
)

# Run the agent
out = agent(
    "Create a CSV file with the latest tax rates for C corporations in the following ten states and the District of Columbia: Alabama, California, Florida, Georgia, Illinois, New York, North Carolina, Ohio, Texas, and Washington."
)
print(out)

```

-------

### Misc Agent Settings
We provide vast array of features to save agent states using json, yaml, toml, upload pdfs, batched jobs, and much more!

```python
# # Convert the agent object to a dictionary
print(agent.to_dict())
print(agent.to_toml())
print(agent.model_dump_json())
print(agent.model_dump_yaml())

# Ingest documents into the agent's knowledge base
agent.ingest_docs("your_pdf_path.pdf")

# Receive a message from a user and process it
agent.receive_message(name="agent_name", message="message")

# Send a message from the agent to a user
agent.send_agent_message(agent_name="agent_name", message="message")

# Ingest multiple documents into the agent's knowledge base
agent.ingest_docs("your_pdf_path.pdf", "your_csv_path.csv")

# Run the agent with a filtered system prompt
agent.filtered_run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"
)

# Run the agent with multiple system prompts
agent.bulk_run(
    [
        "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?",
        "Another system prompt",
    ]
)

# Add a memory to the agent
agent.add_memory("Add a memory to the agent")

# Check the number of available tokens for the agent
agent.check_available_tokens()

# Perform token checks for the agent
agent.tokens_checks()

# Print the dashboard of the agent
agent.print_dashboard()

# Fetch all the documents from the doc folders
agent.get_docs_from_doc_folders()

# Activate agent ops
agent.activate_agentops()
agent.check_end_session_agentops()

# Dump the model to a JSON file
agent.model_dump_json()
print(agent.to_toml())

```



### `Agent`with Pydantic BaseModel as Output Type
The following is an example of an agent that intakes a pydantic basemodel and outputs it at the same time:

```python
from pydantic import BaseModel, Field
from swarms import Agent
from swarm_models import Anthropic


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
    list_base_models=[tool_schema],
    function_calling_format_type="OpenAI",
    function_calling_type="json",  # or soon yaml
)

# Run the agent to generate the person's information
generated_data = agent.run(task)

# Print the generated data
print(f"Generated data: {generated_data}")


```

### Multi Modal Autonomous Agent
Run the agent with multiple modalities useful for various real-world tasks in manufacturing, logistics, and health.

```python
import os
from dotenv import load_dotenv
from swarms import Agent

from swarm_models import GPT4VisionAPI

# Load the environment variables
load_dotenv()


# Initialize the language model
llm = GPT4VisionAPI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
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
    agent_name = "Multi-ModalAgent",
    llm=llm, 
    max_loops="auto", 
    autosave=True, 
    dashboard=True, 
    multi_modal=True
)

# Run the workflow on a task
agent.run(task, img)
```
----


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


## Integrating External Agents
Integrating external agents from other agent frameworks is easy with swarms.

Steps:

1. Create a new class that inherits `Agent`
2. Create a `.run(task: str) -> str` method that runs the agent and returns the response. 
3. The new Agent must return a string of the response. But you may add additional methods to save the output to JSON.


### Griptape Example

For example, here's an example on how to create an agent from griptape.

Here’s how you can create a custom **Griptape** agent that integrates with the **Swarms** framework by inheriting from the `Agent` class in **Swarms** and overriding the `run(task: str) -> str` method.


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
    def __init__(self, *args, **kwargs):
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
            # Add additional settings
        )

    # Override the run method to take a task and execute it using the Griptape agent
    def run(self, task: str) -> str:
        # Extract URL and filename from task (you can modify this parsing based on task structure)
        url, filename = task.split(
            ","
        )  # Example of splitting task string
        # Execute the Griptape agent with the task inputs
        result = self.agent.run(url.strip(), filename.strip())
        # Return the final result as a string
        return str(result)


# Example usage:
griptape_swarms_agent = GriptapeSwarmsAgent()
output = griptape_swarms_agent.run(
    "https://griptape.ai, griptape.txt"
)
print(output)
```

### Key Components:
1. **GriptapeSwarmsAgent**: A custom class that inherits from the `SwarmsAgent` class and integrates the Griptape agent.
2. **run(task: str) -> str**: A method that takes a task string, processes it (e.g., splitting into a URL and filename), and runs the Griptape agent with the provided inputs.
3. **Griptape Tools**: The tools integrated into the Griptape agent (e.g., `WebScraperTool`, `PromptSummaryTool`, `FileManagerTool`) allow for web scraping, summarization, and file management.

You can now easily plug this custom Griptape agent into the **Swarms Framework** and use it to run tasks!






# Multi-Agent Orchestration:
Swarms was designed to facilitate the communication between many different and specialized agents from a vast array of other frameworks such as langchain, autogen, crew, and more.

In traditional swarm theory, there are many types of swarms usually for very specialized use-cases and problem sets. Such as Hiearchical and sequential are great for accounting and sales, because there is usually a boss coordinator agent that distributes a workload to other specialized agents.


| **Name**                      | **Description**                                                                                                                                                         | **Code Link**               | **Use Cases**                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|---------------------------------------------------------------------------------------------------|
| Hierarchical Swarms           | A system where agents are organized in a hierarchy, with higher-level agents coordinating lower-level agents to achieve complex tasks.                                   | [Code Link](#)              | Manufacturing process optimization, multi-level sales management, healthcare resource coordination |
| Agent Rearrange               | A setup where agents rearrange themselves dynamically based on the task requirements and environmental conditions.                                                       | [Code Link](https://docs.swarms.world/en/latest/swarms/structs/agent_rearrange/)              | Adaptive manufacturing lines, dynamic sales territory realignment, flexible healthcare staffing  |
| Concurrent Workflows          | Agents perform different tasks simultaneously, coordinating to complete a larger goal.                                                                                  | [Code Link](#)              | Concurrent production lines, parallel sales operations, simultaneous patient care processes       |
| Sequential Coordination       | Agents perform tasks in a specific sequence, where the completion of one task triggers the start of the next.                                                           | [Code Link](https://docs.swarms.world/en/latest/swarms/structs/sequential_workflow/)              | Step-by-step assembly lines, sequential sales processes, stepwise patient treatment workflows     |
| Parallel Processing           | Agents work on different parts of a task simultaneously to speed up the overall process.                                                                                | [Code Link](#)              | Parallel data processing in manufacturing, simultaneous sales analytics, concurrent medical tests  |



### `SequentialWorkflow`
Sequential Workflow enables you to sequentially execute tasks with `Agent` and then pass the output into the next agent and onwards until you have specified your max loops.

```python
from swarms import Agent, SequentialWorkflow

from swarm_models import Anthropic


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

------

## `AgentRearrange`
Inspired by Einops and einsum, this orchestration techniques enables you to map out the relationships between various agents. For example you specify linear and sequential relationships like `a -> a1 -> a2 -> a3` or concurrent relationships where the first agent will send a message to 3 agents all at once: `a -> a1, a2, a3`. You can customize your workflow to mix sequential and concurrent relationships. [Docs Available:](https://docs.swarms.world/en/latest/swarms/structs/agent_rearrange/)

```python

from swarms import Agent, AgentRearrange


from swarm_models import Anthropic

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

```

## `HierarhicalSwarm`
Coming soon...


## `GraphSwarm`

```python
import os

from dotenv import load_dotenv


from swarms import Agent, Edge, GraphWorkflow, Node, NodeType

from swarm_models import OpenAIChat

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAIChat(
    temperature=0.5, openai_api_key=api_key, max_tokens=4000
)
agent1 = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)
agent2 = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)

def sample_task():
    print("Running sample task")
    return "Task completed"

wf_graph = GraphWorkflow()
wf_graph.add_node(Node(id="agent1", type=NodeType.AGENT, agent=agent1))
wf_graph.add_node(Node(id="agent2", type=NodeType.AGENT, agent=agent2))
wf_graph.add_node(
    Node(id="task1", type=NodeType.TASK, callable=sample_task)
)
wf_graph.add_edge(Edge(source="agent1", target="task1"))
wf_graph.add_edge(Edge(source="agent2", target="task1"))

wf_graph.set_entry_points(["agent1", "agent2"])
wf_graph.set_end_points(["task1"])

print(wf_graph.visualize())

# Run the workflow
results = wf_graph.run()
print("Execution results:", results)

```

## `MixtureOfAgents`
This is an implementation from the paper: "Mixture-of-Agents Enhances Large Language Model Capabilities" by together.ai, it achieves SOTA on AlpacaEval 2.0, MT-Bench and FLASK, surpassing GPT-4 Omni. Great for tasks that need to be parallelized and then sequentially fed into another loop

```python
from swarms import Agent, OpenAIChat, MixtureOfAgents

# Initialize the director agent
director = Agent(
    agent_name="Director",
    system_prompt="Directs the tasks for the accountants",
    llm=OpenAIChat(),
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="director.json",
)

# Initialize accountant 1
accountant1 = Agent(
    agent_name="Accountant1",
    system_prompt="Prepares financial statements",
    llm=OpenAIChat(),
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="accountant1.json",
)

# Initialize accountant 2
accountant2 = Agent(
    agent_name="Accountant2",
    system_prompt="Audits financial records",
    llm=OpenAIChat(),
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="accountant2.json",
)

# Create a list of agents
agents = [director, accountant1, accountant2]


# Swarm
swarm = MixtureOfAgents(
    name="Mixture of Accountants",
    agents=agents,
    layers=3,
    final_agent=director,
)


# Run the swarm
out = swarm.run("Prepare financial statements and audit financial records")
print(out)
```


## SpreadSheetSwarm
An all-new swarm architecture that makes it easy to manage and oversee the outputs of thousands of agents all at once! 

[Learn more at the docs here:](https://docs.swarms.world/en/latest/swarms/structs/spreadsheet_swarm/)

```python
import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Define custom system prompts for each social media platform
TWITTER_AGENT_SYS_PROMPT = """
You are a Twitter marketing expert specializing in real estate. Your task is to create engaging, concise tweets to promote properties, analyze trends to maximize engagement, and use appropriate hashtags and timing to reach potential buyers.
"""

INSTAGRAM_AGENT_SYS_PROMPT = """
You are an Instagram marketing expert focusing on real estate. Your task is to create visually appealing posts with engaging captions and hashtags to showcase properties, targeting specific demographics interested in real estate.
"""

FACEBOOK_AGENT_SYS_PROMPT = """
You are a Facebook marketing expert for real estate. Your task is to craft posts optimized for engagement and reach on Facebook, including using images, links, and targeted messaging to attract potential property buyers.
"""

LINKEDIN_AGENT_SYS_PROMPT = """
You are a LinkedIn marketing expert for the real estate industry. Your task is to create professional and informative posts, highlighting property features, market trends, and investment opportunities, tailored to professionals and investors.
"""

EMAIL_AGENT_SYS_PROMPT = """
You are an Email marketing expert specializing in real estate. Your task is to write compelling email campaigns to promote properties, focusing on personalization, subject lines, and effective call-to-action strategies to drive conversions.
"""

# Example usage:
api_key = os.getenv("OPENAI_API_KEY")

# Model
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize your agents for different social media platforms
agents = [
    Agent(
        agent_name="Twitter-RealEstate-Agent",
        system_prompt=TWITTER_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="twitter_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Instagram-RealEstate-Agent",
        system_prompt=INSTAGRAM_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="instagram_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Facebook-RealEstate-Agent",
        system_prompt=FACEBOOK_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="facebook_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="LinkedIn-RealEstate-Agent",
        system_prompt=LINKEDIN_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="linkedin_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Email-RealEstate-Agent",
        system_prompt=EMAIL_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="email_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
]

# Create a Swarm with the list of agents
swarm = SpreadSheetSwarm(
    name="Real-Estate-Marketing-Swarm",
    description="A swarm that processes real estate marketing tasks using multiple agents on different threads.",
    agents=agents,
    autosave_on=True,
    save_file_path="real_estate_marketing_spreadsheet.csv",
    run_all_agents=False,
    max_loops=2,
)

# Run the swarm
swarm.run(
    task="""
    Create posts to promote luxury properties in North Texas, highlighting their features, location, and investment potential. Include relevant hashtags, images, and engaging captions.

    
    Property:
    $10,399,000
    1609 Meandering Way Dr, Roanoke, TX 76262
    Link to the property: https://www.zillow.com/homedetails/1609-Meandering-Way-Dr-Roanoke-TX-76262/308879785_zpid/
    
    What's special
    Unveiling a new custom estate in the prestigious gated Quail Hollow Estates! This impeccable residence, set on a sprawling acre surrounded by majestic trees, features a gourmet kitchen equipped with top-tier Subzero and Wolf appliances. European soft-close cabinets and drawers, paired with a double Cambria Quartzite island, perfect for family gatherings. The first-floor game room&media room add extra layers of entertainment. Step into the outdoor sanctuary, where a sparkling pool and spa, and sunken fire pit, beckon leisure. The lavish master suite features stunning marble accents, custom his&her closets, and a secure storm shelter.Throughout the home,indulge in the visual charm of designer lighting and wallpaper, elevating every space. The property is complete with a 6-car garage and a sports court, catering to the preferences of basketball or pickleball enthusiasts. This residence seamlessly combines luxury&recreational amenities, making it a must-see for the discerning buyer.
    
    Facts & features
    Interior
    Bedrooms & bathrooms
    Bedrooms: 6
    Bathrooms: 8
    Full bathrooms: 7
    1/2 bathrooms: 1
    Primary bedroom
    Bedroom
    Features: Built-in Features, En Suite Bathroom, Walk-In Closet(s)
    Cooling
    Central Air, Ceiling Fan(s), Electric
    Appliances
    Included: Built-In Gas Range, Built-In Refrigerator, Double Oven, Dishwasher, Gas Cooktop, Disposal, Ice Maker, Microwave, Range, Refrigerator, Some Commercial Grade, Vented Exhaust Fan, Warming Drawer, Wine Cooler
    Features
    Wet Bar, Built-in Features, Dry Bar, Decorative/Designer Lighting Fixtures, Eat-in Kitchen, Elevator, High Speed Internet, Kitchen Island, Pantry, Smart Home, Cable TV, Walk-In Closet(s), Wired for Sound
    Flooring: Hardwood
    Has basement: No
    Number of fireplaces: 3
    Fireplace features: Living Room, Primary Bedroom
    Interior area
    Total interior livable area: 10,466 sqft
    Total spaces: 12
    Parking features: Additional Parking
    Attached garage spaces: 6
    Carport spaces: 6
    Features
    Levels: Two
    Stories: 2
    Patio & porch: Covered
    Exterior features: Built-in Barbecue, Barbecue, Gas Grill, Lighting, Outdoor Grill, Outdoor Living Area, Private Yard, Sport Court, Fire Pit
    Pool features: Heated, In Ground, Pool, Pool/Spa Combo
    Fencing: Wrought Iron
    Lot
    Size: 1.05 Acres
    Details
    Additional structures: Outdoor Kitchen
    Parcel number: 42232692
    Special conditions: Standard
    Construction
    Type & style
    Home type: SingleFamily
    Architectural style: Contemporary/Modern,Detached
    Property subtype: Single Family Residence
    """
)

```


## `ForestSwarm`
The architecture allows for efficient task assignment by selecting the most relevant agent from a set of trees. Tasks are processed asynchronously, with agents selected based on task relevance, calculated by the similarity of system prompts and task keywords. [Learn More with the documentation](https://docs.swarms.world/en/latest/swarms/structs/forest_swarm/)


```python
from swarms.structs.tree_swarm import TreeAgent, Tree, ForestSwarm
# Example Usage:

# Create agents with varying system prompts and dynamically generated distances/keywords
agents_tree1 = [
    TreeAgent(
        system_prompt="Stock Analysis Agent",
        agent_name="Stock Analysis Agent",
    ),
    TreeAgent(
        system_prompt="Financial Planning Agent",
        agent_name="Financial Planning Agent",
    ),
    TreeAgent(
        agent_name="Retirement Strategy Agent",
        system_prompt="Retirement Strategy Agent",
    ),
]

agents_tree2 = [
    TreeAgent(
        system_prompt="Tax Filing Agent",
        agent_name="Tax Filing Agent",
    ),
    TreeAgent(
        system_prompt="Investment Strategy Agent",
        agent_name="Investment Strategy Agent",
    ),
    TreeAgent(
        system_prompt="ROTH IRA Agent", agent_name="ROTH IRA Agent"
    ),
]

# Create trees
tree1 = Tree(tree_name="Financial Tree", agents=agents_tree1)
tree2 = Tree(tree_name="Investment Tree", agents=agents_tree2)

# Create the ForestSwarm
multi_agent_structure = ForestSwarm(trees=[tree1, tree2])

# Run a task
task = "Our company is incorporated in delaware, how do we do our taxes for free?"
output = multi_agent_structure.run(task)
print(output)
```

----------

## Onboarding Session
Get onboarded now with the creator and lead maintainer of Swarms, Kye Gomez, who will show you how to get started with the installation, usage examples, and starting to build your custom use case! [CLICK HERE](https://cal.com/swarms/swarms-onboarding-session)


---

## Documentation
Documentation is located here at: [docs.swarms.world](https://docs.swarms.world)

-----

## Folder Structure
The swarms package has been meticlously crafted for extreme use-ability and understanding, the swarms package is split up into various modules such as `swarms.agents` that holds pre-built agents, `swarms.structs` that holds a vast array of structures like `Agent` and multi agent structures. The 3 most important are `structs`, `models`, and `agents`.

```sh
├── __init__.py
├── agents
├── artifacts
├── memory
├── schemas
├── models -> swarm_models
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

----



## Accelerate Backlog
Accelerate Bugs, Features, and Demos to implement by supporting us here:

<a href="https://polar.sh/kyegomez"><img src="https://polar.sh/embed/fund-our-backlog.svg?org=kyegomez" /></a>

## Community

Join our growing community around the world, for real-time support, ideas, and discussions on Swarms 😊 

- View our official [Blog](https://docs.swarms.world)
- Chat live with us on [Discord](https://discord.gg/kS3rwKs3ZC)
- Follow us on [Twitter](https://twitter.com/kyegomez)
- Connect with us on [LinkedIn](https://www.linkedin.com/company/the-swarm-corporation)
- Visit us on [YouTube](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ)
- [Join the Swarms community on Discord!](https://discord.gg/AJazBmhKnr)
- Join our Swarms Community Gathering every Thursday at 1pm NYC Time to unlock the potential of autonomous agents in automating your daily tasks [Sign up here](https://lu.ma/5p2jnc2v)

# License

Creative Commons Attribution 4.0 International Public License