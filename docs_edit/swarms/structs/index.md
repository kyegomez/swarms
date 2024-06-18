# Enterprise-Grade and Production Ready Agents

Swarms is an enterprise grade and production ready multi-agent collaboration framework that enables you to orchestrate many agents to work collaboratively at scale to automate real-world activities.

| **Feature**                  | **Description**                                                                                                                                                       | **Performance Impact** | **Documentation Link**        |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-------------------------------|
| Models                       | Pre-trained models that can be utilized for various tasks within the swarm framework.                                                                                  | ⭐⭐⭐                    | [Documentation](https://docs.swarms.world/en/latest/swarms/models/)            |
| Models APIs                  | APIs to interact with and utilize the models effectively, providing interfaces for inference, training, and fine-tuning.                                               | ⭐⭐⭐                    | [Documentation](https://docs.swarms.world/en/latest/swarms/models/)            |
| Agents with Tools            | Agents equipped with specialized tools to perform specific tasks more efficiently, such as data processing, analysis, or interaction with external systems.            | ⭐⭐⭐⭐                   | [Documentation](https://medium.com/@kyeg/the-swarms-tool-system-functions-pydantic-basemodels-as-tools-and-radical-customization-c2a2e227b8ca)            |
| Agents with Memory                       | Mechanisms for agents to store and recall past interactions, improving learning and adaptability over time.                                                            | ⭐⭐⭐⭐                   | [Documentation](https://github.com/kyegomez/swarms/blob/master/playground/structs/agent/agent_with_longterm_memory.py)            |
| Multi-Agent Orchestration    | Coordination of multiple agents to work together seamlessly on complex tasks, leveraging their individual strengths to achieve higher overall performance.              | ⭐⭐⭐⭐⭐                  | [Documentation]()            |

The performance impact is rated on a scale from one to five stars, with multi-agent orchestration being the highest due to its ability to combine the strengths of multiple agents and optimize task execution.

----

## Install 💻
`$ pip3 install -U swarms`

---

# Usage Examples 🤖

### Google Colab Example
Run example in Colab: <a target="_blank" href="https://colab.research.google.com/github/kyegomez/swarms/blob/master/playground/swarms_example.ipynb">
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


### `Agent` + Long Term Memory
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


### `Agent` ++ Long Term Memory ++ Tools!
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


### Devin
Implementation of Devin in less than 90 lines of code with several tools:
terminal, browser, and edit files.

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


### `Agent`with Pydantic BaseModel as Output Type
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

### Multi Modal Autonomous Agent
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

