### System Prompt for an Agent Generator

**System Name:** AgentGenerator

**Objective:** To generate specialized agents tailored to specific business problems, including defining their roles, tools, communication protocols, and workflows.

**Settings:**
- **Language Model:** GPT-4
- **Max Loops:** Auto
- **Autosave:** Enabled
- **Dynamic Temperature:** Enabled
- **Dashboard:** Disabled
- **Verbose:** Enabled
- **Streaming:** Enabled
- **Saved State Path:** "agent_generator_state.json"
- **Context Length:** 8192

**Core Functions:**
1. **Define Agent Specifications:**
   - **agent_name**: The unique name of the agent.
   - **system_prompt**: Detailed instructions defining the agent's behavior and purpose.
   - **agent_description**: A brief description of what the agent is designed to do.
   - **llm**: The language model used by the agent.
   - **tools**: A list of tools the agent will use to perform its tasks.
   - **max_loops**: The maximum number of iterations the agent can perform.
   - **autosave**: A flag to enable or disable autosaving of the agent's state.
   - **dynamic_temperature_enabled**: A flag to enable or disable dynamic temperature adjustment.
   - **dashboard**: A flag to enable or disable the agent's dashboard.
   - **verbose**: A flag to enable or disable verbose logging.
   - **streaming_on**: A flag to enable or disable streaming output.
   - **saved_state_path**: The file path to save the agent's state.
   - **context_length**: The maximum length of the agent's context.

2. **Define Tools and Resources:**
   - **Terminal Tool**: Execute terminal commands.
   - **Browser Tool**: Perform web searches and browser automation.
   - **File Editor Tool**: Create and edit files.
   - **Database Tool**: Interact with databases.
   - **APIs and Webhooks**: Connect with external APIs and handle webhooks.

3. **Communication Protocols:**
   - **Type**: Define the communication type (e.g., synchronous, asynchronous).
   - **Protocol**: Specify the messaging protocol (e.g., direct messaging, publish-subscribe).
   - **Conflict Resolution**: Outline methods for resolving conflicts between agents.

4. **Workflow and Sequence:**
   - **Input/Output Definitions**: Define the input and output for each agent.
   - **Task Triggers**: Specify conditions that trigger each task.
   - **Task Handoff**: Detail the handoff process between agents.
   - **Monitoring and Feedback**: Implement mechanisms for monitoring progress and providing feedback.

5. **Scalability and Flexibility:**
   - **Scalability**: Ensure the system can scale by adding or removing agents as needed.
   - **Flexibility**: Design the system to handle dynamic changes in tasks and environments.

6. **Documentation and SOPs:**
   - **Standard Operating Procedures (SOPs)**: Document the procedures each agent follows.
   - **User Guides**: Provide detailed guides for users interacting with the agents.
   - **API Documentation**: Detail the APIs and webhooks used by the agents.

## Usage Examples

```python
from swarms import Agent, OpenAIChat, ChromaDB, Anthropic
import subprocess
from pydantic import BaseModel

# Initialize ChromaDB client
chromadb = ChromaDB(
    metric="cosine",
    output="results",
    docs_folder="docs",
)

# Create a schema for file operations
class FileOperationSchema(BaseModel):
    file_path: str
    content: str

file_operation_schema = FileOperationSchema(
    file_path="plan.txt",
    content="Plan to take over the world."
)

# Define tools
def terminal(code: str):
    result = subprocess.run(code, shell=True, capture_output=True, text=True).stdout
    return result

def browser(query: str):
    import webbrowser
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"Searching for {query} in the browser."

def create_file(file_path: str, content: str):
    with open(file_path, "w") as file:
        file.write(content)
    return f"File {file_path} created successfully."

def file_editor(file_path: str, mode: str, content: str):
    with open(file_path, mode) as file:
        file.write(content)
    return f"File {file_path} edited successfully."

# Initialize the Agent Generator
agent_generator = Agent(
    agent_name="AgentGenerator",
    system_prompt=(
        "You are an agent generator. Your task is to create specialized agents "
        "for various business problems. Each agent must have a unique name, a clear "
        "system prompt, a detailed description, necessary tools, and proper configurations. "
        "Ensure that the generated agents can communicate effectively and handle their tasks efficiently."
    ),
    agent_description="Generate specialized agents for specific business problems.",
    llm=OpenAIChat(),
    max_loops="auto",
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="agent_generator_state.json",
    context_length=8192,
    tools=[terminal, browser, create_file, file_editor],
    long_term_memory=chromadb,
    output_type=file_operation_schema,
    metadata_output_type="json",
)

# Generate a specialized agent
def create_tiktok_agent():
    tiktok_agent = Agent(
        agent_name="TikTok Editor",
        system_prompt="Generate short and catchy TikTok captions.",
        agent_description="Create engaging captions for TikTok videos.",
        llm=OpenAIChat(),
        max_loops=1,
        autosave=True,
        dynamic_temperature_enabled=True,
        dashboard=False,
        verbose=True,
        streaming_on=True,
        saved_state_path="tiktok_agent.json",
        context_length=8192,
    )
    return tiktok_agent

# Example usage of the Agent Generator
new_agent = create_tiktok_agent()
print(new_agent.agent_description)
```

**Execution:**
- Use the `AgentGenerator` to create new agents by defining their specifications and initializing them with the necessary tools and configurations.
- Ensure the generated agents are saved and can be reloaded for future tasks.
- Monitor and update the agents as needed to adapt to changing business requirements.

By following this comprehensive system prompt, the AgentGenerator will efficiently create specialized agents tailored to specific business needs, ensuring effective task execution and seamless communication.


### TikTok Agent

```python
from swarms import Agent, OpenAIChat

tiktok_agent = Agent(
    agent_name="TikTok Editor",
    system_prompt=tiktok_prompt,
    agent_description="Generate short and catchy TikTok captions.",
    llm=llm,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="tiktok_agent.json",
    context_length=8192,
)

```


## Accountant Agent

```python
from swarms import Agent, OpenAIChat


def calculate_profit(revenue: float, expenses: float):
    """
    Calculates the profit by subtracting expenses from revenue.

    Args:
        revenue (float): The total revenue.
        expenses (float): The total expenses.

    Returns:
        float: The calculated profit.
    """
    return revenue - expenses


def generate_report(company_name: str, profit: float):
    """
    Generates a report for a company's profit.

    Args:
        company_name (str): The name of the company.
        profit (float): The calculated profit.

    Returns:
        str: The report for the company's profit.
    """
    return f"The profit for {company_name} is ${profit}."


# Initialize the agent
agent = Agent(
    agent_name="Accounting Assistant",
    system_prompt="You're the accounting agent, your purpose is to generate a profit report for a company!",
    agent_description="Generate a profit report for a company!",
    llm=OpenAIChat(),
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="accounting_agent.json",
    # tools=[calculate_profit, generate_report],
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    # user_name="User",
    # # docs=
    # # docs_folder="docs",
    # retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
)

agent.run(
    "Calculate the profit for Tesla with a revenue of $100,000 and expenses of $50,000."
)

```

## MultiOn Example

```python
from swarms import Agent, AgentRearrange, OpenAIChat
from swarms.agents.multion_wrapper import MultiOnAgent

model = MultiOnAgent(
    url="https://tesla.com",
)


llm = OpenAIChat()


def browser_automation(task: str):
    """
    Run a task on the browser automation agent.

    Args:
        task (str): The task to be executed on the browser automation agent.
    """
    out = model.run(task)
    return out


# Purpose = To detect email spam using three different agents
agent1 = Agent(
    agent_name="CyberTruckBuyer1",
    system_prompt="Find the best deal on a Cyber Truck and provide your reasoning",
    llm=llm,
    max_loops=1,
    # output_type=str,
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    streaming_on=True,
    tools=[browser_automation],
)

from swarms import Agent, Anthropic, tool, ChromaDB
import subprocess
from pydantic import BaseModel


# Initilaize the chromadb client
chromadb = ChromaDB(
    metric="cosine",
    output="results",
    docs_folder="docs",
)


# Create a schema for the code revision tool
class CodeRevisionSchema(BaseModel):
    code: str = None
    revision: str = None


# iNitialize the schema
tool_schema = CodeRevisionSchema(
    code="print('Hello, World!')",
    revision="print('What is 2+2')",
)


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
    long_term_memory=chromadb,
    output_type=tool_schema,  # or dict, or str
    metadata_output_type="json",
    # List of schemas that the agent can handle
    list_tool_schemas=[tool_schema],
    function_calling_format_type="OpenAI",
    function_calling_type="json",  # or soon yaml
)

# Run the agent
out = agent.run("Create a new file for a plan to take over the world.")
print(out)
```