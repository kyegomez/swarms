# SwarmRouter Documentation

The `SwarmRouter` class is a flexible routing system designed to manage different types of swarms for task execution. It provides a unified interface to interact with various swarm types, including:

| Swarm Type | Description |
|------------|-------------|
| `AgentRearrange` | Optimizes agent arrangement for task execution |
| `MixtureOfAgents` | Combines multiple agent types for diverse tasks |
| `SpreadSheetSwarm` | Uses spreadsheet-like operations for task management |
| `SequentialWorkflow` | Executes tasks sequentially |
| `ConcurrentWorkflow` | Executes tasks in parallel |
| `GroupChat` | Facilitates communication among agents in a group chat format |
| `MultiAgentRouter` | Routes tasks between multiple agents |
| `AutoSwarmBuilder` | Automatically builds swarm structure |
| `HiearchicalSwarm` | Hierarchical organization of agents |
| `MajorityVoting` | Uses majority voting for decision making |
| `MALT` | Multi-Agent Language Tasks |
| `DeepResearchSwarm` | Specialized for deep research tasks |
| `CouncilAsAJudge` | Council-based judgment system |
| `InteractiveGroupChat` | Interactive group chat with user participation |
| `auto` | Automatically selects best swarm type via embedding search |

## Classes

### Document

A Pydantic model for representing document data.

| Attribute | Type | Description |
| --- | --- | --- |
| `file_path` | str | Path to the document file. |
| `data` | str | Content of the document. |

### SwarmLog

A Pydantic model for capturing log entries.

| Attribute | Type | Description |
| --- | --- | --- |
| `id` | str | Unique identifier for the log entry. |
| `timestamp` | datetime | Time of log creation. |
| `level` | str | Log level (e.g., "info", "error"). |
| `message` | str | Log message content. |
| `swarm_type` | SwarmType | Type of swarm associated with the log. |
| `task` | str | Task being performed (optional). |
| `metadata` | Dict[str, Any] | Additional metadata (optional). |
| `documents` | List[Document] | List of documents associated with the log. |

### SwarmRouterConfig

Configuration model for SwarmRouter.

| Attribute | Type | Description |
| --- | --- | --- |
| `name` | str | Name identifier for the SwarmRouter instance |
| `description` | str | Description of the SwarmRouter's purpose |
| `swarm_type` | SwarmType | Type of swarm to use |
| `rearrange_flow` | Optional[str] | Flow configuration string |
| `rules` | Optional[str] | Rules to inject into every agent |
| `multi_agent_collab_prompt` | bool | Whether to enable multi-agent collaboration prompts |
| `task` | str | The task to be executed by the swarm |

### SwarmRouter

Main class for routing tasks to different swarm types.

| Attribute | Type | Description |
| --- | --- | --- |
| `name` | str | Name of the SwarmRouter instance |
| `description` | str | Description of the SwarmRouter's purpose |
| `max_loops` | int | Maximum number of loops to perform |
| `agents` | List[Union[Agent, Callable]] | List of Agent objects or callable functions |
| `swarm_type` | SwarmType | Type of swarm to be used |
| `autosave` | bool | Flag to enable/disable autosave |
| `rearrange_flow` | str | The flow for the AgentRearrange swarm type |
| `return_json` | bool | Flag to enable/disable returning the result in JSON format |
| `auto_generate_prompts` | bool | Flag to enable/disable auto generation of prompts |
| `shared_memory_system` | Any | Shared memory system for agents |
| `rules` | str | Rules to inject into every agent |
| `documents` | List[str] | List of document file paths |
| `output_type` | OutputType | Output format type (e.g., "string", "dict", "list", "json", "yaml", "xml") |
| `no_cluster_ops` | bool | Flag to disable cluster operations |
| `speaker_fn` | callable | Speaker function for GroupChat swarm type |
| `load_agents_from_csv` | bool | Flag to enable/disable loading agents from CSV |
| `csv_file_path` | str | Path to the CSV file for loading agents |
| `return_entire_history` | bool | Flag to enable/disable returning the entire conversation history |
| `multi_agent_collab_prompt` | bool | Whether to enable multi-agent collaboration prompts |

#### Methods:

| Method | Parameters | Description |
| --- | --- | --- |
| `__init__` | `name: str = "swarm-router", description: str = "Routes your task to the desired swarm", max_loops: int = 1, agents: List[Union[Agent, Callable]] = [], swarm_type: SwarmType = "SequentialWorkflow", autosave: bool = False, rearrange_flow: str = None, return_json: bool = False, auto_generate_prompts: bool = False, shared_memory_system: Any = None, rules: str = None, documents: List[str] = [], output_type: OutputType = "dict", no_cluster_ops: bool = False, speaker_fn: callable = None, load_agents_from_csv: bool = False, csv_file_path: str = None, return_entire_history: bool = True, multi_agent_collab_prompt: bool = True` | Initialize the SwarmRouter |
| `setup` | None | Set up the SwarmRouter by activating APE and handling shared memory and rules |
| `activate_shared_memory` | None | Activate shared memory with all agents |
| `handle_rules` | None | Inject rules to every agent |
| `activate_ape` | None | Activate automatic prompt engineering for agents that support it |
| `reliability_check` | None | Perform reliability checks on the SwarmRouter configuration |
| `_create_swarm` | `task: str = None, *args, **kwargs` | Create and return the specified swarm type |
| `update_system_prompt_for_agent_in_swarm` | None | Update system prompts for all agents with collaboration prompts |
| `_log` | `level: str, message: str, task: str = "", metadata: Dict[str, Any] = None` | Create a log entry |
| `_run` | `task: str, img: Optional[str] = None, model_response: Optional[str] = None, *args, **kwargs` | Run the specified task on the selected swarm type |
| `run` | `task: str, img: Optional[str] = None, model_response: Optional[str] = None, *args, **kwargs` | Execute a task on the selected swarm type |
| `__call__` | `task: str, *args, **kwargs` | Make the SwarmRouter instance callable |
| `batch_run` | `tasks: List[str], *args, **kwargs` | Execute multiple tasks in sequence |
| `async_run` | `task: str, *args, **kwargs` | Execute a task asynchronously |
| `get_logs` | None | Retrieve all logged entries |
| `concurrent_run` | `task: str, *args, **kwargs` | Execute a task using concurrent execution |
| `concurrent_batch_run` | `tasks: List[str], *args, **kwargs` | Execute multiple tasks concurrently |


## Installation

To use the SwarmRouter, first install the required dependencies:

```bash
pip install swarms swarm_models
```

## Basic Usage

```python
import os
from dotenv import load_dotenv
from swarms import Agent, SwarmRouter, SwarmType
from swarm_models import OpenAIChat

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Model
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)

# Define specialized system prompts for each agent
DATA_EXTRACTOR_PROMPT = """You are a highly specialized private equity agent focused on data extraction from various documents. Your expertise includes:
1. Extracting key financial metrics (revenue, EBITDA, growth rates, etc.) from financial statements and reports
2. Identifying and extracting important contract terms from legal documents
3. Pulling out relevant market data from industry reports and analyses
4. Extracting operational KPIs from management presentations and internal reports
5. Identifying and extracting key personnel information from organizational charts and bios
Provide accurate, structured data extracted from various document types to support investment analysis."""

SUMMARIZER_PROMPT = """You are an expert private equity agent specializing in summarizing complex documents. Your core competencies include:
1. Distilling lengthy financial reports into concise executive summaries
2. Summarizing legal documents, highlighting key terms and potential risks
3. Condensing industry reports to capture essential market trends and competitive dynamics
4. Summarizing management presentations to highlight key strategic initiatives and projections
5. Creating brief overviews of technical documents, emphasizing critical points for non-technical stakeholders
Deliver clear, concise summaries that capture the essence of various documents while highlighting information crucial for investment decisions."""

# Initialize specialized agents
data_extractor_agent = Agent(
    agent_name="Data-Extractor",
    system_prompt=DATA_EXTRACTOR_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="data_extractor_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

summarizer_agent = Agent(
    agent_name="Document-Summarizer",
    system_prompt=SUMMARIZER_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="summarizer_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

# Initialize the SwarmRouter
router = SwarmRouter(
    name="pe-document-analysis-swarm",
    description="Analyze documents for private equity due diligence and investment decision-making",
    max_loops=1,
    agents=[data_extractor_agent, summarizer_agent],
    swarm_type="ConcurrentWorkflow",
    autosave=True,
    return_json=True,
)

# Example usage
if __name__ == "__main__":
    # Run a comprehensive private equity document analysis task
    result = router.run(
        "Where is the best place to find template term sheets for series A startups? Provide links and references"
    )
    print(result)

    # Retrieve and print logs
    for log in router.get_logs():
        print(f"{log.timestamp} - {log.level}: {log.message}")
```

## Advanced Usage

### Changing Swarm Types

You can create multiple SwarmRouter instances with different swarm types:

```python
sequential_router = SwarmRouter(
    name="SequentialRouter",
    agents=[agent1, agent2],
    swarm_type="SequentialWorkflow"
)

concurrent_router = SwarmRouter(
    name="ConcurrentRouter",
    agents=[agent1, agent2],
    swarm_type="ConcurrentWorkflow"
)
```

### Automatic Swarm Type Selection

You can let the SwarmRouter automatically select the best swarm type for a given task:

```python
auto_router = SwarmRouter(
    name="AutoRouter",
    agents=[agent1, agent2],
    swarm_type="auto"
)

result = auto_router.run("Analyze and summarize the quarterly financial report")
```

### Loading Agents from CSV

To load agents from a CSV file:

```python
csv_router = SwarmRouter(
    name="CSVAgentRouter",
    load_agents_from_csv=True,
    csv_file_path="agents.csv",
    swarm_type="SequentialWorkflow"
)

result = csv_router.run("Process the client data")
```

### Using Shared Memory System

To enable shared memory across agents:

```python
from swarms.memory import SemanticMemory

memory_system = SemanticMemory()

memory_router = SwarmRouter(
    name="MemoryRouter",
    agents=[agent1, agent2],
    shared_memory_system=memory_system,
    swarm_type="SequentialWorkflow"
)

result = memory_router.run("Analyze historical data and make predictions")
```

### Injecting Rules to All Agents

To inject common rules into all agents:

```python
rules = """
1. Always provide sources for your information
2. Check your calculations twice
3. Explain your reasoning clearly
4. Highlight uncertainties and assumptions
"""

rules_router = SwarmRouter(
    name="RulesRouter",
    agents=[agent1, agent2],
    rules=rules,
    swarm_type="SequentialWorkflow"
)

result = rules_router.run("Analyze the investment opportunity")
```

## Use Cases

### AgentRearrange

Use Case: Optimizing agent order for complex multi-step tasks.

```python
rearrange_router = SwarmRouter(
    name="TaskOptimizer",
    description="Optimize agent order for multi-step tasks",
    max_loops=3,
    agents=[data_extractor, analyzer, summarizer],
    swarm_type="AgentRearrange",
    rearrange_flow=f"{data_extractor.name} -> {analyzer.name} -> {summarizer.name}"
)

result = rearrange_router.run("Analyze and summarize the quarterly financial report")
```

### MixtureOfAgents

Use Case: Combining diverse expert agents for comprehensive analysis.

```python
mixture_router = SwarmRouter(
    name="ExpertPanel",
    description="Combine insights from various expert agents",
    max_loops=1,
    agents=[financial_expert, market_analyst, tech_specialist, aggregator],
    swarm_type="MixtureOfAgents"
)

result = mixture_router.run("Evaluate the potential acquisition of TechStartup Inc.")
```

### SpreadSheetSwarm

Use Case: Collaborative data processing and analysis.

```python
spreadsheet_router = SwarmRouter(
    name="DataProcessor",
    description="Collaborative data processing and analysis",
    max_loops=1,
    agents=[data_cleaner, statistical_analyzer, visualizer],
    swarm_type="SpreadSheetSwarm"
)

result = spreadsheet_router.run("Process and visualize customer churn data")
```

### SequentialWorkflow

Use Case: Step-by-step document analysis and report generation.

```python
sequential_router = SwarmRouter(
    name="ReportGenerator",
    description="Generate comprehensive reports sequentially",
    max_loops=1,
    agents=[data_extractor, analyzer, writer, reviewer],
    swarm_type="SequentialWorkflow",
    return_entire_history=True
)

result = sequential_router.run("Create a due diligence report for Project Alpha")
```

### ConcurrentWorkflow

Use Case: Parallel processing of multiple data sources.

```python
concurrent_router = SwarmRouter(
    name="MultiSourceAnalyzer",
    description="Analyze multiple data sources concurrently",
    max_loops=1,
    agents=[financial_analyst, market_researcher, competitor_analyst],
    swarm_type="ConcurrentWorkflow",
    output_type="string"
)

result = concurrent_router.run("Conduct a comprehensive market analysis for Product X")
```

### GroupChat

Use Case: Simulating a group discussion with multiple agents.

```python
group_chat_router = SwarmRouter(
    name="GroupChat",
    description="Simulate a group discussion with multiple agents",
    max_loops=10,
    agents=[financial_analyst, market_researcher, competitor_analyst],
    swarm_type="GroupChat",
    speaker_fn=custom_speaker_function
)

result = group_chat_router.run("Discuss the pros and cons of expanding into the Asian market")
```

### MultiAgentRouter

Use Case: Routing tasks to the most appropriate agent.

```python
multi_agent_router = SwarmRouter(
    name="MultiAgentRouter",
    description="Route tasks to specialized agents",
    max_loops=1,
    agents=[financial_analyst, market_researcher, competitor_analyst],
    swarm_type="MultiAgentRouter",
    shared_memory_system=memory_system
)

result = multi_agent_router.run("Analyze the competitive landscape for our new product")
```

See [MultiAgentRouter Minimal Example](../examples/multi_agent_router_minimal.md) for a lightweight demonstration.

### HierarchicalSwarm

Use Case: Creating a hierarchical structure of agents with a director.

```python
hierarchical_router = SwarmRouter(
    name="HierarchicalSwarm",
    description="Hierarchical organization of agents with a director",
    max_loops=3,
    agents=[director, analyst1, analyst2, researcher],
    swarm_type="HiearchicalSwarm",
    return_all_history=True
)

result = hierarchical_router.run("Develop a comprehensive market entry strategy")
```

### MajorityVoting

Use Case: Using consensus among multiple agents for decision-making.

```python
voting_router = SwarmRouter(
    name="MajorityVoting",
    description="Make decisions using consensus among agents",
    max_loops=1,
    agents=[analyst1, analyst2, analyst3, consensus_agent],
    swarm_type="MajorityVoting"
)

result = voting_router.run("Should we invest in Company X based on the available data?")
```

### Auto Select (Experimental)
Autonomously selects the right swarm by conducting vector search on your input task or name or description or all 3.

```python
auto_router = SwarmRouter(
    name="MultiSourceAnalyzer",
    description="Analyze multiple data sources concurrently",
    max_loops=1,
    agents=[financial_analyst, market_researcher, competitor_analyst],
    swarm_type="auto" # Set this to 'auto' for it to auto select your swarm. It's match words like concurrently multiple -> "ConcurrentWorkflow"
)

result = auto_router.run("Conduct a comprehensive market analysis for Product X")
```

### InteractiveGroupChat

Use Case: Interactive group discussions with user participation.

```python
interactive_chat_router = SwarmRouter(
    name="InteractiveGroupChat",
    description="Interactive group chat with user participation",
    max_loops=10,
    agents=[financial_analyst, market_researcher, competitor_analyst],
    swarm_type="InteractiveGroupChat",
    output_type="string"
)

result = interactive_chat_router.run("Discuss the market trends and provide interactive analysis")
```

The InteractiveGroupChat allows for dynamic interaction between agents and users, enabling real-time participation in group discussions and decision-making processes. This is particularly useful for scenarios requiring human input or validation during the conversation flow.

## Advanced Features

### Processing Documents

To process documents with the SwarmRouter:

```python
document_router = SwarmRouter(
    name="DocumentProcessor",
    agents=[document_analyzer, summarizer],
    documents=["report.pdf", "contract.docx", "data.csv"],
    swarm_type="SequentialWorkflow"
)

result = document_router.run("Extract key information from the provided documents")
```

### Batch Processing

To process multiple tasks in a batch:

```python
tasks = ["Analyze Q1 report", "Summarize competitor landscape", "Evaluate market trends"]
results = router.batch_run(tasks)
```

### Asynchronous Execution

For asynchronous task execution:

```python
result = await router.async_run("Generate financial projections")
```

### Concurrent Execution

To run a single task concurrently:

```python
result = router.concurrent_run("Analyze multiple data streams")
```

### Concurrent Batch Processing

To process multiple tasks concurrently:

```python
tasks = ["Task 1", "Task 2", "Task 3"]
results = router.concurrent_batch_run(tasks)
```

### Using the SwarmRouter as a Callable

You can use the SwarmRouter instance directly as a callable:

```python
router = SwarmRouter(
    name="CallableRouter",
    agents=[agent1, agent2],
    swarm_type="SequentialWorkflow"
)

result = router("Analyze the market data")  # Equivalent to router.run("Analyze the market data")
```

### Using the swarm_router Function

For quick one-off tasks, you can use the swarm_router function:

```python
from swarms import swarm_router

result = swarm_router(
    name="QuickRouter",
    agents=[agent1, agent2],
    swarm_type="ConcurrentWorkflow",
    task="Analyze the quarterly report"
)
```
