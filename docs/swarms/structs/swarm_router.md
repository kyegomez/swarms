# SwarmRouter Documentation

The `SwarmRouter` class is a flexible routing system designed to manage different types of swarms for task execution. It provides a unified interface to interact with various swarm types.

Full Path: `from swarms.structs.swarm_router`

## Initialization Parameters

Main class for routing tasks to different swarm types.

| Attribute | Type | Description |
| --- | --- | --- |
| `id` | str | Unique identifier for the SwarmRouter instance (auto-generated if not provided) |
| `name` | str | Name of the SwarmRouter instance |
| `description` | str | Description of the SwarmRouter's purpose |
| `max_loops` | int | Maximum number of loops to perform |
| `agents` | List[Union[Agent, Callable]] | List of Agent objects or callable functions |
| `swarm_type` | SwarmType | Type of swarm to be used |
| `autosave` | bool | Flag to enable/disable autosave. When enabled, automatically saves swarm configuration, state, and metadata to `workspace_dir/swarms/SwarmRouter/{swarm-name}-{timestamp}/`. Saves `config.json` on initialization, and `state.json` + `metadata.json` after each run. Defaults to `False` |
| `autosave_use_timestamp` | bool | If `True`, use timestamp in directory name; if `False`, use UUID. Defaults to `True` |
| `rearrange_flow` | str | The flow for the AgentRearrange swarm type |
| `return_json` | bool | Flag to enable/disable returning the result in JSON format |
| `auto_generate_prompts` | bool | Flag to enable/disable auto generation of prompts |
| `shared_memory_system` | Any | Shared memory system for agents |
| `rules` | str | Rules to inject into every agent |
| `documents` | List[str] | List of document file paths |
| `output_type` | OutputType | Output format type (e.g., "string", "dict", "list", "json", "yaml", "xml", "dict-all-except-first"). Defaults to "dict-all-except-first" |
| `speaker_fn` | callable | Legacy speaker function for GroupChat swarm type (deprecated, use speaker_function instead) |
| `speaker_function` | str | Speaker function name for GroupChat swarm type (e.g., "round-robin-speaker", "random-speaker", "priority-speaker", "random-dynamic-speaker") |
| `load_agents_from_csv` | bool | Flag to enable/disable loading agents from CSV |
| `csv_file_path` | str | Path to the CSV file for loading agents |
| `return_entire_history` | bool | Flag to enable/disable returning the entire conversation history. Defaults to `True` |
| `multi_agent_collab_prompt` | bool | Whether to enable multi-agent collaboration prompts. Defaults to `True` |
| `list_all_agents` | bool | Flag to enable/disable listing all agents to each other. Defaults to `False` |
| `conversation` | Any | Conversation object for managing agent interactions |
| `agents_config` | Optional[Dict[Any, Any]] | Configuration dictionary for agents |
| `heavy_swarm_loops_per_agent` | int | Number of loops per agent for HeavySwarm (default: 1) |
| `heavy_swarm_question_agent_model_name` | str | Model name for the question agent in HeavySwarm (default: "gpt-4.1") |
| `heavy_swarm_worker_model_name` | str | Model name for worker agents in HeavySwarm (default: "gpt-4.1") |
| `heavy_swarm_swarm_show_output` | bool | Flag to show output for HeavySwarm (default: True) |
| `telemetry_enabled` | bool | Flag to enable/disable telemetry logging (default: False) |
| `council_judge_model_name` | str | Model name for the judge in CouncilAsAJudge (default: "gpt-4o-mini") |
| `verbose` | bool | Flag to enable/disable verbose logging (default: False) |
| `worker_tools` | List[Callable] | List of tools available to worker agents |
| `aggregation_strategy` | str | Aggregation strategy for HeavySwarm (default: "synthesis") |
| `chairman_model` | str | Model name for the Chairman in LLMCouncil (default: "gpt-5.1") |

### Methods

#### `run()`

Execute a task on the selected swarm type.

**Input Parameters:**

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `task` | `Optional[str]` | No | `None` | The task to be executed by the swarm |
| `img` | `Optional[str]` | No | `None` | Path to an image file for vision tasks |
| `tasks` | `Optional[List[str]]` | No | `None` | List of tasks (used for BatchedGridWorkflow) |
| `*args` | `Any` | No | - | Variable length argument list |
| `**kwargs` | `Any` | No | - | Arbitrary keyword arguments |

**Output:**

| Type | Description |
| --- | --- |
| `Any` | The result of the swarm's execution. The exact type depends on the `output_type` configuration (e.g., `str`, `dict`, `list`, `json`, `yaml`, `xml`) |

**Example:**

```python
result = router.run(
    task="Analyze the market trends and provide recommendations",
    img="chart.png"  # Optional
)
```

---

### `batch_run()`

Execute multiple tasks in sequence on the selected swarm type.

**Input Parameters:**

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `tasks` | `List[str]` | Yes | - | List of tasks to be executed sequentially |
| `img` | `Optional[str]` | No | `None` | Path to an image file for vision tasks |
| `imgs` | `Optional[List[str]]` | No | `None` | List of image file paths for vision tasks |
| `*args` | `Any` | No | - | Variable length argument list |
| `**kwargs` | `Any` | No | - | Arbitrary keyword arguments |

**Output:**

| Type | Description |
| --- | --- |
| `List[Any]` | A list of results from the swarm's execution, one result per task. Each result type depends on the `output_type` configuration |

**Example:**

```python
tasks = ["Analyze Q1 report", "Summarize competitor landscape", "Evaluate market trends"]
results = router.batch_run(tasks, img="report.png")  # Optional img parameter
```

## Available Swarm Types

The `SwarmRouter` supports many various multi-agent architectures for various applications.

| Swarm Type | Description |
|------------|-------------|
| `AgentRearrange` | Optimizes agent arrangement for task execution |
| `MixtureOfAgents` | Combines multiple agent types for diverse tasks |
| `SequentialWorkflow` | Executes tasks sequentially |
| `ConcurrentWorkflow` | Executes tasks in parallel |
| `GroupChat` | Facilitates communication among agents in a group chat format |
| `MultiAgentRouter` | Routes tasks between multiple agents |
| `AutoSwarmBuilder` | Automatically builds swarm structure |
| `HiearchicalSwarm` | Hierarchical organization of agents |
| `MajorityVoting` | Uses majority voting for decision making |
| `CouncilAsAJudge` | Council-based judgment system |
| `HeavySwarm` | Heavy swarm architecture with question and worker agents |
| `BatchedGridWorkflow` | Batched grid workflow for parallel task processing |
| `LLMCouncil` | Council of specialized LLM agents with peer review and synthesis |
| `DebateWithJudge` | Debate architecture with Pro/Con agents and a Judge for self-refinement |
| `auto` | Automatically selects best swarm type via embedding search |

## Basic Usage

```python
import os
from dotenv import load_dotenv
from swarms import Agent, SwarmRouter, SwarmType

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
    model_name="gpt-4.1",
    max_loops=1,
)

summarizer_agent = Agent(
    agent_name="Document-Summarizer",
    system_prompt=SUMMARIZER_PROMPT,
    model_name="gpt-4.1",
    max_loops=1,
)

# Initialize the SwarmRouter
router = SwarmRouter(
    name="pe-document-analysis-swarm",
    description="Analyze documents for private equity due diligence and investment decision-making",
    max_loops=1,
    agents=[data_extractor_agent, summarizer_agent],
    swarm_type="ConcurrentWorkflow",
)

# Example usage
if __name__ == "__main__":
    # Run a comprehensive private equity document analysis task
    result = router.run(
        task="Where is the best place to find template term sheets for series A startups? Provide links and references",
        img=None  # Optional: provide image path for vision tasks
    )
    print(result)
    
    # For BatchedGridWorkflow, you can pass multiple tasks:
    # result = router.run(tasks=["Task 1", "Task 2", "Task 3"])
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

### HeavySwarm

Use Case: Complex task decomposition with question and worker agents.

```python
heavy_swarm_router = SwarmRouter(
    name="HeavySwarm",
    description="Complex task decomposition and execution",
    swarm_type="HeavySwarm",
    heavy_swarm_loops_per_agent=2,
    heavy_swarm_question_agent_model_name="gpt-4.1",
    heavy_swarm_worker_model_name="gpt-4.1",
    heavy_swarm_swarm_show_output=True,
    worker_tools=[tool1, tool2],
    aggregation_strategy="synthesis",
    output_type="string"
)

result = heavy_swarm_router.run("Analyze market trends and provide comprehensive recommendations")
```

HeavySwarm uses a question agent to decompose complex tasks and worker agents to execute subtasks, making it ideal for complex problem-solving scenarios.

### BatchedGridWorkflow

Use Case: Parallel processing of multiple tasks in a batched grid format.

```python
batched_grid_router = SwarmRouter(
    name="BatchedGridWorkflow",
    description="Process multiple tasks in parallel batches",
    max_loops=1,
    agents=[agent1, agent2, agent3],
    swarm_type="BatchedGridWorkflow"
)

result = batched_grid_router.run(tasks=["Task 1", "Task 2", "Task 3"])
```

BatchedGridWorkflow is designed for efficiently processing multiple tasks in parallel batches, optimizing resource utilization.

### LLMCouncil

Use Case: Collaborative analysis with multiple specialized LLM agents that evaluate each other's responses and synthesize a final answer.

```python
llm_council_router = SwarmRouter(
    name="LLMCouncil",
    description="Collaborative council of LLM agents with peer review",
    swarm_type="LLMCouncil",
    chairman_model="gpt-5.1",  # Model for the Chairman agent
    output_type="dict",  # Output format: "dict", "list", "string", "json", "yaml", "final", etc.
    verbose=True  # Show progress and intermediate results
)

result = llm_council_router.run("What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?")
```

LLMCouncil creates a council of specialized agents (GPT-5.1, Gemini, Claude, Grok by default) that:
1. Each independently responds to the query
2. Evaluates and ranks each other's anonymized responses
3. A Chairman synthesizes all responses and evaluations into a final comprehensive answer

The council automatically tracks all messages in a conversation object and supports flexible output formats. Note: LLMCouncil uses default council members and doesn't require the `agents` parameter.

### DebateWithJudge

Use Case: Structured debate architecture where two agents (Pro and Con) present opposing arguments, and a Judge agent evaluates and synthesizes the arguments over multiple rounds to progressively refine the answer.

```python
from swarms import Agent, SwarmRouter

# Create three specialized agents for the debate
pro_agent = Agent(
    agent_name="Pro-Agent",
    system_prompt="You are an expert at presenting strong, well-reasoned arguments in favor of positions. "
                  "You provide compelling evidence and logical reasoning to support your stance.",
    model_name="gpt-4.1",
    max_loops=1,
)

con_agent = Agent(
    agent_name="Con-Agent",
    system_prompt="You are an expert at presenting strong, well-reasoned counter-arguments. "
                  "You identify weaknesses in opposing arguments and present compelling evidence against positions.",
    model_name="gpt-4.1",
    max_loops=1,
)

judge_agent = Agent(
    agent_name="Judge-Agent",
    system_prompt="You are an impartial judge evaluating debates. You carefully assess both arguments, "
                  "identify strengths and weaknesses, and provide refined synthesis that incorporates "
                  "the best elements from both sides.",
    model_name="gpt-4.1",
    max_loops=1,
)

# Initialize the SwarmRouter with DebateWithJudge
debate_router = SwarmRouter(
    name="DebateWithJudge",
    description="Structured debate with Pro/Con agents and Judge for self-refinement",
    swarm_type="DebateWithJudge",
    agents=[pro_agent, con_agent, judge_agent],  # Must be exactly 3 agents
    max_loops=3,  # Number of debate rounds
    output_type="str-all-except-first",  # Output format
    verbose=True  # Show progress and intermediate results
)

# Run a debate on a topic
result = debate_router.run(
    "Should artificial intelligence development be regulated by governments?"
)
```

DebateWithJudge implements a multi-round debate system where:
1. **Pro Agent** presents arguments in favor of the topic
2. **Con Agent** presents counter-arguments against the topic
3. **Judge Agent** evaluates both arguments and provides synthesis
4. The process repeats for N rounds (specified by `max_loops`), with each round refining the discussion based on the judge's feedback

The architecture progressively improves the answer through iterative refinement, making it ideal for complex topics requiring thorough analysis from multiple perspectives. Note: DebateWithJudge requires exactly 3 agents (pro_agent, con_agent, judge_agent) in that order.

## Advanced Features

### Autosave Functionality

The SwarmRouter supports automatic saving of configurations, state, and metadata when `autosave=True`. This feature helps with persistence, debugging, and tracking swarm executions.

**Directory Structure:**
```
workspace_dir/
└── swarms/
    └── SwarmRouter/
        └── {swarm-name}-{timestamp}/
            ├── config.json      # Initial configuration
            ├── state.json       # Current state (conversation, logs)
            └── metadata.json    # Execution metadata
```

**Files Saved:**
- **config.json**: Contains the initial swarm configuration, including all parameters set during initialization
- **state.json**: Contains the current state including conversation history and logs
- **metadata.json**: Contains execution metadata including task details, results summary, and execution timestamps

**Example:**

```python
router = SwarmRouter(
    name="MySwarm",
    description="Example swarm with autosave",
    agents=[agent1, agent2],
    swarm_type="SequentialWorkflow",
    autosave=True,  # Enable autosave
    autosave_use_timestamp=True,  # Use timestamp in directory name
    max_loops=1
)

# Configuration is automatically saved on initialization
# State and metadata are saved after each run
result = router.run("Analyze the market trends")
```

**Note:** The `WORKSPACE_DIR` environment variable must be set for autosave to work. If not set, autosave will be silently disabled.

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
results = router.batch_run(tasks, img="image.png")  # Optional: img parameter for image tasks
```

### Concurrent Execution

To run a single task concurrently:

```python
result = router.concurrent_run("Analyze multiple data streams", img="image.png")  # Optional: img parameter
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
