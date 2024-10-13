# SwarmRouter Documentation

## Overview

The `SwarmRouter` class is a flexible routing system designed to manage different types of swarms for task execution. It provides a unified interface to interact with various swarm types, including `AgentRearrange`, `MixtureOfAgents`, `SpreadSheetSwarm`, `SequentialWorkflow`, and `ConcurrentWorkflow`. We will be continously adding more and more swarm architectures here as we progress with new architectures.

## Classes

### SwarmLog

A Pydantic model for capturing log entries.

#### Attributes:
- `id` (str): Unique identifier for the log entry.
- `timestamp` (datetime): Time of log creation.
- `level` (str): Log level (e.g., "info", "error").
- `message` (str): Log message content.
- `swarm_type` (SwarmType): Type of swarm associated with the log.
- `task` (str): Task being performed (optional).
- `metadata` (Dict[str, Any]): Additional metadata (optional).

### SwarmRouter

Main class for routing tasks to different swarm types.

#### Attributes:
- `name` (str): Name of the SwarmRouter instance.
- `description` (str): Description of the SwarmRouter instance.
- `max_loops` (int): Maximum number of loops to perform.
- `agents` (List[Agent]): List of Agent objects to be used in the swarm.
- `swarm_type` (SwarmType): Type of swarm to be used.
- `swarm` (Union[AgentRearrange, MixtureOfAgents, SpreadSheetSwarm, SequentialWorkflow, ConcurrentWorkflow]): Instantiated swarm object.
- `logs` (List[SwarmLog]): List of log entries captured during operations.

#### Methods:
- `__init__(self, name: str, description: str, max_loops: int, agents: List[Agent], swarm_type: SwarmType, *args, **kwargs)`: Initialize the SwarmRouter.
- `_create_swarm(self, *args, **kwargs)`: Create and return the specified swarm type.
- `_log(self, level: str, message: str, task: str, metadata: Dict[str, Any])`: Create a log entry and add it to the logs list.
- `run(self, task: str, *args, **kwargs)`: Run the specified task on the selected swarm.
- `get_logs(self)`: Retrieve all logged entries.

## Installation

To use the SwarmRouter, first install the required dependencies:

```bash
pip install swarms swarm_models
```

## Basic Usage
```python
import os
from dotenv import load_dotenv
from swarms import Agent, SwarmRouter
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

FINANCIAL_ANALYST_PROMPT = """You are a specialized private equity agent focused on financial analysis. Your key responsibilities include:
1. Analyzing historical financial statements to identify trends and potential issues
2. Evaluating the quality of earnings and potential adjustments to EBITDA
3. Assessing working capital requirements and cash flow dynamics
4. Analyzing capital structure and debt capacity
5. Evaluating financial projections and underlying assumptions
Provide thorough, insightful financial analysis to inform investment decisions and valuation."""

MARKET_ANALYST_PROMPT = """You are a highly skilled private equity agent specializing in market analysis. Your expertise covers:
1. Analyzing industry trends, growth drivers, and potential disruptors
2. Evaluating competitive landscape and market positioning
3. Assessing market size, segmentation, and growth potential
4. Analyzing customer dynamics, including concentration and loyalty
5. Identifying potential regulatory or macroeconomic impacts on the market
Deliver comprehensive market analysis to assess the attractiveness and risks of potential investments."""

OPERATIONAL_ANALYST_PROMPT = """You are an expert private equity agent focused on operational analysis. Your core competencies include:
1. Evaluating operational efficiency and identifying improvement opportunities
2. Analyzing supply chain and procurement processes
3. Assessing sales and marketing effectiveness
4. Evaluating IT systems and digital capabilities
5. Identifying potential synergies in merger or add-on acquisition scenarios
Provide detailed operational analysis to uncover value creation opportunities and potential risks."""

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

financial_analyst_agent = Agent(
    agent_name="Financial-Analyst",
    system_prompt=FINANCIAL_ANALYST_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="financial_analyst_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

market_analyst_agent = Agent(
    agent_name="Market-Analyst",
    system_prompt=MARKET_ANALYST_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="market_analyst_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

operational_analyst_agent = Agent(
    agent_name="Operational-Analyst",
    system_prompt=OPERATIONAL_ANALYST_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="operational_analyst_agent.json",
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
    agents=[
        data_extractor_agent,
        summarizer_agent,
        financial_analyst_agent,
        market_analyst_agent,
        operational_analyst_agent,
    ],
    swarm_type="ConcurrentWorkflow",  # or "SequentialWorkflow" or "ConcurrentWorkflow" or
)

# Example usage
if __name__ == "__main__":
    # Run a comprehensive private equity document analysis task
    result = router.run(
        "Where is the best place to find template term sheets for series A startups. Provide links and references"
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
    swarm_type=SwarmType.SequentialWorkflow
)

concurrent_router = SwarmRouter(
    name="ConcurrentRouter",
    agents=[agent1, agent2],
    swarm_type=SwarmType.ConcurrentWorkflow
)
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
    swarm_type=SwarmType.AgentRearrange,
    flow = f"{data_extractor.name} -> {analyzer.name} -> {summarizer.name}"
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
    agents=[financial_expert, market_analyst, tech_specialist],
    swarm_type=SwarmType.MixtureOfAgents
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
    swarm_type=SwarmType.SpreadSheetSwarm
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
    swarm_type=SwarmType.SequentialWorkflow
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
    swarm_type=SwarmType.ConcurrentWorkflow
)

result = concurrent_router.run("Conduct a comprehensive market analysis for Product X")
```

## Error Handling

The `SwarmRouter` includes error handling in the `run` method. If an exception occurs during task execution, it will be logged and re-raised for the caller to handle. Always wrap the `run` method in a try-except block:

```python
try:
    result = router.run("Complex analysis task")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    # Handle the error appropriately
```

## Best Practices

1. Choose the appropriate swarm type based on your task requirements.
2. Provide clear and specific tasks to the swarm for optimal results.
3. Regularly review logs to monitor performance and identify potential issues.
4. Use descriptive names and descriptions for your SwarmRouter and agents.
5. Implement proper error handling in your application code.
6. Consider the nature of your tasks when choosing a swarm type (e.g., use ConcurrentWorkflow for tasks that can be parallelized).
7. Optimize your agents' prompts and configurations for best performance within the swarm.