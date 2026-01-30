# ConcurrentWorkflow Examples

The ConcurrentWorkflow architecture enables parallel execution of multiple agents, allowing them to work simultaneously on different aspects of a task. This is particularly useful for complex tasks that can be broken down into independent subtasks.

## Prerequisites

- Python 3.7+
- OpenAI API key or other supported LLM provider keys
- Swarms library

## Installation

```bash
pip3 install -U swarms
```

## Environment Variables

```plaintext
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
```

## Basic Usage

### 1. Initialize Specialized Agents

```python
from swarms import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow

# Initialize market research agent
market_researcher = Agent(
    agent_name="Market-Researcher",
    system_prompt="""You are a market research specialist. Your tasks include:
    1. Analyzing market trends and patterns
    2. Identifying market opportunities and threats
    3. Evaluating competitor strategies
    4. Assessing customer needs and preferences
    5. Providing actionable market insights""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

# Initialize financial analyst agent
financial_analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="""You are a financial analysis expert. Your responsibilities include:
    1. Analyzing financial statements
    2. Evaluating investment opportunities
    3. Assessing risk factors
    4. Providing financial forecasts
    5. Recommending financial strategies""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

# Initialize technical analyst agent
technical_analyst = Agent(
    agent_name="Technical-Analyst",
    system_prompt="""You are a technical analysis specialist. Your focus areas include:
    1. Analyzing price patterns and trends
    2. Evaluating technical indicators
    3. Identifying support and resistance levels
    4. Assessing market momentum
    5. Providing trading recommendations""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

# Create list of agents
agents = [market_researcher, financial_analyst, technical_analyst]

# Initialize the concurrent workflow with dashboard
router = ConcurrentWorkflow(
    name="market-analysis-router",
    agents=agents,
    max_loops=1,
    show_dashboard=True,  # Enable the real-time dashboard
)

# Run the workflow
result = router.run(
    "Analyze Tesla (TSLA) stock from market, financial, and technical perspectives"
)
```

## Features

### Real-time Dashboard

The ConcurrentWorkflow includes a powerful real-time dashboard feature that provides comprehensive monitoring and visualization of agent execution. Enable it by setting `show_dashboard=True` during workflow initialization.

#### Dashboard Features

- **Live Status Tracking**: Real-time updates showing each agent's execution status
- **Progress Visualization**: Visual indicators of agent progress and completion
- **Output Streaming**: Live display of agent outputs as they're generated
- **Error Monitoring**: Immediate visibility into any agent failures or errors
- **Performance Metrics**: Execution time and completion statistics
- **Clean Display**: Automatic cleanup and formatting for optimal viewing

#### Dashboard Status Values

- **"pending"**: Agent is queued but not yet started
- **"running"**: Agent is currently executing the task
- **"completed"**: Agent finished successfully with output
- **"error"**: Agent execution failed with error details

#### Dashboard Configuration

```python
# Enable dashboard with custom configuration
workflow = ConcurrentWorkflow(
    name="my-workflow",
    agents=agents,
    show_dashboard=True,  # Enable real-time monitoring
    output_type="dict",   # Configure output format
    auto_save=True,       # Auto-save conversation history
)
```

#### Dashboard Behavior

When `show_dashboard=True`:
- Individual agent print outputs are automatically disabled to prevent conflicts
- Dashboard updates every 100ms for smooth real-time streaming
- Initial dashboard shows all agents as "pending"
- Real-time updates show status changes and output previews
- Final dashboard displays complete results summary
- Automatic cleanup of dashboard resources after completion

### Concurrent Execution

- **ThreadPoolExecutor**: Uses 95% of available CPU cores for optimal performance
- **True Parallelism**: Agents execute simultaneously, not sequentially
- **Thread Safety**: Safe concurrent access to shared resources
- **Error Isolation**: Individual agent failures don't affect others
- **Resource Management**: Automatic thread lifecycle management

### Output Formatting Options

The workflow supports multiple output aggregation formats:

- **"dict-all-except-first"**: Dictionary with all agent outputs except the first (default)
- **"dict"**: Complete dictionary with all agent outputs keyed by agent name
- **"str"**: Concatenated string of all agent outputs
- **"list"**: List of individual agent outputs in completion order

```python
# Configure output format
workflow = ConcurrentWorkflow(
    agents=agents,
    output_type="dict",  # Get complete dictionary of results
    show_dashboard=True
)
```

### Advanced Features

#### Auto Prompt Engineering

Enable automatic prompt optimization for all agents:

```python
workflow = ConcurrentWorkflow(
    agents=agents,
    auto_generate_prompts=True,  # Enable automatic prompt engineering
    show_dashboard=True
)
```

## Autosave Feature

Autosave is enabled by default (`autosave=True`). Conversation history is automatically saved to `{workspace_dir}/swarms/ConcurrentWorkflow/{workflow-name}-{timestamp}/conversation_history.json`.

To set a custom workspace directory name, use the `WORKSPACE_DIR` environment variable:

```python
import os
from swarms import Agent, ConcurrentWorkflow

# Set custom workspace directory where conversation history will be saved
# If not set, defaults to 'agent_workspace' in the current directory
os.environ["WORKSPACE_DIR"] = "my_project"

# Create workflow (autosave enabled by default)
workflow = ConcurrentWorkflow(
    name="analysis-workflow",
    agents=[analyst1, analyst2, analyst3],
)

# Run workflow - conversation automatically saved
result = workflow.run("Analyze market trends")
```

#### Multimodal Support

Support for image inputs across all agents:

```python
# Single image input
result = workflow.run(
    task="Analyze this chart",
    img="financial_chart.png"
)

# Multiple image inputs
result = workflow.run(
    task="Compare these charts",
    imgs=["chart1.png", "chart2.png", "chart3.png"]
)
```

## Best Practices

### 1. Dashboard Usage

- **Development & Debugging**: Use dashboard for real-time monitoring during development
- **Production**: Consider disabling dashboard for headless execution in production
- **Performance**: Dashboard adds minimal overhead but provides valuable insights
- **Error Handling**: Dashboard immediately shows which agents fail and why

### 2. Agent Configuration

- **Specialization**: Use specialized agents for specific tasks
- **Model Selection**: Choose appropriate models for each agent's role
- **Temperature**: Configure temperature based on task requirements
- **System Prompts**: Write clear, specific system prompts for each agent

### 3. Resource Management

- **CPU Utilization**: Workflow automatically uses 95% of available cores
- **Memory**: Monitor conversation history growth in long-running workflows
- **Rate Limits**: Handle API rate limits appropriately for your LLM provider
- **Error Recovery**: Implement fallback mechanisms for failed agents

### 4. Task Design

- **Independence**: Ensure tasks can be processed concurrently without dependencies
- **Granularity**: Break complex tasks into independent subtasks
- **Balance**: Distribute work evenly across agents for optimal performance

## Example Implementations

### Comprehensive Market Analysis

```python
from swarms import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow

# Initialize specialized agents
market_analyst = Agent(
    agent_name="Market-Analyst",
    system_prompt="""You are a market analysis specialist focusing on:
    1. Market trends and patterns
    2. Competitive analysis
    3. Market opportunities
    4. Industry dynamics
    5. Growth potential""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

financial_analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="""You are a financial analysis expert specializing in:
    1. Financial statements analysis
    2. Ratio analysis
    3. Cash flow analysis
    4. Valuation metrics
    5. Risk assessment""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

risk_analyst = Agent(
    agent_name="Risk-Analyst",
    system_prompt="""You are a risk assessment specialist focusing on:
    1. Market risks
    2. Operational risks
    3. Financial risks
    4. Regulatory risks
    5. Strategic risks""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

# Create the concurrent workflow with dashboard
workflow = ConcurrentWorkflow(
    name="comprehensive-analysis-workflow",
    agents=[market_analyst, financial_analyst, risk_analyst],
    max_loops=1,
    show_dashboard=True,  # Enable real-time monitoring
    output_type="dict",   # Get structured results
    auto_save=True,       # Save conversation history
)

try:
    result = workflow.run(
        """Provide a comprehensive analysis of Apple Inc. (AAPL) including:
        1. Market position and competitive analysis
        2. Financial performance and health
        3. Risk assessment and mitigation strategies"""
    )
    
    # Process and display results
    print("\nAnalysis Results:")
    print("=" * 50)
    for agent_name, output in result.items():
        print(f"\nAnalysis from {agent_name}:")
        print("-" * 40)
        print(output)
        
except Exception as e:
    print(f"Error during analysis: {str(e)}")
```

### Batch Processing with Dashboard

```python
# Process multiple tasks sequentially with concurrent agent execution
tasks = [
    "Analyze Q1 financial performance and market position",
    "Analyze Q2 financial performance and market position", 
    "Analyze Q3 financial performance and market position",
    "Analyze Q4 financial performance and market position"
]

# Optional: corresponding images for each task
charts = ["q1_chart.png", "q2_chart.png", "q3_chart.png", "q4_chart.png"]

# Batch processing with dashboard monitoring
results = workflow.batch_run(tasks, imgs=charts)

print(f"Completed {len(results)} quarterly analyses")
for i, result in enumerate(results):
    print(f"\nQ{i+1} Analysis Results:")
    print(result)
```

### Multimodal Analysis

```python
# Analyze financial charts with multiple specialized agents
workflow = ConcurrentWorkflow(
    agents=[technical_analyst, fundamental_analyst, sentiment_analyst],
    show_dashboard=True,
    output_type="dict"
)

# Analyze a single chart
result = workflow.run(
    task="Analyze this stock chart and provide trading insights",
    img="stock_chart.png"
)

# Analyze multiple charts
result = workflow.run(
    task="Compare these three charts and identify patterns",
    imgs=["chart1.png", "chart2.png", "chart3.png"]
)
```

### Error Handling and Monitoring

```python
# Workflow with comprehensive error handling
workflow = ConcurrentWorkflow(
    agents=agents,
    show_dashboard=True,  # Monitor execution in real-time
    auto_save=True,       # Preserve results even if errors occur
    output_type="dict"    # Get structured results for easier processing
)

try:
    result = workflow.run("Complex analysis task")
    
    # Check for errors in results
    for agent_name, output in result.items():
        if output.startswith("Error:"):
            print(f"Agent {agent_name} failed: {output}")
        else:
            print(f"Agent {agent_name} completed successfully")
            
except Exception as e:
    print(f"Workflow execution failed: {str(e)}")
    # Results may still be available for successful agents
```

## Performance Tips

1. **Agent Count**: Use 2+ agents to benefit from concurrent execution
2. **CPU Utilization**: Workflow automatically optimizes for available cores
3. **Dashboard Overhead**: Minimal performance impact for valuable monitoring
4. **Memory Management**: Clear conversation history for very large batch jobs
5. **Error Recovery**: Failed agents don't stop successful ones

## Use Cases

- **Multi-perspective Analysis**: Financial, legal, technical reviews
- **Consensus Building**: Voting systems and decision making
- **Parallel Processing**: Data analysis and batch operations
- **A/B Testing**: Different agent configurations and strategies
- **Redundancy**: Reliability improvements through multiple agents
- **Real-time Monitoring**: Development and debugging workflows

This guide demonstrates how to effectively use the ConcurrentWorkflow architecture with its advanced dashboard feature for parallel processing of complex tasks using multiple specialized agents. 