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

The ConcurrentWorkflow now includes a real-time dashboard feature that can be enabled by setting `show_dashboard=True`. This provides:

- Live status of each agent's execution
- Progress tracking
- Real-time output visualization
- Task completion metrics

### Concurrent Execution

- Multiple agents work simultaneously
- Efficient resource utilization
- Automatic task distribution
- Built-in thread management

## Best Practices

1. Task Distribution:
   - Break down complex tasks into independent subtasks
   - Assign appropriate agents to each subtask
   - Ensure tasks can be processed concurrently

2. Agent Configuration:
   - Use specialized agents for specific tasks
   - Configure appropriate model parameters
   - Set meaningful system prompts

3. Resource Management:
   - Monitor concurrent execution through the dashboard
   - Handle rate limits appropriately
   - Manage memory usage

4. Error Handling:
   - Implement proper error handling
   - Log errors and exceptions
   - Provide fallback mechanisms

## Example Implementation

Here's a complete example showing how to use ConcurrentWorkflow for a comprehensive market analysis:

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
    for agent_output in result:
        print(f"\nAnalysis from {agent_output['agent']}:")
        print("-" * 40)
        print(agent_output['output'])
        
except Exception as e:
    print(f"Error during analysis: {str(e)}")
```

This guide demonstrates how to effectively use the ConcurrentWorkflow architecture with its new dashboard feature for parallel processing of complex tasks using multiple specialized agents. 