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
from swarms import Agent, ConcurrentWorkflow

# Initialize market research agent
market_researcher = Agent(
    agent_name="Market-Researcher",
    system_prompt="""You are a market research specialist. Your tasks include:
    1. Analyzing market trends and patterns
    2. Identifying market opportunities and threats
    3. Evaluating competitor strategies
    4. Assessing customer needs and preferences
    5. Providing actionable market insights""",
    model_name="gpt-4o",
    max_loops=1,
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
    model_name="gpt-4o",
    max_loops=1,
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
    model_name="gpt-4o",
    max_loops=1,
)
```

### 2. Create and Run ConcurrentWorkflow

```python
# Create list of agents
agents = [market_researcher, financial_analyst, technical_analyst]

# Initialize the concurrent workflow
workflow = ConcurrentWorkflow(
    name="market-analysis-workflow",
    agents=agents,
    max_loops=1,
)

# Run the workflow
result = workflow.run(
    "Analyze Tesla (TSLA) stock from market, financial, and technical perspectives"
)
```

## Advanced Usage

### 1. Custom Agent Configuration

```python
from swarms import Agent, ConcurrentWorkflow

# Initialize agents with custom configurations
sentiment_analyzer = Agent(
    agent_name="Sentiment-Analyzer",
    system_prompt="You analyze social media sentiment...",
    model_name="gpt-4o",
    max_loops=1,
    temperature=0.7,
    streaming_on=True,
    verbose=True,
)

news_analyzer = Agent(
    agent_name="News-Analyzer",
    system_prompt="You analyze news articles and reports...",
    model_name="gpt-4o",
    max_loops=1,
    temperature=0.5,
    streaming_on=True,
    verbose=True,
)

# Create and run workflow
workflow = ConcurrentWorkflow(
    name="sentiment-analysis-workflow",
    agents=[sentiment_analyzer, news_analyzer],
    max_loops=1,
    verbose=True,
)

result = workflow.run(
    "Analyze the market sentiment for Bitcoin based on social media and news"
)
```

### 2. Error Handling and Logging

```python
try:
    workflow = ConcurrentWorkflow(
        name="error-handled-workflow",
        agents=agents,
        max_loops=1,
        verbose=True,
    )
    
    result = workflow.run("Complex analysis task")
    
    # Process results
    for agent_result in result:
        print(f"Agent {agent_result['agent']}: {agent_result['output']}")
        
except Exception as e:
    print(f"Error in workflow: {str(e)}")
```

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
   - Monitor concurrent execution
   - Handle rate limits appropriately
   - Manage memory usage

4. Error Handling:
   - Implement proper error handling
   - Log errors and exceptions
   - Provide fallback mechanisms

## Example Implementation

Here's a complete example showing how to use ConcurrentWorkflow for a comprehensive market analysis:

```python
import os
from swarms import Agent, ConcurrentWorkflow

# Initialize specialized agents
market_analyst = Agent(
    agent_name="Market-Analyst",
    system_prompt="""You are a market analysis specialist focusing on:
    1. Market trends and patterns
    2. Competitive analysis
    3. Market opportunities
    4. Industry dynamics
    5. Growth potential""",
    model_name="gpt-4o",
    max_loops=1,
)

financial_analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="""You are a financial analysis expert specializing in:
    1. Financial statements analysis
    2. Ratio analysis
    3. Cash flow analysis
    4. Valuation metrics
    5. Risk assessment""",
    model_name="gpt-4o",
    max_loops=1,
)

risk_analyst = Agent(
    agent_name="Risk-Analyst",
    system_prompt="""You are a risk assessment specialist focusing on:
    1. Market risks
    2. Operational risks
    3. Financial risks
    4. Regulatory risks
    5. Strategic risks""",
    model_name="gpt-4o",
    max_loops=1,
)

# Create the concurrent workflow
workflow = ConcurrentWorkflow(
    name="comprehensive-analysis-workflow",
    agents=[market_analyst, financial_analyst, risk_analyst],
    max_loops=1,
    verbose=True,
)

# Run the analysis
try:
    result = workflow.run(
        """Provide a comprehensive analysis of Apple Inc. (AAPL) including:
        1. Market position and competitive analysis
        2. Financial performance and health
        3. Risk assessment and mitigation strategies"""
    )
    
    # Process and display results
    for agent_result in result:
        print(f"\nAnalysis from {agent_result['agent']}:")
        print(agent_result['output'])
        print("-" * 50)
        
except Exception as e:
    print(f"Error during analysis: {str(e)}")
```

This comprehensive guide demonstrates how to effectively use the ConcurrentWorkflow architecture for parallel processing of complex tasks using multiple specialized agents. 