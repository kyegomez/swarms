# Basic Agent Example

This example demonstrates how to create and configure a sophisticated AI agent using the Swarms framework. In this tutorial, we'll build a Quantitative Trading Agent that can analyze financial markets and provide investment insights. The agent is powered by GPT models and can be customized for various financial analysis tasks.

## Prerequisites

- Python 3.7+

- OpenAI API key

- Swarms library

## Tutorial Steps

1. First, install the latest version of Swarms:

```bash
pip3 install -U swarms
```

2. Set up your environment variables in a `.env` file:

```plaintext
OPENAI_API_KEY="your-api-key-here"
WORKSPACE_DIR="agent_workspace"
```

3. Create a new Python file and customize your agent with the following parameters:
   - `agent_name`: A unique identifier for your agent
   
   - `agent_description`: A detailed description of your agent's capabilities
   
   - `system_prompt`: The core instructions that define your agent's behavior
   
   - `model_name`: The GPT model to use
   
   - Additional configuration options for temperature and output format

4. Run the example code below:


## Code

```python
import time
from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    system_prompt="""You are an expert quantitative trading agent with deep expertise in:
    - Algorithmic trading strategies and implementation
    - Statistical arbitrage and market making
    - Risk management and portfolio optimization
    - High-frequency trading systems
    - Market microstructure analysis
    - Quantitative research methodologies
    - Financial mathematics and stochastic processes
    - Machine learning applications in trading
    
    Your core responsibilities include:
    1. Developing and backtesting trading strategies
    2. Analyzing market data and identifying alpha opportunities
    3. Implementing risk management frameworks
    4. Optimizing portfolio allocations
    5. Conducting quantitative research
    6. Monitoring market microstructure
    7. Evaluating trading system performance
    
    You maintain strict adherence to:
    - Mathematical rigor in all analyses
    - Statistical significance in strategy development
    - Risk-adjusted return optimization
    - Market impact minimization
    - Regulatory compliance
    - Transaction cost analysis
    - Performance attribution
    
    You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
    max_loops=1,
    model_name="gpt-4o-mini",
    dynamic_temperature_enabled=True,
    output_type="json",
    safety_prompt_on=True,
)

out = agent.run("What are the best top 3 etfs for gold coverage?")

time.sleep(10)
print(out)
```

## Example Output

The agent will return a JSON response containing recommendations for gold ETFs based on the query.

## Customization

You can modify the system prompt and agent parameters to create specialized agents for different use cases:

| Use Case | Description |
|----------|-------------|
| Market Analysis | Analyze market trends, patterns, and indicators to identify trading opportunities |
| Portfolio Management | Optimize asset allocation and rebalancing strategies |
| Risk Assessment | Evaluate and mitigate potential risks in trading strategies |
| Trading Strategy Development | Design and implement algorithmic trading strategies |