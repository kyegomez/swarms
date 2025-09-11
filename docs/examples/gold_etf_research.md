# Gold ETF Research with HeavySwarm

This example demonstrates how to use HeavySwarm to create a specialized research team that analyzes and compares gold ETFs using web search capabilities. The HeavySwarm orchestrates multiple agents to conduct comprehensive research and provide structured investment recommendations.

## Install

```bash
pip3 install -U swarms swarms-tools
```

## Environment Setup

```bash
EXA_API_KEY="your_exa_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

## Code

```python
from swarms import HeavySwarm
from swarms_tools import exa_search

# Initialize the HeavySwarm for gold ETF research
swarm = HeavySwarm(
    name="Gold ETF Research Team",
    description="A team of agents that research the best gold ETFs",
    worker_model_name="claude-sonnet-4-20250514",
    show_dashboard=True,
    question_agent_model_name="gpt-4.1",
    loops_per_agent=1,
    agent_prints_on=False,
    worker_tools=[exa_search],
    random_loops_per_agent=True,
)

# Define the research task
prompt = (
    "Find the best 3 gold ETFs. For each ETF, provide the ticker symbol, "
    "full name, current price, expense ratio, assets under management, and "
    "a brief explanation of why it is considered among the best. Present the information "
    "in a clear, structured format suitable for investors. Scrape the data from the web. "
)

# Execute the research
out = swarm.run(prompt)
print(out)
```

## Conclusion

This example demonstrates how HeavySwarm can be used to create specialized research teams for financial analysis. By leveraging multiple agents with web search capabilities, you can build powerful systems that provide comprehensive, real-time investment research and recommendations. The pattern can be easily adapted for various financial research tasks including stock analysis, sector research, and portfolio optimization.
