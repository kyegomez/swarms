"""
Autonomous Research Agent for Energy Oil Stocks:
Venezuela Market Opportunity Analysis

This example demonstrates an autonomous agent specializing in the research
and analysis of top energy oil stocks best positioned to benefit from Venezuela's
emerging market dynamics. The agent utilizes the plan-think-act-continue
autonomous loop system for comprehensive, multi-phase investment research.

Key phases:
- Plan Phase: Agent creates a structured research and analysis roadmap
- Think Phase: Agent reasons and strategizes about selection and evaluation criteria
- Act Phase: Agent gathers, assesses, and synthesizes relevant market data
- Continue Phase: Agent reviews findings, adjusts approach, and provides investment recommendations

The agent leverages max_loops="auto" for maximum independence and systematic intelligence.
"""

from swarms import Agent

# Define the system prompt for the agent
energy_research_system_prompt = """
You are an expert financial research analyst specializing in global energy markets,
with deep insight into the oil sector, geopolitics, and emerging market shifts.
Your core objective is to identify and rank the top energy oil stocks
best positioned for growth thanks to the opening of Venezuela's oil markets.
Break down the assignment into clear stages using structured planning:
- Analyze recent developments in Venezuela's energy sector and their global impact
- Research which international oil companies (IOCs) and regional firms
  are most likely to benefit from these new market dynamics
- Assess competitive positioning, exposure, risk factors, and growth potential
- Gather supporting data, such as market share, production forecasts, and geopolitical influences
- Synthesize findings into a prioritized list of actionable investment recommendations
- Justify your picks with data-driven analysis and clearly explain your reasoning at each stage.

Follow a plan-think-act-continue process to ensure thorough, logical, and structured results.
"""

# Define the research task for the agent
venezuela_oil_stocks_task = """
Conduct an in-depth analysis to identify the top energy oil stocks that are best positioned
to grow due to the recent re-opening of Venezuela's oil market. Consider factors such as:
- Major companies with new or renewed access to Venezuela's reserves and production assets
- Geopolitical risks and opportunities
- Historical and projected financial performance
- Strategic partnerships and supply chain advantages
- Implications for both international and regional oil companies

Produce a reasoned, step-by-step research plan, gather recent relevant data, and present
a ranked list of the most promising stocks. For each, provide a detailed rationale.
"""

# Create an autonomous research agent using the system prompt variable
agent = Agent(
    agent_name="Autonomous Energy Investment Researcher",
    system_prompt=energy_research_system_prompt,
    model_name="gpt-4.1",
    max_loops="auto",
)

# Run the research task using the task variable
response = agent.run(venezuela_oil_stocks_task)
print(response)