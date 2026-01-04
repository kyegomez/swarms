"""
SpreadsheetSwarm Example

This example demonstrates SpreadsheetSwarm for concurrent processing
of tasks with automatic CSV tracking of results and metadata.
"""

from swarms import Agent, SpreadSheetSwarm

market_researcher = Agent(
    agent_name="Market-Researcher",
    system_prompt="You are a market research analyst. Analyze market trends, competitors, and opportunities.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

financial_analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="You are a financial analyst. Analyze financial data, calculate metrics, and provide insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

risk_assessor = Agent(
    agent_name="Risk-Assessor",
    system_prompt="You are a risk assessment specialist. Identify and evaluate potential risks.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

swarm = SpreadSheetSwarm(
    name="Market-Analysis-Swarm",
    description="A swarm of specialized financial analysis agents",
    agents=[market_researcher, financial_analyst, risk_assessor],
    max_loops=1,
    autosave=True,
)

task = "What are the top 3 energy stocks to invest in for 2024? Provide detailed analysis."
result = swarm.run(task=task)

print("SpreadsheetSwarm Result:")
print(result)
