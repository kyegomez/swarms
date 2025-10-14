from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.prompts.moa_prompt import MOA_AGGREGATOR_SYSTEM_PROMPT
from swarms.structs.mixture_of_agents import MixtureOfAgents

# Initialize the equity analyst agents
equity_analyst_1 = Agent(
    agent_name="Equity-Analyst-1",
    agent_description="Equity research analyst focused on fundamental analysis",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
)

equity_analyst_2 = Agent(
    agent_name="Equity-Analyst-2",
    agent_description="Equity research analyst focused on technical analysis",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
)

equity_analyst_3 = Agent(
    agent_name="Equity-Analyst-3",
    agent_description="Equity research analyst focused on quantitative analysis and risk modeling",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
)


swarm = MixtureOfAgents(
    name="Equity-Research-Swarm",
    agents=[equity_analyst_1, equity_analyst_2, equity_analyst_3],
    output_type="dict",
    layers=1,
    aggregator_system_prompt=MOA_AGGREGATOR_SYSTEM_PROMPT,
)


out = swarm.run(
    task="Analyze Exchange-Traded Funds (ETFs) and stocks related to copper. Focus on fundamentals including supply/demand factors, production costs, major market participants, and recent price trends. Create a detailed analysis table in markdown.",
)

print(out)
