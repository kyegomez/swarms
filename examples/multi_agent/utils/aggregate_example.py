from swarms.structs.agent import Agent
from swarms.structs.ma_blocks import aggregate


agents = [
    Agent(
        agent_name="Sector-Financial-Analyst",
        agent_description="Senior financial analyst at BlackRock.",
        system_prompt="You are a financial analyst tasked with optimizing asset allocations for a $50B portfolio. Provide clear, quantitative recommendations for each sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Sector-Risk-Analyst",
        agent_description="Expert risk management analyst.",
        system_prompt="You are a risk analyst responsible for advising on risk allocation within a $50B portfolio. Provide detailed insights on risk exposures for each sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Tech-Sector-Analyst",
        agent_description="Technology sector analyst.",
        system_prompt="You are a tech sector analyst focused on capital and risk allocations. Provide data-backed insights for the tech sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
]

out = aggregate(
    workers=agents,
    task="What is the best sector to invest in?",
    type="all",
)

print(out)
