from swarms import Agent, HierarchicalSwarm


# Initialize agents for a $50B portfolio analysis
agents = [
    Agent(
        agent_name="Sector-Financial-Analyst",
        agent_description="Senior financial analyst at BlackRock.",
        system_prompt="You are a financial analyst tasked with optimizing asset allocations for a $50B portfolio. Provide clear, quantitative recommendations for each sector.",
        max_loops=1,
        model_name="gpt-4.1",
        max_tokens=3000,
        streaming_on=True,
    ),
    Agent(
        agent_name="Sector-Risk-Analyst",
        agent_description="Expert risk management analyst.",
        system_prompt="You are a risk analyst responsible for advising on risk allocation within a $50B portfolio. Provide detailed insights on risk exposures for each sector.",
        max_loops=1,
        model_name="gpt-4.1",
        max_tokens=3000,
        streaming_on=True,
    ),
    Agent(
        agent_name="Tech-Sector-Analyst",
        agent_description="Technology sector analyst.",
        system_prompt="You are a tech sector analyst focused on capital and risk allocations. Provide data-backed insights for the tech sector.",
        max_loops=1,
        model_name="gpt-4.1",
        max_tokens=3000,
        streaming_on=True,
    ),
]

# Create hierarchical swarm system
hiearchical_swarm = HierarchicalSwarm(
    name="Sector-Investment-Advisory-System",
    description="System for sector analysis and optimal allocations.",
    agents=agents,
    max_loops=1,
    output_type="all",
    director_feedback_on=True,
)


result = hiearchical_swarm.run(
    task=(
        "Simulate the allocation of a $50B fund specifically for the pharmaceutical sector. "
        "Provide specific tickers (e.g., PFE, MRK, JNJ, LLY, BMY, etc.) and a clear rationale for why funds should be allocated to each company. "
        "Present a table showing each ticker, company name, allocation percentage, and allocation amount in USD. "
        "Include a brief summary of the overall allocation strategy and the reasoning behind the choices."
        "Only call the Sector-Financial-Analyst agent to do the analysis. Nobody else should do the analysis."
    )
)

print(result)
