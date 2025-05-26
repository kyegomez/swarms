from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm


# Initialize agents for a $50B portfolio analysis
agents = [
    Agent(
        agent_name="Sector-Financial-Analyst",
        agent_description="Senior financial analyst at BlackRock.",
        system_prompt="You are a financial analyst tasked with optimizing asset allocations for a $50B portfolio. Provide clear, quantitative recommendations for each sector.",
        max_loops=1,
        model_name="groq/deepseek-r1-distill-qwen-32b",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Sector-Risk-Analyst",
        agent_description="Expert risk management analyst.",
        system_prompt="You are a risk analyst responsible for advising on risk allocation within a $50B portfolio. Provide detailed insights on risk exposures for each sector.",
        max_loops=1,
        model_name="groq/deepseek-r1-distill-qwen-32b",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Tech-Sector-Analyst",
        agent_description="Technology sector analyst.",
        system_prompt="You are a tech sector analyst focused on capital and risk allocations. Provide data-backed insights for the tech sector.",
        max_loops=1,
        model_name="groq/deepseek-r1-distill-qwen-32b",
        max_tokens=3000,
    ),
]

# Create hierarchical swarm system
majority_voting = HierarchicalSwarm(
    name="Sector-Investment-Advisory-System",
    description="System for sector analysis and optimal allocations.",
    agents=agents,
    # director=director_agent,
    max_loops=1,
    output_type="dict",
)

# Run the analysis
result = majority_voting.run(
    task="Evaluate market sectors and determine optimal allocation for a $50B portfolio. Include a detailed table of allocations, risk assessments, and a consolidated strategy."
)

print(result)
