from swarms import Agent, MajorityVoting

# Initialize multiple agents with different specialties
agents = [
    Agent(
        agent_name="Financial-Analysis-Agent",
        agent_description="Personal finance advisor focused on market analysis",
        system_prompt="You are a financial advisor specializing in market analysis and investment opportunities.",
        max_loops=1,
        model_name="gpt-4o",
    ),
    Agent(
        agent_name="Risk-Assessment-Agent",
        agent_description="Risk analysis and portfolio management expert",
        system_prompt="You are a risk assessment expert focused on evaluating investment risks and portfolio diversification.",
        max_loops=1,
        model_name="gpt-4o",
    ),
    Agent(
        agent_name="Tech-Investment-Agent",
        agent_description="Technology sector investment specialist",
        system_prompt="You are a technology investment specialist focused on AI, emerging tech, and growth opportunities.",
        max_loops=1,
        model_name="gpt-4o",
    ),
]


consensus_agent = Agent(
    agent_name="Consensus-Agent",
    agent_description="Consensus agent focused on analyzing investment advice",
    system_prompt="You are a consensus agent focused on analyzing investment advice and providing a final answer.",
    max_loops=1,
    model_name="gpt-4o",
)

# Create majority voting system
majority_voting = MajorityVoting(
    name="Investment-Advisory-System",
    description="Multi-agent system for investment advice",
    agents=agents,
    verbose=True,
    consensus_agent=consensus_agent,
)

# Run the analysis with majority voting
result = majority_voting.run(
    task="Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown.",
)

print(result)
