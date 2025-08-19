from swarms.utils.agent_loader import load_agent_from_markdown

# Load the Finance Advisor agent from markdown
agent = load_agent_from_markdown("Finance_advisor.md")

# Use the agent to get financial advice
response = agent.run(
    "I have $10,000 to invest. What's a good strategy for a beginner?"
)