from swarms.utils import load_agent_from_markdown

agent = load_agent_from_markdown("finance_advisor.md")

agent.run(
    task="Analyze the financial market trends for 2023."
)