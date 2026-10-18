from pathlib import Path

from swarms import load_agents_from_markdown

agents = load_agents_from_markdown(
    [str(Path(__file__).resolve().with_name("finance_advisor.md"))]
)

# Use the agent
response = agents[0].run(
    "I have $100k to invest. I want to hedge my bets on the energy companies that will benefit from the AI revoltion"
    "What are the top 4 stocks to invest in?"
)
