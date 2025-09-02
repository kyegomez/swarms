from swarms import Agent
from swarms_tools import scrape_and_format_sync

agent = Agent(
    agent_name="Web Scraper Agent",
    model_name="gpt-4o-mini",
    tools=[scrape_and_format_sync],
    dynamic_context_window=True,
    dynamic_temperature_enabled=True,
    max_loops=1,
    system_prompt="You are a web scraper agent. You are given a URL and you need to scrape the website and return the data in a structured format. The format type should be full",
)

out = agent.run(
    "Scrape swarms.ai website and provide a full report of the company  does. The format type should be full."
)
print(out)
