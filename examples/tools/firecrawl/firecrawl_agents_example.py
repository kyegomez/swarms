from swarms_tools import crawl_entire_site_firecrawl

from swarms import Agent

agent = Agent(
    agent_name="Marketing Copy Improver",
    model_name="gpt-4.1",
    tools=[crawl_entire_site_firecrawl],
    dynamic_context_window=True,
    dynamic_temperature_enabled=True,
    max_loops=1,
    system_prompt=(
        "You are a world-class marketing copy improver. "
        "Given a website URL, your job is to crawl the entire site, analyze all marketing copy, "
        "and rewrite it to maximize clarity, engagement, and conversion. "
        "Return the improved marketing copy in a structured, easy-to-read format. "
        "Be concise, persuasive, and ensure the tone matches the brand. "
        "Highlight key value propositions and calls to action."
    ),
)

out = agent.run(
    "Crawl 2-3 pages of swarms.ai and improve the marketing copy found on those pages. Return the improved copy in a structured format."
)
print(out)
