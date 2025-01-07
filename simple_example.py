from swarms import Agent

agent = Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    # interactive=True,
    streaming_on=True,
    encryption_key="tou",
    enable_transit_encryption=True,
    enable_rest_encryption=True,
)

print(agent.run("What are 5 hft algorithms"))
