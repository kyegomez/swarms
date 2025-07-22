from swarms.structs.heavy_swarm import HeavySwarm


swarm = HeavySwarm(
    worker_model_name="claude-3-5-sonnet-20240620",
    show_dashboard=True,
    question_agent_model_name="gpt-4.1",
    loops_per_agent=1,
)


out = swarm.run(
    "Provide 3 publicly traded biotech companies that are currently trading below their cash value. For each company identified, provide available data or projections for the next 6 months, including any relevant financial metrics, upcoming catalysts, or events that could impact valuation. Present your findings in a clear, structured format. Be very specific and provide their ticker symbol, name, and the current price, cash value, and the percentage difference between the two."
)

print(out)
