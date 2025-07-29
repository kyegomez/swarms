from swarms.structs.heavy_swarm import HeavySwarm

swarm = HeavySwarm(
    worker_model_name="claude-3-5-sonnet-20240620",
    show_dashboard=True,
    question_agent_model_name="gpt-4.1",
    loops_per_agent=1,
)

out = swarm.run(
    "List the top 5 gold and commodity ETFs with the best performance and lowest expense ratios. For each ETF, provide the ticker symbol, full name, current price, 1-year and 5-year returns (in %), and the expense ratio. Also, specify which major brokerages (e.g., Fidelity, Schwab, Vanguard, E*TRADE) offer these ETFs for purchase. Present your findings in a clear, structured table."
)

print(out)
