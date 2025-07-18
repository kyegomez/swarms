from swarms.structs.heavy_swarm import HeavySwarm


swarm = HeavySwarm(
    worker_model_name="gpt-4o-mini",
    show_dashboard=False,
    question_agent_model_name="gpt-4.1",
    loops_per_agent=1,
)


out = swarm.run(
    "Identify the top 3 energy sector ETFs listed on US exchanges that offer the highest potential for growth over the next 3-5 years. Focus specifically on funds with significant exposure to companies in the nuclear, natural gas, or oil industries. For each ETF, provide the rationale for its selection, recent performance metrics, sector allocation breakdown, and any notable holdings related to nuclear, gas, or oil. Exclude broad-based energy ETFs that do not have a clear emphasis on these sub-sectors."
)

print(out)
