from swarms import Agent, SwarmRouter

researcher = Agent(
    agent_name="Research-Specialist",
    system_prompt="You research and gather factual information on topics.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Data-Analyst",
    system_prompt="You analyze data and identify patterns and insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

strategist = Agent(
    agent_name="Business-Strategist",
    system_prompt="You develop strategic recommendations based on analysis.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

swarm = SwarmRouter(
    name="Research-Analysis-Strategy-Swarm",
    agents=[researcher, analyst, strategist],
    swarm_type="RoundRobin",
    max_loops=1,
    verbose=True,
)

task = "Analyze the renewable energy market and provide strategic recommendations"
result = swarm.run(task)

print("Round Robin Swarm Result:")
print(result)
