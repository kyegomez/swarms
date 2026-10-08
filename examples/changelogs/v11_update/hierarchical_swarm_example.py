from swarms import Agent, HierarchicalSwarm

researcher = Agent(agent_name="Researcher", model_name="gpt-5.4")
analyst = Agent(agent_name="Analyst", model_name="gpt-5.4")
writer = Agent(agent_name="Writer", model_name="claude-sonnet-4-6")

swarm = HierarchicalSwarm(
    name="ResearchDirector",
    director_model_name="gpt-5.4",
    agents=[researcher, analyst, writer],
    parallel_execution=True,
    agent_as_judge=True,
    director_temperature=1.0,
    planning_enabled=False,
)

result = swarm.run(
    "Produce a competitive analysis of the top 5 cloud providers"
)
# Console prints a rich Tree panel showing which agent gets which subtask
print(result)
