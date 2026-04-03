from swarms import Agent, HierarchicalSwarm

agents = [
    Agent(
        agent_name="Research-Analyst",
        agent_description="Specializes in research and data gathering",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    ),
    Agent(
        agent_name="Data-Analyst",
        agent_description="Expert in quantitative analysis and pattern recognition",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    ),
    Agent(
        agent_name="Strategy-Consultant",
        agent_description="Specializes in strategic planning and recommendations",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    ),
]

swarm = HierarchicalSwarm(
    name="FullFeature-Swarm",
    agents=agents,
    max_loops=1,
    director_model_name="gpt-4.1",
    parallel_execution=False,  # agents run sequentially (one after the other)
    agent_as_judge=True,  # judge agent scores each output
    judge_agent_model_name="gpt-4.1",
)


result = swarm.run(
    task="Analyze AI infrastructure investment trends and recommend the top 3 opportunities for 2025."
)
print(result)
