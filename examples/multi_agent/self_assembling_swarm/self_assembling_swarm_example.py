from swarms import Agent, SelfAssemblingSwarm

agents = [
    Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name="gpt-5.4-mini",
        max_loops=1,
        print_on=True,
    )
    for name, prompt in [
        ("Researcher", "You find facts and sources."),
        ("Analyst", "You compare options and weigh trade-offs."),
        ("Writer", "You write clear, structured prose."),
        ("Critic", "You find errors and gaps."),
    ]
]

swarm = SelfAssemblingSwarm(agents=agents, max_loops=1, verbose=True)

print(swarm.run("Compare PostgreSQL and MongoDB for a new SaaS app."))
print(swarm.harness["teams"])
