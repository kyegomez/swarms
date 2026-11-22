from swarms import Agent, TeamFormationSwarm

specialists = [
    ("Market-Researcher", "You size markets and find competitors."),
    ("Data-Analyst", "You turn numbers into clear findings."),
    ("Product-Strategist", "You turn findings into product bets."),
    ("Copywriter", "You write crisp, persuasive prose."),
]

agents = [
    Agent(
        agent_name=name,
        agent_description=prompt,
        system_prompt=prompt,
        model_name="gpt-5.4",
        max_loops=1,
    )
    for name, prompt in specialists
]

swarm = TeamFormationSwarm(
    agents=agents,
    max_loops=2,
    max_team_size=2,
)

answer = swarm.run(
    "Write a one-page launch brief for an AI note-taking app for doctors."
)

for team in swarm.teams:
    members = ", ".join(agent.agent_name for agent in team.members)
    print(f"{team.name} ({team.subtask}): {members}")

print(answer)
