from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
import json

swarm = AutoSwarmBuilder(
    name="My Swarm",
    description="A swarm of agents",
    verbose=True,
    max_loops=1,
    return_agents=True,
    model_name="gpt-4.1",
)

print(
    json.dumps(
        swarm.run(
            task="Create an accounting team to analyze crypto transactions, there must be 5 agents in the team with extremely extensive prompts. Make the prompts extremely detailed and specific and long and comprehensive. Make sure to include all the details of the task in the prompts."
        ),
        indent=4,
    )
)
