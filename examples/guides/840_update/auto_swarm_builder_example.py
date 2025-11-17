import json
from swarms import AutoSwarmBuilder

swarm = AutoSwarmBuilder(
    name="My Swarm",
    description="A swarm of agents",
    verbose=True,
    max_loops=1,
    model_name="gpt-4o-mini",
    execution_type="return-agents",
)

out = swarm.run(
    task="Create an accounting team to analyze crypto transactions, there must be 5 agents in the team with extremely extensive prompts. Make the prompts extremely detailed and specific and long and comprehensive. Make sure to include all the details of the task in the prompts."
)

print(json.dumps(out, indent=4))
