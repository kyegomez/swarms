from swarms import Agent
from swarms.structs import MultiAgentCollaboration

# Agents will directly initialize their language models

# Define collaborating agents
planner = Agent(
    agent_name="Planner",
    system_prompt="Break the objective into clear steps.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="Use the plan to craft the final answer.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Initialize the collaborative workflow
swarm = MultiAgentCollaboration(
    agents=[planner, writer],
    max_loops=4,
)

# Kick off the collaboration
swarm.inject("Manager", "Produce a short overview of reinforcement learning.")

result = swarm.run("Begin")
print(result)

