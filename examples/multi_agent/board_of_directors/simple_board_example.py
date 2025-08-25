from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
)
from swarms.structs.agent import Agent

# Create simple agents for basic tasks
analyst = Agent(
    agent_name="Analyst",
    agent_description="Data analyst",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Content writer",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agents = [analyst, writer]

# Create swarm with default settings
board_swarm = BoardOfDirectorsSwarm(
    name="Simple_Board",
    agents=agents,
    verbose=False,
)

# Execute simple task
task = "Analyze current market trends and create a summary report."

result = board_swarm.run(task=task)

print(result)
