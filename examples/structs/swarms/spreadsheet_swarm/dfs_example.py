import os

from swarms import Agent
from swarm_models import OpenAIChat
from examples.structs.swarms.experimental.dfs_search_swarm import DFSSwarm

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class for each agent
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize multiple agents
agent1 = Agent(
    agent_name="Agent-1",
    system_prompt="Analyze the financial components of a startup's stock incentives.",
    llm=model,
    # max_loops=2,
    # autosave=True,
    dynamic_temperature_enabled=True,
    verbose=True,
    streaming_on=True,
    user_name="swarms_corp",
)

agent2 = Agent(
    agent_name="Agent-2",
    system_prompt="Refine the analysis and identify any potential risks or benefits.",
    llm=model,
    # max_loops=2,
    # autosave=True,
    dynamic_temperature_enabled=True,
    verbose=True,
    streaming_on=True,
    user_name="swarms_corp",
)

# Add more agents as needed
# agent3 = ...
# agent4 = ...

# Create the swarm with the agents
dfs_swarm = DFSSwarm(agents=[agent1, agent2])

# Run the DFS swarm on a task
result = dfs_swarm.run(
    "Start with analyzing the financial components of a startup's stock incentives."
)
print("Final Result:", result)
