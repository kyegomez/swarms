from swarms import Agent
from swarm_models import OpenAIChat

agent = Agent(
    agent_name="API Requester",
    agent_description="This agent is responsible for making API requests.",
    system_prompt="You're a helpful API Requester agent. ",
    llm=OpenAIChat(),
    autosave=True,
    max_loops="auto",
    dashboard=True,
    interactive=True,
)


# Run the agent
out = agent.run("Create an api request to OpenAI in python.")
print(out)
