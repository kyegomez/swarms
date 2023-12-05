from swarms.agents.simple_agent import SimpleAgent
from swarms.structs import Agent
from swarms.models import OpenAIChat

api_key = ""

llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

# Initialize the agent
agent = Agent(
    llm=llm,
    max_loops=5,
)


agent = SimpleAgent(
    name="Optimus Prime",
    agent=agent,
    # Memory
)

out = agent.run("Generate a 10,000 word blog on health and wellness.")
print(out)
