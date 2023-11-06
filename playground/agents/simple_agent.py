from swarms.agents.simple_agent import SimpleAgent
from swarms.structs import Flow
from swarms.models import OpenAIChat

api_key = ""

llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

# Initialize the flow
flow = Flow(
    llm=llm,
    max_loops=5,
)


agent = SimpleAgent(
    name="Optimus Prime",
    flow=flow,
    # Memory
)

out = agent.run("Generate a 10,000 word blog on health and wellness.")
print(out)
