from swarms.agents.simple_agent import SimpleAgent
<<<<<<< HEAD
from swarms.structs import Agent
=======
from swarms.structs import Flow
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
from swarms.models import OpenAIChat

api_key = ""

llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

# Initialize the flow
<<<<<<< HEAD
flow = Agent(
=======
flow = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
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
