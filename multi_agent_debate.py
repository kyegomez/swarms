import os

from dotenv import load_dotenv

from swarms.models import OpenAIChat
from swarms.structs import Flow
from swarms.swarms.multi_agent_collab import MultiAgentCollaboration

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    openai_api_key=api_key,
)


## Initialize the workflow
flow = Flow(llm=llm, max_loops=1, dashboard=True)
flow2 = Flow(llm=llm, max_loops=1, dashboard=True)
flow3 = Flow(llm=llm, max_loops=1, dashboard=True)


swarm = MultiAgentCollaboration(
    agents=[flow, flow2, flow3],
    max_iters=4,
)

swarm.run("Generate a 10,000 word blog on health and wellness.")
