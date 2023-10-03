from langchain.
from swarms.agents import OmniModalAgent



llm = OpenAIChat()

agent = OmniModalAgent(llm)

agent.run("Create a video of a swarm of fish")