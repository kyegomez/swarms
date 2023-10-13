from langchain.llms import OpenAIChat
from swarms.agents import OmniModalAgent


llm = OpenAIChat(model_name="gpt-4")

agent = OmniModalAgent(llm)

agent.run("Create a video of a swarm of fish")
