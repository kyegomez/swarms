from swarms import OmniModalAgent, OpenAIChat

llm = OpenAIChat()

agent = OmniModalAgent(llm)

agent.run("Create a video of a swarm of fish")