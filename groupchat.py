from swarms.structs import Flow
from swarms.models import OpenAIChat
from swarms.swarms.groupchat import GroupChat
from swarms.agents import SimpleAgent

api_key = ""

llm = OpenAIChat(
    openai_api_key=api_key,
)

agent1 = SimpleAgent("Captain Price", Flow(llm=llm, max_loops=4))
agent2 = SimpleAgent("John Mactavis", Flow(llm=llm, max_loops=4))

# Create a groupchat with the 2 agents
chat = GroupChat([agent1, agent2])

# Assign duties to the agents
chat.assign_duty(agent1.name, "Buy the groceries")
chat.assign_duty(agent2.name, "Clean the house")

# Initate a chat
response = chat.run("Captain Price", "Hello, how are you John?")
print(response)
