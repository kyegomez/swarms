from swarms import OpenAIChat
from swarms.structs.agent import Agent
from swarms.structs.message_pool import MessagePool

agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
agent2 = Agent(llm=OpenAIChat(), agent_name="agent2")
agent3 = Agent(llm=OpenAIChat(), agent_name="agent3")

moderator = Agent(agent_name="moderator")
agents = [agent1, agent2, agent3]
message_pool = MessagePool(agents=agents, moderator=moderator, turns=5)
message_pool.add(agent=agent1, content="Hello, agent2!", turn=1)
message_pool.add(agent=agent2, content="Hello, agent1!", turn=1)
message_pool.add(agent=agent3, content="Hello, agent1!", turn=1)
message_pool.get_all_messages()
message_pool.get_visible_messages(agent=agent1, turn=1)
message_pool.get_visible_messages(agent=agent2, turn=1)
