from swarms.structs.message_pool import MessagePool
from swarms import Agent, OpenAIChat
from swarms.memory.chroma_db import ChromaDB


# Agents
agent1 = Agent(
    llm=OpenAIChat(system_prompt="You are a Minecraft player. What's your favorite building style?"),
    agent_name="Steve",
    agent_description="A Minecraft player agent",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

agent2 = Agent(
    llm=OpenAIChat(system_prompt="You are a Minecraft builder. What's your most impressive creation?"),
    agent_name="Bob",
    agent_description="A Minecraft builder agent",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

agent3 = Agent(
    llm=OpenAIChat(system_prompt="You are a Minecraft explorer. What's the most interesting place you've discovered?"),
    agent_name="Alex",
    agent_description="A Minecraft explorer agent",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

agent4 = Agent(
    llm=OpenAIChat(system_prompt="You are a Minecraft adventurer. What's the most dangerous situation you've been in?"),
    agent_name="Ender",
    agent_description="A Minecraft adventurer agent",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

moderator = Agent(
    llm=OpenAIChat(system_prompt="You are a Minecraft moderator. How do you handle conflicts between players?"),
    agent_name="Admin",
    agent_description="A Minecraft moderator agent",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

# Create a message pool
pool = MessagePool(
    moderator=moderator,
    agents=[agent1, agent2, agent3, agent4],
    turns=4,
    show_names=True,
    autosave=True,
)

# Add a message to the pool
pool.add(
    agent=agent1,
    content="Hello, agent2!",
    turn=1,
)


# Get all messages
out = pool.get_all_messages()
print(out)


# Get visible messages
messages = pool.get_visible_messages(agent=agent1, turn=1)
print(messages)

# Get visible messages
# pool.query("Hello, agent2!")
