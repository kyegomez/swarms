import os
from swarms import OpenAIChat, Agent


# Create a chat instance
llm = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create an agent
agent = Agent(
    agent_name="GPT-3",
    llm=llm,
)
