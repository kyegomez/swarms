import os

from dotenv import load_dotenv

from swarms import Agent, OpenAIChat

# Load environment variables
load_dotenv()

# Create a chat instance
llm = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create an agent
agent = Agent(
    agent_name="GPT-3",
    llm=llm,
)
