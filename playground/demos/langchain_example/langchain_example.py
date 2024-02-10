import os
from dotenv import load_dotenv
from swarms import Agent
from langchain.llms import OpenAIChat

# Loading environment variables from .env file
load_dotenv()

# Initialize the model
llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=1000,
)

# Initialize the agent
agent = Agent(
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
)

# Run the workflow on a task
agent.run("Generate a 10,000 word blog on health and wellness.")
