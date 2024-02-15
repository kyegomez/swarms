# Importing necessary modules
import os
from dotenv import load_dotenv
from swarms.agents.worker_agent import Worker
from swarms import OpenAIChat

# Loading environment variables from .env file
load_dotenv()

# Retrieving the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")

# Creating a Worker instance
worker = Worker(
    name="My Worker",
    role="Worker",
    human_in_the_loop=False,
    tools=[],
    temperature=0.5,
    llm=OpenAIChat(openai_api_key=api_key, openai_org_id=org_id, max_tokens=500,),
    verbose=True,
)

# Running the worker with a prompt
out = worker.run(
    "Hello, how are you? Create an image of how your are doing!"
)

# Printing the output
print(out)
