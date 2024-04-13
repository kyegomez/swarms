# Description: This is an example of how to use the Agent class to run a multi-modal workflow
import os

from dotenv import load_dotenv

from swarms import Agent, GPT4VisionAPI

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = GPT4VisionAPI(
    openai_api_key=api_key,
    max_tokens=500,
)

# Initialize the language model
task = "What is the color of the object?"
img = "images/swarms.jpeg"

# Initialize the workflow
agent = Agent(
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=True,
    multi_modal=True,
)

# Run the workflow on a task
out = agent.run(task=task, img=img)
print(out)
