import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms.models import OpenAIChat
from swarms.structs import Agent

<<<<<<< HEAD
# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    model_name="gpt-4",
    openai_api_key=api_key,
    max_tokens=1000,
=======
# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
>>>>>>> 4ae59df8 (tools fix, parse docs, inject tools docs into prompts, and attempt to execute tools, display markdown)
)

## Initialize the workflow
<<<<<<< HEAD
agent = Agent(
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=True,
)

# Run the workflow on a task
out = agent.run("Generate a 10,000 word blog on health and wellness.")
print(out)
=======
flow = Flow(llm=llm, max_loops=1, dashboard=True)

# Run the workflow on a task
out = flow.run("Generate a 10,000 word blog on health and wellness.")

>>>>>>> 4ae59df8 (tools fix, parse docs, inject tools docs into prompts, and attempt to execute tools, display markdown)
