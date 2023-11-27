import os

from dotenv import load_dotenv

<<<<<<< HEAD
# Import the OpenAIChat model and the Agent struct
from swarms.models import OpenAIChat
from swarms.structs import Agent
=======
# Import the OpenAIChat model and the Flow struct
from swarms.models import OpenAIChat
from swarms.structs import Flow
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    openai_api_key=api_key,
)


## Initialize the workflow
<<<<<<< HEAD
flow = Agent(llm=llm, max_loops=1, dashboard=True)
=======
flow = Flow(llm=llm, max_loops=1, dashboard=True)
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4

# Run the workflow on a task
out = flow.run("Generate a 10,000 word blog on health and wellness.")
