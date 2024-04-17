import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.autoscaler import AutoScaler

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
agent = Agent(llm=llm, max_loops=1, dashboard=True)


# Load the autoscaler
autoscaler = AutoScaler(
    initial_agents=2,
    scale_up_factor=1,
    idle_threshold=0.2,
    busy_threshold=0.7,
    agents=[agent],
    autoscale=True,
    min_agents=1,
    max_agents=5,
    custom_scale_strategy=None,
)
print(autoscaler)

# Run the workflow on a task
out = autoscaler.run(
    agent.id, "Generate a 10,000 word blog on health and wellness."
)
print(out)
