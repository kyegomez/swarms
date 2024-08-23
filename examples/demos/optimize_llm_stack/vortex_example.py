import os

from dotenv import load_dotenv

from swarms.models import OpenAIChat
from swarms.structs import Agent

# import modal

load_dotenv()

# Model
llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4",
    max_tokens=1000,
)

# Modal
# stub = modal.Stub(name="swarms")


# Agent
# @stub.function(gpu="any")
agent = Agent(
    llm=llm,
    max_loops=2,
    autosave=True,
    dashboard=True,
)
out = agent.run(
    "Generate a 5,000 word blog on how swarms of autonomous agents"
    " can be used to solve the world's problems."
)
print(out)
