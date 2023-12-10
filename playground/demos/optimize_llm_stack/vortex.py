import os
import subprocess

from dotenv import load_dotenv

from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.utils.phoenix_handler import phoenix_trace_decorator

import modal

load_dotenv()

# Model
llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4",
    max_tokens=4000,
)

# Modal
stub = modal.Stub(name="swarms")


# Agent
@phoenix_trace_decorator(
    "This function is an agent and is traced by Phoenix."
)
@stub.function(gpu="any")
def agent(task: str):
    agent = Agent(
        llm=llm,
        max_loops=1,
    )
    out = agent.run(task=task)
    return out
