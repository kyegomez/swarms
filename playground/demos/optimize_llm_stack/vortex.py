import os
import subprocess

from dotenv import load_dotenv

from swarms.memory import WeaviateClient
from swarms.models.vllm import vLLM
from swarms.structs import Agent
from swarms.utils.phoenix_handler import phoenix_trace_decorator

try:
    import modal
except ImportError:
    print(f"modal not installed, please install it with `pip install modal`")
    subprocess.run(["pip", "install", "modal"])



load_dotenv()

# Model
llm = vLLM()

# Modal
stub = modal.Stub(name="swarms")


# Agent
@phoenix_trace_decorator
@stub.function(gpu="any")
def agent(task: str):
    agent = Agent(
        llm = llm,
        max_loops=1,    
    )
    out = agent.run(task=task)
    return out