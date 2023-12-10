import os
from swarms.structs import Agent
from swarms.memory import WeaviateClient
from swarms.utils.phoenix_handler import phoenix_trace_decorator
from swarms.models.vllm import vLLM
from dotenv import load_dotenv

load_dotenv()


# Model
llm = vLLM()

# Weaviate
weaviate_client = WeaviateClient(
    http_host="localhost",
    http_port="8080",
    http_secure=False,
    grpc_host="localhost",
    grpc_port="8081",
    grpc_secure=False,
    auth_client_secret="YOUR_APIKEY",
    additional_headers={"X-OpenAI-Api-Key": "YOUR_OPENAI_APIKEY"},
    additional_config=None,  # You can pass additional configuration here
)


# Agent
@phoenix_trace_decorator
@
def agent(task: str):
    agent = Agent(
        llm = llm,
        max_loops=1,    
    )
    
    out = agent.run(task=task)
    return out