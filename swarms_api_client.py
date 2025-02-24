from swarms.structs.swarms_api import (
    SwarmsAPIClient,
    SwarmRequest,
    AgentInput,
)
import os

agents = [
    AgentInput(
        agent_name="Medical Researcher",
        description="Conducts medical research and analysis",
        system_prompt="You are a medical researcher specializing in clinical studies.",
    ),
    AgentInput(
        agent_name="Medical Diagnostician",
        description="Provides medical diagnoses based on symptoms and test results",
        system_prompt="You are a medical diagnostician with expertise in identifying diseases.",
    ),
    AgentInput(
        agent_name="Pharmaceutical Expert",
        description="Advises on pharmaceutical treatments and drug interactions",
        system_prompt="You are a pharmaceutical expert knowledgeable about medications and their effects.",
    ),
]

swarm_request = SwarmRequest(
    name="Medical Swarm",
    description="A swarm for medical research and diagnostics",
    agents=agents,
    max_loops=1,
    swarm_type="ConcurrentWorkflow",
)

client = SwarmsAPIClient(api_key=os.getenv("SWARMS_API_KEY"))

response = client.create_swarm(swarm_request)

print(response)
