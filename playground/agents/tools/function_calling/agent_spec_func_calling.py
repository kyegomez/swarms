import json
import os
from swarms.models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field
from typing import List


class AgentSpec(BaseModel):
    agent_name: str = Field(
        ...,
        description="The name of the agent",
    )
    system_prompt: str = Field(
        ...,
        description="The system prompt for the agent",
    )
    agent_description: str = Field(
        ...,
        description="The description of the agent",
    )
    max_tokens: int = Field(
        ...,
        description="The maximum number of tokens to generate in the API response",
    )
    temperature: float = Field(
        ...,
        description="A parameter that controls the randomness of the generated text",
    )
    context_window: int = Field(
        ...,
        description="The context window for the agent",
    )
    model_name: str = Field(
        ...,
        description="The model name for the agent from huggingface",
    )


class SwarmSpec(BaseModel):
    multiple_agents: List[AgentSpec] = Field(
        ...,
        description="The list of agents in the swarm",
    )


# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're an agent creator, you're purpose is to create an agent with the user provided specifications. Think of relevant names, descriptions, and context windows for the agent. You need to provide the name of the agent, the system prompt for the agent, the description of the agent, the maximum number of tokens to generate in the API response, the temperature for the agent, the context window for the agent, and the model name for the agent from huggingface.",
    max_tokens=3000,
    temperature=0.8,
    base_model=SwarmSpec,
    parallel_tool_calls=False,
)


# The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
out = model.run(
    "Create a swarm of agents to generate social media posts. Each agent should have it's own social media"
)


# Define the folder and file name
folder_name = "agent_workspace"
file_name = "agent_output.json"

# Check if the folder exists, if not, create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Write the output to a JSON file
with open(os.path.join(folder_name, file_name), "w") as f:
    json.dump(out, f)
