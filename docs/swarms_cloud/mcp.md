# Swarms API as MCP

- Launch MCP server as a tool
- Put `SWARMS_API_KEY` in `.env`
- Client side code below


## Server Side

```python
# server.py
from datetime import datetime
import os
from typing import Any, Dict, List, Optional

import requests
import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from swarms import SwarmType
from dotenv import load_dotenv

load_dotenv()

class AgentSpec(BaseModel):
    agent_name: Optional[str] = Field(
        description="The unique name assigned to the agent, which identifies its role and functionality within the swarm.",
    )
    description: Optional[str] = Field(
        description="A detailed explanation of the agent's purpose, capabilities, and any specific tasks it is designed to perform.",
    )
    system_prompt: Optional[str] = Field(
        description="The initial instruction or context provided to the agent, guiding its behavior and responses during execution.",
    )
    model_name: Optional[str] = Field(
        default="gpt-4o-mini",
        description="The name of the AI model that the agent will utilize for processing tasks and generating outputs. For example: gpt-4o, gpt-4o-mini, openai/o3-mini",
    )
    auto_generate_prompt: Optional[bool] = Field(
        default=False,
        description="A flag indicating whether the agent should automatically create prompts based on the task requirements.",
    )
    max_tokens: Optional[int] = Field(
        default=8192,
        description="The maximum number of tokens that the agent is allowed to generate in its responses, limiting output length.",
    )
    temperature: Optional[float] = Field(
        default=0.5,
        description="A parameter that controls the randomness of the agent's output; lower values result in more deterministic responses.",
    )
    role: Optional[str] = Field(
        default="worker",
        description="The designated role of the agent within the swarm, which influences its behavior and interaction with other agents.",
    )
    max_loops: Optional[int] = Field(
        default=1,
        description="The maximum number of times the agent is allowed to repeat its task, enabling iterative processing if necessary.",
    )
    # New fields for RAG functionality
    rag_collection: Optional[str] = Field(
        None,
        description="The Qdrant collection name for RAG functionality. If provided, this agent will perform RAG queries.",
    )
    rag_documents: Optional[List[str]] = Field(
        None,
        description="Documents to ingest into the Qdrant collection for RAG. (List of text strings)",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="A dictionary of tools that the agent can use to complete its task.",
    )


class AgentCompletion(BaseModel):
    """
    Configuration for a single agent that works together as a swarm to accomplish tasks.
    """

    agent: AgentSpec = Field(
        ...,
        description="The agent to run.",
    )
    task: Optional[str] = Field(
        ...,
        description="The task to run.",
    )
    img: Optional[str] = Field(
        None,
        description="An optional image URL that may be associated with the swarm's task or representation.",
    )
    output_type: Optional[str] = Field(
        "list",
        description="The type of output to return.",
    )


class AgentCompletionResponse(BaseModel):
    """
    Response from an agent completion.
    """

    agent_id: str = Field(
        ...,
        description="The unique identifier for the agent that completed the task.",
    )
    agent_name: str = Field(
        ...,
        description="The name of the agent that completed the task.",
    )
    agent_description: str = Field(
        ...,
        description="The description of the agent that completed the task.",
    )
    messages: Any = Field(
        ...,
        description="The messages from the agent completion.",
    )

    cost: Dict[str, Any] = Field(
        ...,
        description="The cost of the agent completion.",
    )


class Agents(BaseModel):
    """Configuration for a collection of agents that work together as a swarm to accomplish tasks."""

    agents: List[AgentSpec] = Field(
        description="A list containing the specifications of each agent that will participate in the swarm, detailing their roles and functionalities."
    )


class ScheduleSpec(BaseModel):
    scheduled_time: datetime = Field(
        ...,
        description="The exact date and time (in UTC) when the swarm is scheduled to execute its tasks.",
    )
    timezone: Optional[str] = Field(
        "UTC",
        description="The timezone in which the scheduled time is defined, allowing for proper scheduling across different regions.",
    )


class SwarmSpec(BaseModel):
    name: Optional[str] = Field(
        None,
        description="The name of the swarm, which serves as an identifier for the group of agents and their collective task.",
        max_length=100,
    )
    description: Optional[str] = Field(
        None,
        description="A comprehensive description of the swarm's objectives, capabilities, and intended outcomes.",
    )
    agents: Optional[List[AgentSpec]] = Field(
        None,
        description="A list of agents or specifications that define the agents participating in the swarm.",
    )
    max_loops: Optional[int] = Field(
        default=1,
        description="The maximum number of execution loops allowed for the swarm, enabling repeated processing if needed.",
    )
    swarm_type: Optional[SwarmType] = Field(
        None,
        description="The classification of the swarm, indicating its operational style and methodology.",
    )
    rearrange_flow: Optional[str] = Field(
        None,
        description="Instructions on how to rearrange the flow of tasks among agents, if applicable.",
    )
    task: Optional[str] = Field(
        None,
        description="The specific task or objective that the swarm is designed to accomplish.",
    )
    img: Optional[str] = Field(
        None,
        description="An optional image URL that may be associated with the swarm's task or representation.",
    )
    return_history: Optional[bool] = Field(
        True,
        description="A flag indicating whether the swarm should return its execution history along with the final output.",
    )
    rules: Optional[str] = Field(
        None,
        description="Guidelines or constraints that govern the behavior and interactions of the agents within the swarm.",
    )
    schedule: Optional[ScheduleSpec] = Field(
        None,
        description="Details regarding the scheduling of the swarm's execution, including timing and timezone information.",
    )
    tasks: Optional[List[str]] = Field(
        None,
        description="A list of tasks that the swarm should complete.",
    )
    messages: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="A list of messages that the swarm should complete.",
    )
    # rag_on: Optional[bool] = Field(
    #     None,
    #     description="A flag indicating whether the swarm should use RAG.",
    # )
    # collection_name: Optional[str] = Field(
    #     None,
    #     description="The name of the collection to use for RAG.",
    # )
    stream: Optional[bool] = Field(
        False,
        description="A flag indicating whether the swarm should stream its output.",
    )


class SwarmCompletionResponse(BaseModel):
    """
    Response from a swarm completion.
    """

    status: str = Field(..., description="The status of the swarm completion.")
    swarm_name: str = Field(..., description="The name of the swarm.")
    description: str = Field(..., description="Description of the swarm.")
    swarm_type: str = Field(..., description="The type of the swarm.")
    task: str = Field(
        ..., description="The task that the swarm is designed to accomplish."
    )
    output: List[Dict[str, Any]] = Field(
        ..., description="The output generated by the swarm."
    )
    number_of_agents: int = Field(
        ..., description="The number of agents involved in the swarm."
    )
    # "input_config": Optional[Dict[str, Any]] = Field(None, description="The input configuration for the swarm.")


BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"


# Create an MCP server
mcp = FastMCP("swarms-api")


# Add an addition tool
@mcp.tool(name="swarm_completion", description="Run a swarm completion.")
def swarm_completion(swarm: SwarmSpec) -> Dict[str, Any]:
    api_key = os.getenv("SWARMS_API_KEY")
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    payload = swarm.model_dump()

    response = requests.post(f"{BASE_URL}/v1/swarm/completions", json=payload, headers=headers)
    
    return response.json()

@mcp.tool(name="swarms_available", description="Get the list of available swarms.")
async def swarms_available() -> Any:
    """
    Get the list of available swarms.
    """
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/models/available", headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()


if __name__ == "__main__":
    mcp.run(transport="sse")
```

## Client side

- Call the tool with it's name and the payload config

```python
import asyncio
from fastmcp import Client

swarm_config = {
    "name": "Simple Financial Analysis",
    "description": "A swarm to analyze financial data",
    "agents": [
        {
            "agent_name": "Data Analyzer",
            "description": "Looks at financial data",
            "system_prompt": "Analyze the data.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 1000,
            "temperature": 0.5,
            "auto_generate_prompt": False,
        },
        {
            "agent_name": "Risk Analyst",
            "description": "Checks risk levels",
            "system_prompt": "Evaluate the risks.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 1000,
            "temperature": 0.5,
            "auto_generate_prompt": False,
        },
        {
            "agent_name": "Strategy Checker",
            "description": "Validates strategies",
            "system_prompt": "Review the strategy.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 1000,
            "temperature": 0.5,
            "auto_generate_prompt": False,
        },
    ],
    "max_loops": 1,
    "swarm_type": "SequentialWorkflow",
    "task": "Analyze the financial data and provide insights.",
    "return_history": False,  # Added required field
    "stream": False,  # Added required field
    "rules": None,  # Added optional field
    "img": None,  # Added optional field
}


async def fetch_weather_and_resource():
    """Connect to a server over SSE and fetch available swarms."""

    async with Client(
        transport="http://localhost:8000/sse"
        # SSETransport(
        #     url="http://localhost:8000/sse",
        #     headers={"x_api_key": os.getenv("SWARMS_API_KEY"), "Content-Type": "application/json"}
        # )
    ) as client:
        # Basic connectivity testing
        # print("Ping check:", await client.ping())
        # print("Available tools:", await client.list_tools())
        # print("Swarms available:", await client.call_tool("swarms_available", None))
        result = await client.call_tool("swarm_completion", {"swarm": swarm_config})
        print("Swarm completion:", result)


# Execute the function
if __name__ == "__main__":
    asyncio.run(fetch_weather_and_resource())


```