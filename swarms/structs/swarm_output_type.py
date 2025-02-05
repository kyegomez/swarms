import time
from typing import List
import uuid
from pydantic import BaseModel, Field


class AgentRespond(BaseModel):
    id: str = Field(default=uuid.uuid4().hex)
    timestamp: str = Field(default=time.time())
    agent_position: int = Field(description="Agent in swarm position")
    agent_name: str
    agent_response: str = Field(description="Agent response")


class SwarmOutput(BaseModel):
    id: str = Field(default=uuid.uuid4().hex)
    timestamp: str = Field(default=time.time())
    name: str = Field(description="Swarm name")
    description: str = Field(description="Swarm description")
    swarm_type: str = Field(description="Swarm type")
    agent_outputs: List[AgentRespond] = Field(
        description="List of agent responses"
    )
