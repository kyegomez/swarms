"""
Boss selects what agent to use
B -> W1, W2, W3
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from swarms.utils.json_utils import str_to_json


class HierarchicalSwarm(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    agents: Optional[List[str]] = Field(
        None, title="List of agents in the hierarchical swarm"
    )
    task: Optional[str] = Field(
        None, title="Task to be done by the agents"
    )


all_agents = HierarchicalSwarm()

agents_schema = HierarchicalSwarm.model_json_schema()
agents_schema = str_to_json(agents_schema)
print(agents_schema)
