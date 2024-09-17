from typing import List
from pydantic import BaseModel
from swarms.schemas.agent_step_schemas import Step


class Plan(BaseModel):
    steps: List[Step]

    class Config:
        orm_mode = True
