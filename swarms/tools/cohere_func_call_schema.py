from pydantic import BaseModel, Field
from typing import Dict


class ParameterDefinition(BaseModel):
    description: str = Field(
        ..., title="Description of the parameter"
    )
    type: str = Field(..., title="Type of the parameter")
    required: bool = Field(..., title="Is the parameter required?")


class CohereFuncSchema(BaseModel):
    name: str = Field(..., title="Name of the tool")
    description: str = Field(..., title="Description of the tool")
    parameter_definitions: Dict[str, ParameterDefinition] = Field(
        ..., title="Parameter definitions for the tool"
    )
