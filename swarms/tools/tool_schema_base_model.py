from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PropertySchema(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, "PropertySchema"]] = None
    required: Optional[List[str]] = None


class ParameterSchema(BaseModel):
    type: str
    properties: Dict[str, PropertySchema]
    required: Optional[List[str]] = None


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: ParameterSchema


class Tool(BaseModel):
    type: str
    function: FunctionDefinition


class ToolSet(BaseModel):
    tools: List[Tool]


# model = ToolSet(
#     tools=[
#         Tool(
#             type="function",
#             function=FunctionDefinition(
#                 name="test",
#                 description="test",
#                 parameters=ParameterSchema(
#                     type="object",
#                     properties={
#                         "weather_tool": PropertySchema(
#                             type="string",
#                             description="Get the weather in a given location",
#                         )
#                     },
#                     required=["weather_tool"],
#                 ),
#             ),
#         ),
#     ]
# )

# print(model.model_dump_json(indent=4))
