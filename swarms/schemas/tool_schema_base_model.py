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
