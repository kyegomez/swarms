import time
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field


class AssistantRequest(BaseModel):
    model: str = Field(
        ...,
        description="ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.",
    )
    name: Optional[Union[str, None]] = Field(
        None,
        description="The name of the assistant. The maximum length is 256 characters.",
    )
    description: Optional[Union[str, None]] = Field(
        None,
        description="The description of the assistant. The maximum length is 512 characters.",
    )
    instructions: Optional[Union[str, None]] = Field(
        None,
        description="The system instructions that the assistant uses. The maximum length is 256,000 characters.",
    )
    tools: Optional[List[Dict[str, Union[str, None]]]] = Field(
        default_factory=list,
        description="A list of tool enabled on the assistant. There can be a maximum of 128 tools per assistant. Tools can be of types code_interpreter, file_search, or function.",
    )
    tool_resources: Optional[Union[Dict, None]] = Field(
        None,
        description="A set of resources that are used by the assistant's tools. The resources are specific to the type of tool. For example, the code_interpreter tool requires a list of file IDs, while the file_search tool requires a list of vector store IDs.",
    )
    metadata: Optional[Dict[str, Union[str, None]]] = Field(
        default_factory=dict,
        description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maximum of 512 characters long.",
    )
    temperature: Optional[Union[float, None]] = Field(
        1.0,
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
    )
    top_p: Optional[Union[float, None]] = Field(
        1.0,
        description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.",
    )
    response_format: Optional[Union[str, Dict[str, Union[str, None]]]] = (
        Field(
            None,
            description="Specifies the format that the model must output. Compatible with GPT-4o, GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106. Setting to { 'type': 'json_object' } enables JSON mode, which guarantees the message the model generates is valid JSON.",
        )
    )


class AssistantResponse(BaseModel):
    id: str = Field(
        ..., description="The unique identifier for the assistant."
    )
    object: str = Field(
        ..., description="The type of object returned, e.g., 'assistant'."
    )
    created_at: int = Field(
        time.time(),
        description="The timestamp (in seconds since Unix epoch) when the assistant was created.",
    )
    name: Optional[Union[str, None]] = Field(
        None,
        description="The name of the assistant. The maximum length is 256 characters.",
    )
    description: Optional[Union[str, None]] = Field(
        None,
        description="The description of the assistant. The maximum length is 512 characters.",
    )
    model: str = Field(
        ..., description="ID of the model used by the assistant."
    )
    instructions: Optional[Union[str, None]] = Field(
        None,
        description="The system instructions that the assistant uses. The maximum length is 256,000 characters.",
    )
    tools: Optional[List[Dict[str, Union[str, None]]]] = Field(
        default_factory=list,
        description="A list of tool enabled on the assistant.",
    )
    metadata: Optional[Dict[str, Union[str, None]]] = Field(
        default_factory=dict,
        description="Set of 16 key-value pairs that can be attached to an object.",
    )
    temperature: float = Field(
        1.0, description="The sampling temperature used by the assistant."
    )
    top_p: float = Field(
        1.0,
        description="The nucleus sampling value used by the assistant.",
    )
    response_format: Optional[Union[str, Dict[str, Union[str, None]]]] = (
        Field(
            None,
            description="Specifies the format that the model outputs.",
        )
    )
