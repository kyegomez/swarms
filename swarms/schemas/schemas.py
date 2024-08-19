from __future__ import annotations

import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field
from swarms_cloud.schema.agent_api_schemas import (
    AgentChatCompletionResponse,
)


class Step(BaseModel):
    step_id: str = Field(
        uuid.uuid4().hex,
        description="The ID of the task step.",
        examples=["6bb1801a-fd80-45e8-899a-4dd723cc602e"],
    )
    time: float = Field(
        time.time(),
        description="The time taken to complete the task step.",
    )
    response: AgentChatCompletionResponse = Field(
        ...,
        description="The response from the agent.",
    )


class ManySteps(BaseModel):
    agent_id: Optional[str] = Field(
        ...,
        description="The ID of the agent.",
        examples=["financial-agent-1"],
    )
    agent_name: Optional[str] = Field(
        ...,
        description="The ID of the agent.",
        examples=["financial-agent-1"],
    )
    task: Optional[str] = Field(
        ...,
        description="The name of the task.",
        examples=["Write to file"],
    )
    number_of_steps: Optional[int] = Field(
        ...,
        description="The number of steps in the task.",
        examples=[3],
    )
    run_id: Optional[str] = Field(
        uuid.uuid4().hex,
        description="The ID of the task this step belongs to.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    steps: List[Step] = Field(
        ...,
        description="A list of task steps.",
    )
    full_history: Optional[str] = Field(
        ...,
        description="The full history of the task.",
        examples=[
            "I am going to use the write_to_file command and write"
            " Washington to a file called output.txt"
            " <write_to_file('output.txt', 'Washington')"
        ],
    )
    total_tokens: Optional[int] = Field(
        ...,
        description="The total number of tokens generated.",
        examples=[7894],
    )
    # total_cost_in_dollar: Optional[str] = Field(
    #     default_factory=lambda: "0,24$",
    #     description="The total cost of the task.",
    #     examples=["0,24$"],
    # )


class GenerationOutputMetadata(BaseModel):
    num_of_tokens: int = Field(
        ...,
        description="The number of tokens generated.",
        examples=[7894],
    )
    estimated_cost: str = Field(
        ...,
        description="The estimated cost of the generation.",
        examples=["0,24$"],
    )
    time_to_generate: str = Field(
        ...,
        description="The time taken to generate the output.",
        examples=["1.2s"],
    )
    tokens_per_second: int = Field(
        ...,
        description="The number of tokens generated per second.",
        examples=[657],
    )
    model_name: str = Field(
        ...,
        description="The model used to generate the output.",
        examples=["gpt-3.5-turbo"],
    )
    max_tokens: int = Field(
        ...,
        description="The maximum number of tokens allowed to generate.",
        examples=[2048],
    )
    temperature: float = Field(
        ...,
        description="The temperature used for generation.",
        examples=[0.7],
    )
    top_p: float = Field(
        ...,
        description="The top p value used for generation.",
        examples=[0.9],
    )
    frequency_penalty: float = Field(
        ...,
        description="The frequency penalty used for generation.",
        examples=[0.0],
    )
    presence_penalty: float = Field(
        ...,
        description="The presence penalty used for generation.",
        examples=[0.0],
    )
    stop_sequence: str | None = Field(
        None,
        description="The sequence used to stop the generation.",
        examples=["<stop_sequence>"],
    )
    model_type: str = Field(
        ...,
        description="The type of model used for generation.",
        examples=["text"],
    )
    model_version: str = Field(
        ...,
        description="The version of the model used for generation.",
        examples=["1.0.0"],
    )
    model_description: str = Field(
        ...,
        description="The description of the model used for generation.",
        examples=["A model that generates text."],
    )
    model_author: str = Field(
        ...,
        description="The author of the model used for generation.",
        examples=["John Doe"],
    )
    n: int = Field(
        ...,
        description="The number of outputs generated.",
        examples=[1],
    )
    n_best: int = Field(
        ...,
        description="The number of best outputs generated.",
        examples=[1],
    )
