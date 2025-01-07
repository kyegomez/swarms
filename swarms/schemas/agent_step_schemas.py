from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from swarms.schemas.base_schemas import (
    AgentChatCompletionResponse,
)


def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


uuid_hex = uuid.uuid4().hex


class Step(BaseModel):
    step_id: str | None = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The ID of the task step.",
        examples=["6bb1801a-fd80-45e8-899a-4dd723cc602e"],
    )
    time: float | None = Field(
        default_factory=get_current_time,
        description="The time taken to complete the task step.",
    )
    response: AgentChatCompletionResponse | None


class ManySteps(BaseModel):
    agent_id: str | None = Field(
        ...,
        description="The ID of the agent.",
        examples=["financial-agent-1"],
    )
    agent_name: str | None = Field(
        ...,
        description="The ID of the agent.",
        examples=["financial-agent-1"],
    )
    task: str | None = Field(
        ...,
        description="The name of the task.",
        examples=["Write to file"],
    )
    max_loops: Any | None = Field(
        ...,
        description="The number of steps in the task.",
        examples=[3],
    )
    run_id: str | None = Field(
        uuid.uuid4().hex,
        description="The ID of the task this step belongs to.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    steps: list[Step | Any] | None = Field(
        [],
        description="The steps of the task.",
    )
    full_history: str | None = Field(
        ...,
        description="The full history of the task.",
        examples=[
            "I am going to use the write_to_file command and write"
            " Washington to a file called output.txt"
            " <write_to_file('output.txt', 'Washington')"
        ],
    )
    total_tokens: int | None = Field(
        ...,
        description="The total number of tokens generated.",
        examples=[7894],
    )
    stopping_token: str | None = Field(
        ...,
        description="The token at which the task stopped.",
    )
    interactive: bool | None = Field(
        ...,
        description="The interactive status of the task.",
        examples=[True],
    )
    dynamic_temperature_enabled: bool | None = Field(
        ...,
        description="The dynamic temperature status of the task.",
        examples=[True],
    )
