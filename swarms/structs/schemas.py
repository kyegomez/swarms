from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskInput(BaseModel):
    task: Any = Field(
        ...,
        description=(
            "The input parameters for the task. Any value is allowed."
        ),
        examples=['{\n"debug": false,\n"mode": "benchmarks"\n}'],
    )


class Artifact(BaseModel):
    """
    Represents an artifact.

    Attributes:
        artifact_id (str): Id of the artifact.
        file_name (str): Filename of the artifact.
        relative_path (str, optional): Relative path of the artifact in the agent's workspace.
    """

    artifact_id: str = Field(
        ...,
        description="Id of the artifact",
        examples=["b225e278-8b4c-4f99-a696-8facf19f0e56"],
    )
    file_name: str = Field(
        ...,
        description="Filename of the artifact",
        examples=["main.py"],
    )
    relative_path: str | None = Field(
        None,
        description=(
            "Relative path of the artifact in the agent's workspace"
        ),
        examples=["python/code/"],
    )


class ArtifactUpload(BaseModel):
    file: bytes = Field(..., description="File to upload")
    relative_path: str | None = Field(
        None,
        description=(
            "Relative path of the artifact in the agent's workspace"
        ),
        examples=["python/code/"],
    )


class StepInput(BaseModel):
    step: Any = Field(
        ...,
        description=(
            "Input parameters for the task step. Any value is" " allowed."
        ),
        examples=['{\n"file_to_refactor": "models.py"\n}'],
    )


class StepOutput(BaseModel):
    step: Any = Field(
        ...,
        description=(
            "Output that the task step has produced. Any value is"
            " allowed."
        ),
        examples=['{\n"tokens": 7894,\n"estimated_cost": "0,24$"\n}'],
    )


class TaskRequestBody(BaseModel):
    input: str | None = Field(
        None,
        description="Input prompt for the task.",
        examples=["Write the words you receive to the file 'output.txt'."],
    )
    additional_input: TaskInput | None = None


class Task(TaskRequestBody):
    task_id: str = Field(
        ...,
        description="The ID of the task.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    artifacts: list[Artifact] = Field(
        [],
        description="A list of artifacts that the task has produced.",
        examples=[
            [
                "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
                "ab7b4091-2560-4692-a4fe-d831ea3ca7d6",
            ]
        ],
    )


class StepRequestBody(BaseModel):
    input: str | None = Field(
        None,
        description="Input prompt for the step.",
        examples=["Washington"],
    )
    additional_input: StepInput | None = None


class Status(Enum):
    created = "created"
    running = "running"
    completed = "completed"


class Step(StepRequestBody):
    task_id: str = Field(
        ...,
        description="The ID of the task this step belongs to.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    step_id: str = Field(
        ...,
        description="The ID of the task step.",
        examples=["6bb1801a-fd80-45e8-899a-4dd723cc602e"],
    )
    name: str | None = Field(
        None,
        description="The name of the task step.",
        examples=["Write to file"],
    )
    status: Status = Field(..., description="The status of the task step.")
    output: str | None = Field(
        None,
        description="Output of the task step.",
        examples=[
            "I am going to use the write_to_file command and write"
            " Washington to a file called output.txt"
            " <write_to_file('output.txt', 'Washington')"
        ],
    )
    additional_output: StepOutput | None = None
    artifacts: list[Artifact] = Field(
        [],
        description="A list of artifacts that the step has produced.",
    )
    is_last: bool | None = Field(
        False,
        description="Whether this is the last step in the task.",
    )
