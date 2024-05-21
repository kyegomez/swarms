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


class Step(BaseModel):
    task_id: str = Field(
        ...,
        description="The ID of the task this step belongs to.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    step_id: int = Field(
        ...,
        description="The ID of the task step.",
        examples=["6bb1801a-fd80-45e8-899a-4dd723cc602e"],
    )
    name: str | None = Field(
        None,
        description="The name of the task step.",
        examples=["Write to file"],
    )
    output: str | None = Field(
        None,
        description="Output of the task step.",
        examples=[
            "I am going to use the write_to_file command and write"
            " Washington to a file called output.txt"
            " <write_to_file('output.txt', 'Washington')"
        ],
    )
    artifacts: list[Artifact] = Field(
        [],
        description="A list of artifacts that the step has produced.",
    )
    max_loops: int = Field(
        1,
        description="The maximum number of times to run the workflow.",
    )


class ManySteps(BaseModel):
    task_id: str = Field(
        ...,
        description="The ID of the task this step belongs to.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    steps: list[Step] = Field(
        [],
        description="A list of task steps.",
    )


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
