from __future__ import annotations

import json
import pprint
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, StrictStr
from swarms.artifacts.main import Artifact
from swarms.artifacts.error_artifact import ErrorArtifact


class BaseTask(ABC):

    class State(Enum):
        PENDING = 1
        EXECUTING = 2
        FINISHED = 3

    def __init__(self):
        self.id: str = uuid.uuid4().hex
        self.state: BaseTask.State = self.State.PENDING
        self.parent_ids: List[str] = []
        self.child_ids: List[str] = []
        self.output: Optional[Union[Artifact, ErrorArtifact]] = None
        self.structure = None

    @property
    @abstractmethod
    def input(self) -> Any:
        pass

    @property
    def parents(self) -> List[BaseTask]:
        return [
            self.structure.find_task(parent_id) for parent_id in self.parent_ids
        ]

    @property
    def children(self) -> List[BaseTask]:
        return [
            self.structure.find_task(child_id) for child_id in self.child_ids
        ]

    def __rshift__(self, child: BaseTask) -> BaseTask:
        return self.add_child(child)

    def __lshift__(self, child: BaseTask) -> BaseTask:
        return self.add_parent(child)

    def preprocess(self, structure) -> BaseTask:
        self.structure = structure
        return self

    def add_child(self, child: BaseTask) -> BaseTask:
        if self.structure:
            child.structure = self.structure
        elif child.structure:
            self.structure = child.structure

        if child not in self.structure.tasks:
            self.structure.tasks.append(child)

        if self not in self.structure.tasks:
            self.structure.tasks.append(self)

        if child.id not in self.child_ids:
            self.child_ids.append(child.id)

        if self.id not in child.parent_ids:
            child.parent_ids.append(self.id)

        return child

    def add_parent(self, parent: BaseTask) -> BaseTask:
        if self.structure:
            parent.structure = self.structure
        elif parent.structure:
            self.structure = parent.structure

        if parent not in self.structure.tasks:
            self.structure.tasks.append(parent)

        if self not in self.structure.tasks:
            self.structure.tasks.append(self)

        if parent.id not in self.parent_ids:
            self.parent_ids.append(parent.id)

        if self.id not in parent.child_ids:
            parent.child_ids.append(self.id)

        return parent

    def is_pending(self) -> bool:
        return self.state == self.State.PENDING

    def is_finished(self) -> bool:
        return self.state == self.State.FINISHED

    def is_executing(self) -> bool:
        return self.state == self.State.EXECUTING

    def before_run(self) -> None:
        pass

    def after_run(self) -> None:
        pass

    def execute(self) -> Optional[Union[Artifact, ErrorArtifact]]:
        try:
            self.state = self.State.EXECUTING
            self.before_run()
            self.output = self.run()
            self.after_run()
        except Exception as e:
            self.output = ErrorArtifact(str(e))
        finally:
            self.state = self.State.FINISHED
            return self.output

    def can_execute(self) -> bool:
        return self.state == self.State.PENDING and all(
            parent.is_finished() for parent in self.parents)

    def reset(self) -> BaseTask:
        self.state = self.State.PENDING
        self.output = None
        return self

    @abstractmethod
    def run(self) -> Optional[Union[Artifact, ErrorArtifact]]:
        pass


class Task(BaseModel):
    input: Optional[StrictStr] = Field(None,
                                       description="Input prompt for the task")
    additional_input: Optional[Any] = Field(
        None, description="Input parameters for the task. Any value is allowed")
    task_id: StrictStr = Field(..., description="ID of the task")

    class Config:
        allow_population_by_field_name = True
        validate_assignment = True

    def to_str(self) -> str:
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        return json.dumps(self.dict(by_alias=True, exclude_none=True))

    @classmethod
    def from_json(cls, json_str: str) -> "Task":
        return cls.parse_raw(json_str)

    def to_dict(self) -> dict:
        _dict = self.dict(by_alias=True, exclude_none=True)
        if self.artifacts:
            _dict["artifacts"] = [
                artifact.dict(by_alias=True, exclude_none=True)
                for artifact in self.artifacts
            ]
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> "Task":
        if obj is None:
            return None
        if not isinstance(obj, dict):
            raise ValueError("Input must be a dictionary.")
        if "artifacts" in obj:
            obj["artifacts"] = [
                Artifact.parse_obj(artifact) for artifact in obj["artifacts"]
            ]
        return cls.parse_obj(obj)
