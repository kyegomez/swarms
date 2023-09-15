from __future__ import annotations

import json
import pprint
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from swarms.artifacts.main import Artifact
from pydantic import BaseModel, Field, StrictStr, conlist

from swarms.artifacts.error_artifact import ErrorArtifact


class BaseTask(ABC):
    class State(Enum):
        PENDING = 1
        EXECUTING = 2
        FINISHED = 3

    def __init__(self):
        self.id = uuid.uuid4().hex
        self.state = self.State.PENDING
        self.parent_ids = []
        self.child_ids = []
        self.output = None
        self.structure = None

    @property
    @abstractmethod
    def input(self):
        pass

    @property
    def parents(self):
        return [self.structure.find_task(parent_id) for parent_id in self.parent_ids]

    @property
    def children(self):
        return [self.structure.find_task(child_id) for child_id in self.child_ids]

    def __rshift__(self, child):
        return self.add_child(child)

    def __lshift__(self, child):
        return self.add_parent(child)

    def preprocess(self, structure):
        self.structure = structure
        return self

    def add_child(self, child):
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

    def add_parent(self, parent):
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

    def is_pending(self):
        return self.state == self.State.PENDING

    def is_finished(self):
        return self.state == self.State.FINISHED

    def is_executing(self):
        return self.state == self.State.EXECUTING

    def before_run(self):
        pass

    def after_run(self):
        pass

    def execute(self):
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

    def can_execute(self):
        return self.state == self.State.PENDING and all(parent.is_finished() for parent in self.parents)

    def reset(self):
        self.state = self.State.PENDING
        self.output = None
        return self

    @abstractmethod
    def run(self):
        pass













class Task(BaseModel):
    input: Optional[StrictStr] = Field(
        None,
        description="Input prompt for the task"
    )
    additional_input: Optional[Any] = Field(
        None,
        description="Input parameters for the task. Any value is allowed"
    )
    task_id: StrictStr = Field(
        ...,
        description="ID of the task"
    )
    artifacts: conlist(Artifact) = Field(
        ...,
        description="A list of artifacts that the task has been produced"
    )

    __properties = ["input", "additional_input", "task_id", "artifact"]

    class Config:
        #pydantic config
        
        allow_population_by_field_name = True
        validate_assignment = True

    def to_str(self) -> str:
        """Returns the str representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))
    
    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> Task:
        """Create an instance of Task from a json string"""
        return cls.from_dict(json.loads(json_str))
    
    def to_dict(self):
        """Returns the dict representation of the model using alias"""
        _dict = self.dict(by_alias=True, exclude={}, exclude_none=True)
        _items =[]
        if self.artifacts:
            for _item in self.artifacts:
                if _item:
                    _items.append(_item.to_dict())
            _dict["artifacts"] = _items
        #set to None if additional input is None
        # and __fields__set contains the field
        if self.additional_input is None and "additional_input" in self.__fields__set__:
            _dict["additional_input"] = None
        
        return _dict
    
    @classmethod
    def from_dict(cls, obj: dict) -> Task:
        """Create an instance of Task from dict"""
        if obj is None:
            return None
        
        if not isinstance(obj, dict):
            return Task.parse_obj(obj)
        
        _obj = Task.parse_obj(
            {
                "input": obj.get("input"),
                "additional_input": obj.get("additional_input"),
                "task_id": obj.get("task_id"),
                "artifacts": [
                    Artifact.from_dict(_item) for _item in obj.get("artifacts")
                ]
                if obj.get("artifacts") is not None
                else None,
            }
        )