from __future__ import annotations

import json
import pprint
from typing import Any, Optional

from artifacts.main import Artifact
from pydantic import BaseModel, Field, StrictStr, conlist


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