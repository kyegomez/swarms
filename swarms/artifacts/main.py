from __future__ import annotations
import pprint
import json

from typing import Optional
from pydantic import BaseModel, Field, StrictStr

class Artifact(BaseModel):
    """

    Artifact that has the task has been produced
    """

    artifact_id: StrictStr = Field(
        ...,
        description="ID of the artifact"
    )
    file_name: StrictStr = Field(
        ...,
        description="Filename of the artifact"
    )
    relative_path: Optional[StrictStr] = Field(
        None, description="Relative path of the artifact"
    )
    __properties = ["artifact_id", "file_name", "relative_path"]

    class Config:
        """Pydantic configuration"""

        allow_population_by_field_name = True
        validate_assignment = True

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))
    
    @classmethod
    def from_json(cls, json_str: str) -> Artifact:
        """Create an instance of Artifact from a json string"""
        return cls.from_dict(json.loads(json_str))
    
    def to_dict(self):
        """Returns the dict representation of the model"""
        _dict = self.dict(by_alias=True, exclude={}, exclude_none=True)
        return _dict
    
    @classmethod
    def from_dict(cls, obj: dict) -> Artifact:
        """Create an instance of Artifact from a dict"""
        
        if obj is None:
            return None
        
        if not isinstance(obj, dict):
            return Artifact.parse_obj(obj)
        
        _obj = Artifact.parse_obj(
            {
                "artifact_id": obj.get("artifact_id"),
                "file_name": obj.get("file_name"),
                "relative_path": obj.get("relative_path"),
            }
        )

        return _obj


