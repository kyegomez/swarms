from __future__ import annotations
from attr import define, field
from swarms.artifacts.base import BaseArtifact


@define(frozen=True)
class ErrorArtifact(BaseArtifact):
    value: str = field(converter=str)

    def __add__(self, other: ErrorArtifact) -> ErrorArtifact:
        return ErrorArtifact(self.value + other.value)
    
    def to_text(self) -> str:
        return self.value
    
    def to_dict(self) -> dict:
        from griptape.schemas import ErrorArtifactSchema

        return dict(ErrorArtifactSchema().dump(self))
    