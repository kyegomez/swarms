from __future__ import annotations
import json
import uuid
from abc import ABC, abstractmethod
from attr import define, field, Factory
from marshmallow import class_registry
from marshmallow.exceptions import RegistryError


@define
class BaseArtifact(ABC):
    id: str = field(default=Factory(lambda: uuid.uuid4().hex), kw_only=True)
    name: str = field(default=Factory(lambda self: self.id, takes_self=True), kw_only=True)
    value: any = field()
    type: str = field(default=Factory(lambda self: self.__class__.__name__, takes_self=True), kw_only=True)

    @classmethod
    def value_to_bytes(cls, value: any) -> bytes:
        if isinstance(value, bytes):
            return value
        else:
            return str(value).encode()

    @classmethod
    def value_to_dict(cls, value: any) -> dict:
        if isinstance(value, dict):
            dict_value = value
        else:
            dict_value = json.loads(value)

        return {k: v for k, v in dict_value.items()}

    @classmethod
    def from_dict(cls, artifact_dict: dict) -> BaseArtifact:
        from griptape.schemas import (
            TextArtifactSchema,
            InfoArtifactSchema,
            ErrorArtifactSchema,
            BlobArtifactSchema,
            CsvRowArtifactSchema,
            ListArtifactSchema
        )

        class_registry.register("TextArtifact", TextArtifactSchema)
        class_registry.register("InfoArtifact", InfoArtifactSchema)
        class_registry.register("ErrorArtifact", ErrorArtifactSchema)
        class_registry.register("BlobArtifact", BlobArtifactSchema)
        class_registry.register("CsvRowArtifact", CsvRowArtifactSchema)
        class_registry.register("ListArtifact", ListArtifactSchema)

        try:
            return class_registry.get_class(artifact_dict["type"])().load(artifact_dict)
        except RegistryError:
            raise ValueError("Unsupported artifact type")

    @classmethod
    def from_json(cls, artifact_str: str) -> BaseArtifact:
        return cls.from_dict(json.loads(artifact_str))

    def __str__(self):
        return json.dumps(self.to_dict())

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @abstractmethod
    def to_text(self) -> str:
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    @abstractmethod
    def __add__(self, other: BaseArtifact) -> BaseArtifact:
        ...
