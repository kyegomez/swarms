from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseArtifact(ABC):
    """
    Base class for artifacts.
    """

    id: str
    name: str
    value: Any

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4().hex
        if self.name is None:
            self.name = self.id

    @classmethod
    def value_to_bytes(cls, value: Any) -> bytes:
        """
        Convert the value to bytes.
        """
        if isinstance(value, bytes):
            return value
        else:
            return str(value).encode()

    @classmethod
    def value_to_dict(cls, value: Any) -> dict:
        """
        Convert the value to a dictionary.
        """
        if isinstance(value, dict):
            dict_value = value
        else:
            dict_value = json.loads(value)

        return {k: v for k, v in dict_value.items()}

    def to_text(self) -> str:
        """
        Convert the value to text.
        """
        return str(self.value)

    def __str__(self) -> str:
        """
        Return a string representation of the artifact.
        """
        return self.to_text()

    def __bool__(self) -> bool:
        """
        Return the boolean value of the artifact.
        """
        return bool(self.value)

    def __len__(self) -> int:
        """
        Return the length of the artifact.
        """
        return len(self.value)

    @abstractmethod
    def __add__(self, other: BaseArtifact) -> BaseArtifact:
        """
        Add two artifacts together.
        """
        ...
