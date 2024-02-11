import json
from abc import ABC, abstractmethod


class JSON(ABC):
    def __init__(self, schema_path):
        """
        Initializes a JSONSchema object.

        Args:
            schema_path (str): The path to the JSON schema file.
        """
        self.schema_path = schema_path
        self.schema = self.load_schema()

    def load_schema(self):
        """
        Loads the JSON schema from the specified file.

        Returns:
            dict: The loaded JSON schema.
        """
        with open(self.schema_path, "r") as f:
            return json.load(f)

    @abstractmethod
    def validate(self, data):
        """
        Validates the given data against the JSON schema.

        Args:
            data (dict): The data to be validated.

        Raises:
            NotImplementedError: This method needs to be implemented by the subclass.
        """
        pass
