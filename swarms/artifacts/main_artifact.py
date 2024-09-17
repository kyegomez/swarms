import time
from swarms.utils.loguru_logger import logger
import os
import json
from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class FileVersion(BaseModel):
    """
    Represents a version of the file with its content and timestamp.
    """

    version_number: int = Field(
        ..., description="The version number of the file"
    )
    content: str = Field(
        ..., description="The content of the file version"
    )
    timestamp: str = Field(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        description="The timestamp of the file version",
    )

    def __str__(self) -> str:
        return f"Version {self.version_number} (Timestamp: {self.timestamp}):\n{self.content}"


class Artifact(BaseModel):
    """
    Represents a file artifact.

    Attributes:
        file_path (str): The path to the file.
        file_type (str): The type of the file.
        contents (str): The contents of the file.
        versions (List[FileVersion]): The list of file versions.
        edit_count (int): The number of times the file has been edited.
    """

    file_path: str = Field(..., description="The path to the file")
    file_type: str = Field(
        ...,
        description="The type of the file",
        # example=".txt",
    )
    contents: str = Field(
        ..., description="The contents of the file in string format"
    )
    versions: List[FileVersion] = Field(default_factory=list)
    edit_count: int = Field(
        ...,
        description="The number of times the file has been edited",
    )

    @validator("file_type", pre=True, always=True)
    def validate_file_type(cls, v, values):
        if not v:
            file_path = values.get("file_path")
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in [
                ".py",
                ".csv",
                ".tsv",
                ".txt",
                ".json",
                ".xml",
                ".html",
                ".yaml",
                ".yml",
                ".md",
                ".rst",
                ".log",
                ".sh",
                ".bat",
                ".ps1",
                ".psm1",
                ".psd1",
                ".ps1xml",
                ".pssc",
                ".reg",
                ".mof",
                ".mfl",
                ".xaml",
                ".xml",
                ".wsf",
                ".config",
                ".ini",
                ".inf",
                ".json5",
                ".hcl",
                ".tf",
                ".tfvars",
                ".tsv",
                ".properties",
            ]:
                raise ValueError("Unsupported file type")
            return ext.lower()
        return v

    def create(self, initial_content: str) -> None:
        """
        Creates a new file artifact with the initial content.
        """
        try:
            self.contents = initial_content
            self.versions.append(
                FileVersion(
                    version_number=1,
                    content=initial_content,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            self.edit_count = 0
        except Exception as e:
            logger.error(f"Error creating artifact: {e}")
            raise e

    def edit(self, new_content: str) -> None:
        """
        Edits the artifact's content, tracking the change in the version history.
        """
        try:
            self.contents = new_content
            self.edit_count += 1
            new_version = FileVersion(
                version_number=len(self.versions) + 1,
                content=new_content,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.versions.append(new_version)
        except Exception as e:
            logger.error(f"Error editing artifact: {e}")
            raise e

    def save(self) -> None:
        """
        Saves the current artifact's contents to the specified file path.
        """
        with open(self.file_path, "w") as f:
            f.write(self.contents)

    def load(self) -> None:
        """
        Loads the file contents from the specified file path into the artifact.
        """
        with open(self.file_path, "r") as f:
            self.contents = f.read()
        self.create(self.contents)

    def get_version(
        self, version_number: int
    ) -> Union[FileVersion, None]:
        """
        Retrieves a specific version of the artifact by its version number.
        """
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None

    def get_contents(self) -> str:
        """
        Returns the current contents of the artifact as a string.
        """
        return self.contents

    def get_version_history(self) -> str:
        """
        Returns the version history of the artifact as a formatted string.
        """
        return "\n\n".join(
            [str(version) for version in self.versions]
        )

    def export_to_json(self, file_path: str) -> None:
        """
        Exports the artifact to a JSON file.

        Args:
            file_path (str): The path to the JSON file where the artifact will be saved.
        """
        with open(file_path, "w") as json_file:
            json.dump(self.dict(), json_file, default=str, indent=4)

    @classmethod
    def import_from_json(cls, file_path: str) -> "Artifact":
        """
        Imports an artifact from a JSON file.

        Args:
            file_path (str): The path to the JSON file to import the artifact from.

        Returns:
            Artifact: The imported artifact instance.
        """
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        # Convert timestamp strings back to datetime objects
        for version in data["versions"]:
            version["timestamp"] = datetime.fromisoformat(
                version["timestamp"]
            )
        return cls(**data)

    def get_metrics(self) -> str:
        """
        Returns all metrics of the artifact as a formatted string.

        Returns:
            str: A string containing all metrics of the artifact.
        """
        metrics = (
            f"File Path: {self.file_path}\n"
            f"File Type: {self.file_type}\n"
            f"Current Contents:\n{self.contents}\n\n"
            f"Edit Count: {self.edit_count}\n"
            f"Version History:\n{self.get_version_history()}"
        )
        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the artifact instance to a dictionary representation.
        """
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """
        Creates an artifact instance from a dictionary representation.
        """
        try:
            # Convert timestamp strings back to datetime objects if necessary
            for version in data.get("versions", []):
                if isinstance(version["timestamp"], str):
                    version["timestamp"] = datetime.fromisoformat(
                        version["timestamp"]
                    )
            return cls(**data)
        except Exception as e:
            logger.error(f"Error creating artifact from dict: {e}")
            raise e


# # Example usage
# artifact = Artifact(file_path="example.txt", file_type=".txt")
# artifact.create("Initial content")
# artifact.edit("First edit")
# artifact.edit("Second edit")
# artifact.save()

# # Export to JSON
# artifact.export_to_json("artifact.json")

# # Import from JSON
# imported_artifact = Artifact.import_from_json("artifact.json")

# # # Get metrics
# print(artifact.get_metrics())
