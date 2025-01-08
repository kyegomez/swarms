# swarms.structs Documentation

## Introduction

The swarms.structs library provides a collection of classes for representing artifacts and their attributes. This documentation will provide an overview of the `Artifact` class, its attributes, functionality, and usage examples.

### Artifact Class

The `Artifact` class represents an artifact and its attributes. It inherits from the `BaseModel` class and includes the following attributes:

#### Attributes

1. `artifact_id (str)`: Id of the artifact.
2. `file_name (str)`: Filename of the artifact.
3. `relative_path (str, optional)`: Relative path of the artifact in the agent's workspace.

These attributes are crucial for identifying and managing different artifacts within a given context.

## Class Definition

The `Artifact` class can be defined as follows:

```python
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
        example="b225e278-8b4c-4f99-a696-8facf19f0e56",
    )
    file_name: str = Field(
        ..., description="Filename of the artifact", example="main.py"
    )
    relative_path: Optional[str] = Field(
        None,
        description=("Relative path of the artifact in the agent's workspace"),
        example="python/code/",
    )
```

The `Artifact` class defines the mandatory and optional attributes and provides corresponding descriptions along with example values.

## Functionality and Usage

The `Artifact` class encapsulates the information and attributes representing an artifact. It provides a structured and organized way to manage artifacts within a given context.

### Example 1: Creating an Artifact instance

To create an instance of the `Artifact` class, you can simply initialize it with the required attributes. Here's an example:

```python
from swarms.structs import Artifact

artifact_instance = Artifact(
    artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
    file_name="main.py",
    relative_path="python/code/",
)
```

In this example, we create an instance of the `Artifact` class with the specified artifact details.

### Example 2: Accessing Artifact attributes

You can access the attributes of the `Artifact` instance using dot notation. Here's how you can access the file name of the artifact:

```python
print(artifact_instance.file_name)
# Output: "main.py"
```

### Example 3: Handling optional attributes

If the `relative_path` attribute is not provided during artifact creation, it will default to `None`. Here's an example:

```python
artifact_instance_no_path = Artifact(
    artifact_id="c280s347-9b7d-3c68-m337-7abvf50j23k", file_name="script.js"
)

print(artifact_instance_no_path.relative_path)
# Output: None
```

By providing default values for optional attributes, the `Artifact` class allows flexibility in defining artifact instances.

### Additional Information and Tips

The `Artifact` class represents a powerful and flexible means of handling various artifacts with different attributes. By utilizing this class, users can organize, manage, and streamline their artifacts with ease.

## References and Resources

For further details and references related to the swarms.structs library and the `Artifact` class, refer to the [official documentation](https://swarms.structs.docs/artifact.html).

This comprehensive documentation provides an in-depth understanding of the `Artifact` class, its attributes, functionality, and usage examples. By following the detailed examples and explanations, developers can effectively leverage the capabilities of the `Artifact` class within their projects.
