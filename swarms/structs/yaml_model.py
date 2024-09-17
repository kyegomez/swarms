from pydantic import BaseModel, Field
import yaml
import json
from swarms.utils.loguru_logger import logger
from typing import Any, Dict
from typing import Type
from dataclasses import is_dataclass, fields


def get_type_name(typ: Type) -> str:
    """Map Python types to simple string representations."""
    if hasattr(typ, "__name__"):
        return typ.__name__
    return str(typ)


def create_yaml_schema_from_dict(
    data: Dict[str, Any], model_class: Type
) -> str:
    """
    Generate a YAML schema based on a dictionary and a class (can be a Pydantic model, regular class, or dataclass).

    Args:
        data: The dictionary with key-value pairs where keys are attribute names and values are example data.
        model_class: The class which the data should conform to, used for obtaining type information.

    Returns:
        A string containing the YAML schema.

    Example usage:
    >>> data = {'name': 'Alice', 'age: 30, 'is_active': True}
    >>> print(create_yaml_schema_from_dict(data, User))

    """
    schema = {}
    if is_dataclass(model_class):
        for field in fields(model_class):
            schema[field.name] = {
                "type": get_type_name(field.type),
                "default": (
                    field.default
                    if field.default != field.default_factory
                    else None
                ),
                "description": field.metadata.get(
                    "description", "No description provided"
                ),
            }
    elif isinstance(model_class, BaseModel):
        for field_name, model_field in model_class.__fields__.items():
            field_info = model_field.field_info
            schema[field_name] = {
                "type": get_type_name(model_field.outer_type_),
                "default": field_info.default,
                "description": (
                    field_info.description
                    or "No description provided."
                ),
            }
    else:
        # Fallback for regular classes (non-dataclass, non-Pydantic)
        for attr_name, attr_value in data.items():
            attr_type = type(attr_value)
            schema[attr_name] = {
                "type": get_type_name(attr_type),
                "description": "No description provided",
            }

    return yaml.safe_dump(schema, sort_keys=False)


def pydantic_type_to_yaml_schema(pydantic_type):
    """
    Map Pydantic types to YAML schema types.

    Args:
        pydantic_type (type): The Pydantic type to be mapped.

    Returns:
        str: The corresponding YAML schema type.

    """
    type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    # For more complex types or generics, you would expand this mapping
    base_type = getattr(pydantic_type, "__origin__", pydantic_type)
    if base_type is None:
        base_type = pydantic_type
    return type_mapping.get(base_type, "string")


class YamlModel(BaseModel):
    """
    A Pydantic model class for working with YAML data.


    Example usage:
    # Example usage with an extended YamlModel
    >>> class User(YamlModel):
        name: str
        age: int
        is_active: bool

    # Create an instance of the User model
    >>> user = User(name="Alice", age=30, is_active=True)

    # Serialize the User instance to YAML and print it
    >>> print(user.to_yaml())

    # Convert JSON to YAML and print
    >>> json_string = '{"name": "Bob", "age": 25, "is_active": false}'
    >>> print(YamlModel.json_to_yaml(json_string))

    # Save the User instance to a YAML file
    >>> user.save_to_yaml('user.yaml')
    """

    input_dict: Dict[str, Any] = Field(
        None,
        title="Data",
        description="The data to be serialized to YAML.",
    )

    def to_yaml(self):
        """
        Serialize the Pydantic model instance to a YAML string.
        """
        return yaml.safe_dump(self.input_dict, sort_keys=False)

    def from_yaml(self, cls, yaml_str: str):
        """
        Create an instance of the class from a YAML string.

        Args:
            yaml_str (str): The YAML string to parse.

        Returns:
            cls: An instance of the class with attributes populated from the YAML data.
                 Returns None if there was an error loading the YAML data.
        """
        try:
            data = yaml.safe_load(yaml_str)
            return cls(**data)
        except ValueError as error:
            logger.error(f"Error loading YAML data: {error}")
            return None

    @staticmethod
    def json_to_yaml(self, json_str: str):
        """
        Convert a JSON string to a YAML string.
        """
        data = json.loads(
            json_str
        )  # Convert JSON string to dictionary
        return yaml.dump(data)

    def save_to_yaml(self, filename: str):
        """
        Save the Pydantic model instance as a YAML file.
        """
        yaml_data = self.to_yaml()
        with open(filename, "w") as file:
            file.write(yaml_data)

    # TODO: Implement a method to create a YAML schema from the model fields
    # @classmethod
    # def create_yaml_schema(cls):
    #     """
    #     Generate a YAML schema based on the fields of the given BaseModel Class.

    #     Args:
    #         cls: The class for which the YAML schema is generated.

    #     Returns:
    #         A YAML representation of the schema.

    #     """
    #     schema = {}
    #     for field_name, model_field in cls.model_fields.items():  # Use model_fields
    #         field_type = model_field.type_  # Assuming type_ for field type access
    #         field_info = model_field  # FieldInfo object already
    #         schema[field_name] = {
    #             'type': pydantic_type_to_yaml_schema(field_type),
    #             'description': field_info.description or "No description provided."
    #         }
    #         if field_info is not None:  # Check for default value directly
    #             schema[field_name]['default'] = field_info.default
    #     return yaml.safe_dump(schema, sort_keys=False)

    def create_yaml_schema_from_dict(
        self, data: Dict[str, Any], model_class: Type
    ) -> str:
        """
        Generate a YAML schema based on a dictionary and a class (can be a Pydantic model, regular class, or dataclass).

        Args:
            data: The dictionary with key-value pairs where keys are attribute names and values are example data.
            model_class: The class which the data should conform to, used for obtaining type information.

        Returns:
            A string containing the YAML schema.

        Example usage:
        >>> data = {'name': 'Alice', 'age: 30, 'is_active': True}
        """
        return create_yaml_schema_from_dict(data, model_class)

    def yaml_to_dict(self, yaml_str: str):
        """
        Convert a YAML string to a Python dictionary.
        """
        return yaml.safe_load(yaml_str)

    def dict_to_yaml(self, data: Dict[str, Any]):
        """
        Convert a Python dictionary to a YAML string.
        """
        return yaml.safe_dump(data, sort_keys=False)


# dict = {'name': 'Alice', 'age': 30, 'is_active': True}

# # Comvert the dictionary to a YAML schema dict to yaml
# yaml_model = YamlModel().dict_to_yaml(dict)
# print(yaml_model)
