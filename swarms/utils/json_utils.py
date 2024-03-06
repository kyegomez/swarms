import json


from pydantic import BaseModel


def base_model_schema_to_json(model: BaseModel):
    """
    Converts the JSON schema of a base model to a formatted JSON string.

    Args:
        model (BaseModel): The base model for which to generate the JSON schema.

    Returns:
        str: The JSON schema of the base model as a formatted JSON string.
    """
    return json.dumps(model.model_json_schema(), indent=2)
