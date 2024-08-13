# modified from Lanarky source https://github.com/auxon/lanarky
from typing import Any
 
import pydantic
from pydantic.fields import FieldInfo
 
try:
    from enum import StrEnum  # type: ignore
except ImportError:
    from enum import Enum
 
    class StrEnum(str, Enum): ...
 
 
PYDANTIC_V2 = pydantic.VERSION.startswith("2.")
 
 
def model_dump(model: pydantic.BaseModel, **kwargs) -> dict[str, Any]:
    """Dump a pydantic model to a dictionary.
 
    Args:
        model: A pydantic model.
    """
    if PYDANTIC_V2:
        return model.model_dump(**kwargs)
    else:
        return model.dict(**kwargs)
 
 
def model_dump_json(model: pydantic.BaseModel, **kwargs) -> str:
    """Dump a pydantic model to a JSON string.
 
    Args:
        model: A pydantic model.
    """
    if PYDANTIC_V2:
        return model.model_dump_json(**kwargs)
    else:
        return model.json(**kwargs)
 
 
def model_fields(model: pydantic.BaseModel) -> dict[str, FieldInfo]:
    """Get the fields of a pydantic model.
 
    Args:
        model: A pydantic model.
    """
    if PYDANTIC_V2:
        return model.model_fields
    else:
        return model.__fields__