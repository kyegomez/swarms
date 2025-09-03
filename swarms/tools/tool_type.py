from typing import Dict, List, Callable, Any, Type, Union

ToolType = Union[
    Dict[str, Any], List[Callable[..., Any]], Callable[..., Any], Any
]

tool_type = Type[ToolType]
