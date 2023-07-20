from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union
from pydantic import BaseModel

class ToolException(Exception):
    pass

class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        pass

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

class Tool(BaseTool):
    def __init__(self, name: str, description: str, func: Callable[..., Any]):
        self.name = name
        self.description = description
        self.func = func

    def run(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.func(*args, **kwargs)
        except ToolException as e:
            raise e

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self.func(*args, **kwargs)
        except ToolException as e:
            raise e

class StructuredTool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        func: Callable[..., Any]
    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func

    def run(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.func(*args, **kwargs)
        except ToolException as e:
            raise e

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self.func(*args, **kwargs)
        except ToolException as e:
            raise e

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    infer_schema: bool = True
) -> Callable:
    def decorator(func: Callable[..., Any]) -> Union[Tool, StructuredTool]:
        nonlocal name, description

        if name is None:
            name = func.__name__
        if description is None:
            description = func.__doc__ or ""

        if args_schema or infer_schema:
            if args_schema is None:
                args_schema = BaseModel

            return StructuredTool(name, description, args_schema, func)
        else:
            return Tool(name, description, func)

    return decorator
