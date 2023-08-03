from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type, Union
from pydantic import BaseModel


from swarms.utils.logger import logger

class ToolScope(Enum):
    GLOBAL = "global"
    SESSION = "session"



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



SessionGetter = Callable[[], Tuple[str, AgentExecutor]]

def tool(
    name: str,
    description: str,
    scope: ToolScope = ToolScope.GLOBAL,
):
    def decorator(func):
        func.name = name
        func.description = description
        func.is_tool = True
        func.scope = scope
        return func

    return decorator


class ToolWrapper:
    def __init__(self, name: str, description: str, scope: ToolScope, func):
        self.name = name
        self.description = description
        self.scope = scope
        self.func = func

    def is_global(self) -> bool:
        return self.scope == ToolScope.GLOBAL

    def is_per_session(self) -> bool:
        return self.scope == ToolScope.SESSION

    def to_tool(
        self,
        get_session: SessionGetter = lambda: [],
    ) -> BaseTool:
        func = self.func
        if self.is_per_session():
            def func(*args, **kwargs):
                return self.func(*args, **kwargs, get_session=get_session)

        return Tool(
            name=self.name,
            description=self.description,
            func=func,
        )


class BaseToolSet:
    def tool_wrappers(cls) -> list[ToolWrapper]:
        methods = [
            getattr(cls, m) for m in dir(cls) if hasattr(getattr(cls, m), "is_tool")
        ]
        return [ToolWrapper(m.name, m.description, m.scope, m) for m in methods]
    


class ToolsFactory:
    @staticmethod
    def from_toolset(
        toolset: BaseToolSet,
        only_global: Optional[bool] = False,
        only_per_session: Optional[bool] = False,
        get_session: SessionGetter = lambda: [],
    ) -> list[BaseTool]:
        tools = []
        for wrapper in toolset.tool_wrappers():
            if only_global and not wrapper.is_global():
                continue
            if only_per_session and not wrapper.is_per_session():
                continue
            tools.append(wrapper.to_tool(get_session=get_session))
        return tools

    @staticmethod
    def create_global_tools(
        toolsets: list[BaseToolSet],
    ) -> list[BaseTool]:
        tools = []
        for toolset in toolsets:
            tools.extend(
                ToolsFactory.from_toolset(
                    toolset=toolset,
                    only_global=True,
                )
            )
        return tools

    @staticmethod
    def create_per_session_tools(
        toolsets: list[BaseToolSet],
        get_session: SessionGetter = lambda: [],
    ) -> list[BaseTool]:
        tools = []
        for toolset in toolsets:
            tools.extend(
                ToolsFactory.from_toolset(
                    toolset=toolset,
                    only_per_session=True,
                    get_session=get_session,
                )
            )
        return tools

    @staticmethod
    def create_global_tools_from_names(
        toolnames: list[str],
        llm: Optional[BaseLLM],
    ) -> list[BaseTool]:
        return load_tools(toolnames, llm=llm)
    