from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type, Tuple
from pydantic import BaseModel


from langchain.llms.base import BaseLLM
from langchain.agents.agent import AgentExecutor
from langchain.agents import load_tools

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

    @abstractmethod
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


SessionGetter = Callable[[], Tuple[str, AgentExecutor]]


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

    def to_tool(self, get_session: SessionGetter = lambda: []) -> BaseTool:
        if self.is_per_session():
            self.func = lambda *args, **kwargs: self.func(*args, **kwargs, get_session=get_session)

        return Tool(name=self.name, description=self.description, func=self.func)


class BaseToolSet:
    def tool_wrappers(cls) -> list[ToolWrapper]:
        methods = [getattr(cls, m) for m in dir(cls) if hasattr(getattr(cls, m), "is_tool")]
        return [ToolWrapper(m.name, m.description, m.scope, m) for m in methods]


class ToolCreator(ABC):
    @abstractmethod
    def create_tools(self, toolsets: list[BaseToolSet]) -> list[BaseTool]:
        pass


class GlobalToolsCreator(ToolCreator):
    def create_tools(self, toolsets: list[BaseToolSet]) -> list[BaseTool]:
        tools = []
        for toolset in toolsets:
            tools.extend(
                ToolsFactory.from_toolset(
                    toolset=toolset,
                    only_global=True,
                )
            )
        return tools


class SessionToolsCreator(ToolCreator):
    def create_tools(self, toolsets: list[BaseToolSet], get_session: SessionGetter = lambda: []) -> list[BaseTool]:
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


class ToolsFactory:
    @staticmethod
    def from_toolset(toolset: BaseToolSet, only_global: Optional[bool] = False, only_per_session: Optional[bool] = False, get_session: SessionGetter = lambda: []) -> list[BaseTool]:
        tools = []
        for wrapper in toolset.tool_wrappers():
            if only_global and not wrapper.is_global():
                continue
            if only_per_session and not wrapper.is_per_session():
                continue
            tools.append(wrapper.to_tool(get_session=get_session))
        return tools

    @staticmethod
    def create_tools(tool_creator: ToolCreator, toolsets: list[BaseToolSet], get_session: SessionGetter = lambda: []):
        return tool_creator.create_tools(toolsets, get_session)

    @staticmethod
    def create_global_tools_from_names(toolnames: list[str], llm: Optional[BaseLLM]) -> list[BaseTool]:
        return load_tools(toolnames, llm=llm)
