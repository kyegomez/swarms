from swarms.tools.tool import tool
from typing import Dict, Callable, Any, List

ToolBuilder = Callable[[Any], tool]
FuncToolBuilder = Callable[[], ToolBuilder]


class ToolsRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, FuncToolBuilder] = {}

    def register(self, tool_name: str, tool: FuncToolBuilder):
        print(f"will register {tool_name}")
        self.tools[tool_name] = tool

    def build(self, tool_name, config):
        ret = self.tools[tool_name]()(config)
        if isinstance(ret, tool):
            return ret
        raise ValueError(
            "Tool builder {} did not return a Tool instance".format(tool_name)
        )

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())


tools_registry = ToolsRegistry()


def register(tool_name):
    def decorator(tool: FuncToolBuilder):
        tools_registry.register(tool_name, tool)
        return tool

    return decorator


def build_tool(tool_name: str, config: Any) -> tool:
    print(f"will build {tool_name}")
    return tools_registry.build(tool_name, config)


def list_tools() -> List[str]:
    return tools_registry.list_tools()
